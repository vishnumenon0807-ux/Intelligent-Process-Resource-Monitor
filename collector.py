"""
collector.py
------------
The sampling loop. Runs continuously, writing one system_snapshots
row and N process_snapshots rows per tick.

Run:
    python collector.py            # 2-second interval
    python collector.py --interval 1
    python collector.py --interval 5 --duration 300   # stop after 5 min

Stop with Ctrl+C — the session row gets closed cleanly.

Two things this file exists to handle correctly:

  1. psutil's disk and network counters are CUMULATIVE SINCE BOOT.
     A raw value is meaningless on its own; you need
     (current - previous) / elapsed to get a rate. That state is
     kept in DeltaTracker.

  2. cpu_percent() is comparative — it reports usage since the
     PREVIOUS call. The first call always returns 0.0, so every
     process gets primed once before the loop starts.
"""

import time
import argparse
import signal
from datetime import datetime, timezone

import psutil

import db
from hardware_info import collect_session_info


# ----------------------------------------------------------------
# GPU sampling (NVIDIA only; degrades to None elsewhere)
# ----------------------------------------------------------------

class GpuSampler:
    def __init__(self):
        self.handle = None
        try:
            import pynvml
            pynvml.nvmlInit()
            if pynvml.nvmlDeviceGetCount() > 0:
                self.handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                self.nvml = pynvml
        except Exception:
            self.handle = None

    def sample(self) -> dict:
        blank = {
            "gpu_percent": None, "gpu_memory_used_mb": None,
            "gpu_memory_percent": None, "gpu_temp_c": None, "gpu_power_w": None,
        }
        if self.handle is None:
            return blank
        try:
            util = self.nvml.nvmlDeviceGetUtilizationRates(self.handle)
            mem = self.nvml.nvmlDeviceGetMemoryInfo(self.handle)
            temp = self.nvml.nvmlDeviceGetTemperature(
                self.handle, self.nvml.NVML_TEMPERATURE_GPU)
            try:
                power = self.nvml.nvmlDeviceGetPowerUsage(self.handle) / 1000.0
            except Exception:
                power = None
            return {
                "gpu_percent": float(util.gpu),
                "gpu_memory_used_mb": mem.used / (1024 ** 2),
                "gpu_memory_percent": mem.used / mem.total * 100,
                "gpu_temp_c": float(temp),
                "gpu_power_w": power,
            }
        except Exception:
            return blank

    def shutdown(self):
        if self.handle is not None:
            try:
                self.nvml.nvmlShutdown()
            except Exception:
                pass


# ----------------------------------------------------------------
# Cumulative counter -> rate
# ----------------------------------------------------------------

class DeltaTracker:
    """Converts psutil's since-boot counters into per-second rates."""

    def __init__(self):
        self.prev_disk = psutil.disk_io_counters()
        self.prev_net = psutil.net_io_counters()
        self.prev_time = time.monotonic()

    def compute(self) -> dict:
        now_t = time.monotonic()
        elapsed = now_t - self.prev_time
        disk = psutil.disk_io_counters()
        net = psutil.net_io_counters()

        def rate(cur, prev):
            if elapsed <= 0 or cur is None or prev is None:
                return None
            delta = cur - prev
            # Counters reset on reboot / interface reset — ignore negatives
            return delta / elapsed if delta >= 0 else None

        result = {
            "disk_read_bytes": disk.read_bytes if disk else None,
            "disk_write_bytes": disk.write_bytes if disk else None,
            "disk_read_rate_bps": rate(disk.read_bytes, self.prev_disk.read_bytes)
                                  if disk and self.prev_disk else None,
            "disk_write_rate_bps": rate(disk.write_bytes, self.prev_disk.write_bytes)
                                   if disk and self.prev_disk else None,
            "net_sent_bytes": net.bytes_sent if net else None,
            "net_recv_bytes": net.bytes_recv if net else None,
            "net_sent_rate_bps": rate(net.bytes_sent, self.prev_net.bytes_sent)
                                 if net and self.prev_net else None,
            "net_recv_rate_bps": rate(net.bytes_recv, self.prev_net.bytes_recv)
                                 if net and self.prev_net else None,
        }

        self.prev_disk, self.prev_net, self.prev_time = disk, net, now_t
        return result


# ----------------------------------------------------------------
# System-wide sample
# ----------------------------------------------------------------

def sample_system(session_id: int, gpu: GpuSampler, deltas: DeltaTracker) -> dict:
    vm = psutil.virtual_memory()
    sm = psutil.swap_memory()

    # cpu_percent(interval=None) is non-blocking and reports usage
    # since the last call — exactly what a fixed-interval loop wants.
    data = {
        "session_id": session_id,
        "timestamp": datetime.now(timezone.utc),
        "cpu_percent": psutil.cpu_percent(interval=None),
        "ram_percent": vm.percent,
        "ram_used_mb": vm.used / (1024 ** 2),
        "ram_available_mb": vm.available / (1024 ** 2),
        "swap_percent": sm.percent,
        "swap_used_mb": sm.used / (1024 ** 2),
        "process_count": len(psutil.pids()),
        "boot_time": datetime.fromtimestamp(psutil.boot_time(), timezone.utc),
    }

    try:
        f = psutil.cpu_freq()
        data["cpu_freq_mhz"] = f.current if f else None
    except Exception:
        data["cpu_freq_mhz"] = None

    # Root partition usage; adjust for Windows if you want a specific drive
    try:
        data["disk_percent"] = psutil.disk_usage("/").percent
    except Exception:
        try:
            data["disk_percent"] = psutil.disk_usage("C:\\").percent
        except Exception:
            data["disk_percent"] = None

    # Not available on Windows
    try:
        data["load_avg_1m"] = psutil.getloadavg()[0]
    except Exception:
        data["load_avg_1m"] = None

    # CPU temperature: Linux only in psutil
    try:
        temps = psutil.sensors_temperatures()
        for key in ("coretemp", "k10temp", "cpu_thermal"):
            if key in temps and temps[key]:
                data["cpu_temp_c"] = temps[key][0].current
                break
    except Exception:
        pass

    try:
        bat = psutil.sensors_battery()
        if bat:
            data["battery_percent"] = bat.percent
            data["on_battery"] = not bat.power_plugged
    except Exception:
        pass

    data.update(deltas.compute())
    data.update(gpu.sample())
    return data


# ----------------------------------------------------------------
# Per-process sample
# ----------------------------------------------------------------

PROC_ATTRS = ["pid", "ppid", "name", "exe", "cmdline", "username", "status",
              "cpu_percent", "memory_percent", "memory_info", "num_threads",
              "nice", "create_time"]


def sample_processes(cpu_count: int) -> list[dict]:
    rows = []
    for proc in psutil.process_iter(PROC_ATTRS):
        try:
            i = proc.info
            mem = i.get("memory_info")
            cmd = i.get("cmdline")

            rows.append({
                "pid": i.get("pid"),
                "ppid": i.get("ppid"),
                "name": i.get("name"),
                "exe_path": i.get("exe"),
                # cmdline can be enormous (Chrome renderers especially)
                "cmdline": (" ".join(cmd)[:2000] if cmd else None),
                "username": i.get("username"),
                "status": i.get("status"),
                # psutil reports per-process CPU as a share of ONE core,
                # so this can exceed 100 on multithreaded processes.
                # Normalising to whole-system percentage keeps it
                # comparable with system_snapshots.cpu_percent.
                "cpu_percent": (i.get("cpu_percent") or 0.0) / cpu_count,
                "ram_percent": i.get("memory_percent"),
                "ram_rss_mb": mem.rss / (1024 ** 2) if mem else None,
                "ram_vms_mb": mem.vms / (1024 ** 2) if mem else None,
                "thread_count": i.get("num_threads"),
                "nice_value": i.get("nice"),
                "create_time": (datetime.fromtimestamp(i["create_time"], timezone.utc)
                                if i.get("create_time") else None),
                "io_read_bytes": None,      # needs admin on Windows; skip for now
                "io_write_bytes": None,
                "open_files_count": None,   # expensive syscall, off by default
                "gpu_percent": None,
            })
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            # Processes die mid-iteration constantly. Expected, not an error.
            continue
    return rows


# ----------------------------------------------------------------
# Main loop
# ----------------------------------------------------------------

running = True


def _stop(signum, frame):
    global running
    running = False
    print("\nshutting down...")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--interval", type=float, default=2.0,
                    help="seconds between samples")
    ap.add_argument("--duration", type=float, default=None,
                    help="stop after N seconds (default: run forever)")
    ap.add_argument("--no-processes", action="store_true",
                    help="skip per-process collection")
    args = ap.parse_args()

    signal.signal(signal.SIGINT, _stop)

    cpu_count = psutil.cpu_count(logical=True) or 1

    db.init_db()
    gpu = GpuSampler()
    session_id = db.insert_session(collect_session_info())
    print(f"session {session_id} | interval {args.interval}s | "
          f"gpu {'yes' if gpu.handle else 'no'}")

    # Prime the CPU counters. Without this the first sample is all zeros.
    psutil.cpu_percent(interval=None)
    for proc in psutil.process_iter():
        try:
            proc.cpu_percent(interval=None)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    time.sleep(1)

    deltas = DeltaTracker()
    started = time.monotonic()
    ticks = 0

    try:
        while running:
            tick_start = time.monotonic()

            sys_data = sample_system(session_id, gpu, deltas)
            snapshot_id = db.insert_system_snapshot(sys_data)

            n_procs = 0
            if not args.no_processes:
                procs = sample_processes(cpu_count)
                db.insert_process_snapshots(snapshot_id, procs)
                n_procs = len(procs)

            ticks += 1
            cpu = sys_data.get("cpu_percent")
            ram = sys_data.get("ram_percent")
            gpu_pct = sys_data.get("gpu_percent")
            print(f"[{ticks:>5}] snapshot {snapshot_id}  "
                  f"cpu {cpu:5.1f}%  ram {ram:5.1f}%  "
                  f"gpu {gpu_pct if gpu_pct is not None else '--':>5}  "
                  f"procs {n_procs}")

            if args.duration and (time.monotonic() - started) >= args.duration:
                break

            # Sleep the REMAINDER of the interval, not the full interval.
            # Otherwise collection time compounds and your samples drift
            # further apart over hours — which quietly corrupts the
            # fixed-timestep assumption the LSTM depends on.
            elapsed = time.monotonic() - tick_start
            time.sleep(max(0.0, args.interval - elapsed))

    finally:
        db.close_session(session_id)
        gpu.shutdown()
        db.close_db()
        print(f"session {session_id} closed after {ticks} samples")


if __name__ == "__main__":
    main()
