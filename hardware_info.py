"""
hardware_info.py
----------------
Collects static hardware/OS facts once at agent startup to populate
the `monitoring_sessions` table.

Install:
    pip install psutil py-cpuinfo nvidia-ml-py
    pip install wmi          # Windows only, for non-NVIDIA GPUs

Every field is optional by design. Hardware detection fails in a lot
of environments (no GPU, locked-down WMI, virtualised CPU), so each
probe is isolated and returns None on failure rather than crashing
the agent at startup.
"""

import platform
import socket
import psutil


AGENT_VERSION = "0.1.0"


# ----------------------------------------------------------------
# CPU
# ----------------------------------------------------------------

def _cpu_model() -> str | None:
    """
    Human-readable CPU name, e.g. "AMD Ryzen 7 5800H".

    platform.processor() is unreliable — on Linux it often returns
    just "x86_64", and on Windows it returns a family/model string
    rather than the marketing name. py-cpuinfo normalises this
    across platforms, so try it first.
    """
    try:
        import cpuinfo
        brand = cpuinfo.get_cpu_info().get("brand_raw")
        if brand:
            return brand
    except Exception:
        pass

    # Fallbacks per platform
    system = platform.system()
    try:
        if system == "Linux":
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if line.startswith("model name"):
                        return line.split(":", 1)[1].strip()

        elif system == "Darwin":
            import subprocess
            return subprocess.check_output(
                ["sysctl", "-n", "machdep.cpu.brand_string"], text=True
            ).strip()

        elif system == "Windows":
            import winreg
            key = winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE,
                r"HARDWARE\DESCRIPTION\System\CentralProcessor\0",
            )
            return winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
    except Exception:
        pass

    return platform.processor() or None


def collect_cpu() -> dict:
    freq = None
    try:
        f = psutil.cpu_freq()
        freq = f.max or f.current if f else None
    except Exception:
        pass

    return {
        "cpu_model": _cpu_model(),
        "cpu_cores": psutil.cpu_count(logical=False),    # physical
        "cpu_threads": psutil.cpu_count(logical=True),   # logical
        "cpu_base_freq_mhz": freq,
    }


# ----------------------------------------------------------------
# Memory
# ----------------------------------------------------------------

def collect_memory() -> dict:
    vm = psutil.virtual_memory()
    sm = psutil.swap_memory()
    return {
        "total_ram_mb": round(vm.total / (1024 ** 2), 1),
        "total_swap_mb": round(sm.total / (1024 ** 2), 1),
    }


# ----------------------------------------------------------------
# GPU
# ----------------------------------------------------------------

def collect_gpu() -> dict:
    """
    NVIDIA via NVML gives the richest data (model, VRAM, driver).
    Falls back to WMI on Windows, which reports any GPU vendor but
    with less detail. Integrated-only machines usually land here.
    """
    # --- NVIDIA (best case) ---
    try:
        import pynvml
        pynvml.nvmlInit()
        if pynvml.nvmlDeviceGetCount() > 0:
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)

            name = pynvml.nvmlDeviceGetName(handle)
            if isinstance(name, bytes):          # older pynvml returns bytes
                name = name.decode()

            mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
            driver = pynvml.nvmlSystemGetDriverVersion()
            if isinstance(driver, bytes):
                driver = driver.decode()

            result = {
                "gpu_model": name,
                "total_gpu_memory_mb": round(mem.total / (1024 ** 2), 1),
                "gpu_driver_version": driver,
                "gpu_vendor": "NVIDIA",
            }
            pynvml.nvmlShutdown()
            return result
        pynvml.nvmlShutdown()
    except Exception:
        pass

    # --- Windows WMI fallback (any vendor) ---
    if platform.system() == "Windows":
        try:
            import wmi
            for gpu in wmi.WMI().Win32_VideoController():
                vram = None
                if gpu.AdapterRAM:
                    # AdapterRAM is signed 32-bit and wraps above 4 GB;
                    # treat implausible values as unknown.
                    raw = int(gpu.AdapterRAM)
                    if raw > 0:
                        vram = round(raw / (1024 ** 2), 1)
                return {
                    "gpu_model": gpu.Name,
                    "total_gpu_memory_mb": vram,
                    "gpu_driver_version": gpu.DriverVersion,
                    "gpu_vendor": (gpu.AdapterCompatibility or "").strip() or None,
                }
        except Exception:
            pass

    # --- Linux fallback ---
    if platform.system() == "Linux":
        try:
            import subprocess
            out = subprocess.check_output(["lspci"], text=True)
            for line in out.splitlines():
                if "VGA compatible controller" in line or "3D controller" in line:
                    return {
                        "gpu_model": line.split(":", 2)[-1].strip(),
                        "total_gpu_memory_mb": None,
                        "gpu_driver_version": None,
                        "gpu_vendor": None,
                    }
        except Exception:
            pass

    return {
        "gpu_model": None,
        "total_gpu_memory_mb": None,
        "gpu_driver_version": None,
        "gpu_vendor": None,
    }


# ----------------------------------------------------------------
# OS / host
# ----------------------------------------------------------------

def collect_host() -> dict:
    return {
        "hostname": socket.gethostname(),
        "os_name": platform.system(),                 # Windows / Linux / Darwin
        "os_version": platform.version(),
        "os_release": platform.release(),
        "architecture": platform.machine(),
        "python_version": platform.python_version(),
        "agent_version": AGENT_VERSION,
    }


# ----------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------

def collect_session_info() -> dict:
    """Everything needed for one monitoring_sessions row."""
    info = {}
    info.update(collect_host())
    info.update(collect_cpu())
    info.update(collect_memory())
    info.update(collect_gpu())
    return info


if __name__ == "__main__":
    from pprint import pprint
    pprint(collect_session_info())
