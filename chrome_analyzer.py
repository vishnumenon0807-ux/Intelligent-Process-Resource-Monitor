"""
chrome_analyzer.py
------------------
Chrome analyzer for the Intelligent Process Resource Monitor.

Answers "what is Chrome actually doing right now" with two levels of
detail:

  DEEP  - Chrome was started with --remote-debugging-port, so the
          DevTools Protocol gives us the real tab list: titles, URLs,
          which are extensions, which are playing audio.

  BASIC - No debugging port. We fall back to process inspection:
          renderer process count as a tab estimate, and Chrome's own
          command-line flags to separate renderers from GPU/utility
          processes.

The analyzer never fails hard. If Chrome is not running it returns
None; if DevTools is unreachable it degrades to BASIC. The root cause
engine records which level was used via analyzer_had_deep_data.

Run standalone to see what it detects:
    python chrome_analyzer.py

To enable DEEP mode, close Chrome fully and relaunch it with:
    chrome.exe --remote-debugging-port=9222
"""

import os
import json
import urllib.request
import urllib.error
from urllib.parse import urlparse
import time
import psutil


DEBUG_PORT = int(os.getenv("CHROME_DEBUG_PORT", "9222"))

# Chromium-family executables. Edge and Brave speak the same protocol,
# so the same analyzer works for them.
CHROME_NAMES = {
    "chrome.exe", "chrome",
    "msedge.exe", "msedge",
    "brave.exe", "brave",
    "Google Chrome", "Google Chrome Helper",
}

# Sites where high CPU is expected rather than anomalous. Used to
# explain load, not to suppress it.
MEDIA_DOMAINS = {
    "youtube.com", "netflix.com", "twitch.tv", "primevideo.com",
    "hotstar.com", "disneyplus.com", "spotify.com", "soundcloud.com",
    "vimeo.com", "dailymotion.com", "meet.google.com", "zoom.us",
    "teams.microsoft.com",
}


# ================================================================
# Process-level inspection (always available)
# ================================================================

def _classify_chrome_process(cmdline: list[str] | None) -> str:
    """
    Chrome tags every child process with a --type= flag. The browser
    process has no --type at all.

        (none)           -> browser (the main process)
        --type=renderer  -> a tab or an extension page
        --type=gpu-process
        --type=utility   -> network service, storage, audio service
        --type=crashpad-handler
    """
    if not cmdline:
        return "unknown"
    for arg in cmdline:
        if arg.startswith("--type="):
            return arg.split("=", 1)[1]
    return "browser"


def collect_chrome_processes() -> dict | None:
    """
    Aggregate every Chrome process into one picture.

    Returns None when Chrome is not running, so the caller can skip
    the analyzer entirely.
    """
        # Prime CPU counters — first reading of any process is always 0
    chrome_procs = [p for p in psutil.process_iter(["name"])
                    if p.info.get("name") in CHROME_NAMES]
    for p in chrome_procs:
        try:
            p.cpu_percent(interval=None)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    time.sleep(0.3)
    cpu_count = psutil.cpu_count(logical=True) or 1

    procs = []
    for p in psutil.process_iter(["pid", "ppid", "name", "cmdline",
                                  "cpu_percent", "memory_info"]):
        try:
            info = p.info
            if info.get("name") not in CHROME_NAMES:
                continue
            mem = info.get("memory_info")
            procs.append({
                "pid": info["pid"],
                "ppid": info.get("ppid"),
                "kind": _classify_chrome_process(info.get("cmdline")),
                # psutil reports process CPU as a share of one core, so
                # divide to make it comparable with system-wide CPU%.
                "cpu_percent": (info.get("cpu_percent") or 0.0) / cpu_count,
                "ram_mb": mem.rss / (1024 ** 2) if mem else 0.0,
            })
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            continue

    if not procs:
        return None

    by_kind: dict[str, int] = {}
    for p in procs:
        by_kind[p["kind"]] = by_kind.get(p["kind"], 0) + 1

    renderers = [p for p in procs if p["kind"] == "renderer"]

    return {
        "process_count": len(procs),
        "total_cpu_percent": round(sum(p["cpu_percent"] for p in procs), 2),
        "total_ram_mb": round(sum(p["ram_mb"] for p in procs), 1),
        "renderer_count": len(renderers),
        "renderer_cpu_percent": round(sum(p["cpu_percent"] for p in renderers), 2),
        "renderer_ram_mb": round(sum(p["ram_mb"] for p in renderers), 1),
        "process_kinds": by_kind,
        "top_processes": sorted(procs, key=lambda p: p["cpu_percent"],
                                reverse=True)[:5],
        "pids": [p["pid"] for p in procs],
        "renderer_pids": [p["pid"] for p in renderers],
    }


# ================================================================
# DevTools Protocol (only when Chrome runs with --remote-debugging-port)
# ================================================================

def fetch_devtools_targets(port: int = DEBUG_PORT, timeout: float = 1.0):
    """
    GET /json returns every open target: tabs, extensions, service
    workers. Returns None if the port is closed, which is the normal
    case for a Chrome started without the flag.
    """
    try:
        with urllib.request.urlopen(
            f"http://127.0.0.1:{port}/json", timeout=timeout
        ) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError):
        return None


def _domain_of(url: str) -> str | None:
    try:
        host = urlparse(url).netloc.lower()
        return host[4:] if host.startswith("www.") else host or None
    except Exception:
        return None


def parse_targets(targets: list[dict]) -> dict:
    """Split the raw target list into tabs, extensions, and workers."""
    tabs, extensions, workers = [], [], []

    for t in targets:
        ttype = t.get("type")
        url = t.get("url", "")
        entry = {
            "target_id": t.get("id"),
            "tab_title": (t.get("title") or "")[:500],
            "tab_url": url[:2000],
            "domain": _domain_of(url),
            "target_type": ttype,
        }

        if ttype == "page":
            # DevTools does not expose audio state on /json, so infer
            # likely media from the domain. Marked as an estimate in
            # the evidence dict rather than asserted as fact.
            entry["is_media_site"] = entry["domain"] in MEDIA_DOMAINS
            tabs.append(entry)
        elif ttype in ("background_page", "service_worker") and \
                url.startswith("chrome-extension://"):
            extensions.append(entry)
        elif ttype in ("worker", "shared_worker"):
            workers.append(entry)

    media_tabs = [t for t in tabs if t.get("is_media_site")]

    domains: dict[str, int] = {}
    for t in tabs:
        if t["domain"]:
            domains[t["domain"]] = domains.get(t["domain"], 0) + 1

    return {
        "tab_count": len(tabs),
        "extension_count": len(extensions),
        "worker_count": len(workers),
        "media_tab_count": len(media_tabs),
        "tabs": tabs,
        "extensions": extensions,
        "top_domains": sorted(domains.items(), key=lambda kv: kv[1],
                              reverse=True)[:5],
    }


# ================================================================
# Main analyzer
# ================================================================

def analyze(system_cpu_percent: float | None = None,
            system_ram_percent: float | None = None) -> dict | None:
    """
    Full Chrome analysis.

    Pass the current system-wide CPU/RAM so the analyzer can report
    what share of total load Chrome accounts for — that number is what
    the root cause engine uses for explains_percent.
    """
    procs = collect_chrome_processes()
    if procs is None:
        return None

    result = {
        "analyzer": "chrome",
        "detected": True,
        "depth": "basic",
        **procs,
    }

    targets = fetch_devtools_targets()
    if targets is not None:
        result["depth"] = "deep"
        result.update(parse_targets(targets))
    else:
        # Without DevTools, renderer count is the best tab proxy we
        # have. It over-counts: extensions and some iframes get their
        # own renderer, and Chrome merges renderers under memory
        # pressure. Labelled as an estimate everywhere it is used.
        result["tab_count_estimated"] = procs["renderer_count"]
        result["tabs"] = []
        result["extensions"] = []

    if system_cpu_percent and system_cpu_percent > 0:
        result["share_of_system_cpu"] = round(
            result["total_cpu_percent"] / system_cpu_percent * 100, 1)

    return result


# ================================================================
# Explanation
# ================================================================

def explain(analysis: dict) -> dict:
    """
    Turn the analysis into the fields root_causes expects: a
    human-readable explanation, structured evidence, a confidence
    score, and recommendations.
    """
    if not analysis:
        return {}

    deep = analysis["depth"] == "deep"
    lines = []
    recommendations = []

    cpu = analysis["total_cpu_percent"]
    ram_gb = analysis["total_ram_mb"] / 1024

    lines.append(f"Chrome is using {cpu:.1f}% CPU and {ram_gb:.1f} GB RAM "
                 f"across {analysis['process_count']} processes.")

    if deep:
        tabs = analysis["tab_count"]
        lines.append(f"{tabs} tabs are open.")

        if analysis["extension_count"]:
            lines.append(f"{analysis['extension_count']} extensions are active.")

        if analysis["media_tab_count"]:
            lines.append(f"{analysis['media_tab_count']} tabs are on media sites, "
                         f"which typically sustain high CPU and GPU use.")

        heavy = [d for d, c in analysis["top_domains"] if c >= 3]
        if heavy:
            lines.append("Most tabs are on: " + ", ".join(heavy) + ".")

        if tabs > 15:
            recommendations.append("Close unused tabs — each open tab holds "
                                   "memory even when inactive.")
        if tabs > 10:
            recommendations.append("Enable Chrome Memory Saver "
                                   "(Settings > Performance) to unload "
                                   "background tabs automatically.")
        if analysis["extension_count"] > 5:
            recommendations.append("Review extensions — each one runs its own "
                                   "process.")
        if analysis["media_tab_count"] > 2:
            recommendations.append("Pause background video or audio tabs.")
    else:
        est = analysis.get("tab_count_estimated", 0)
        lines.append(f"Approximately {est} tabs or extension pages are open "
                     f"(estimated from renderer processes; exact tab data "
                     f"requires Chrome's debugging port).")
        if est > 15:
            recommendations.append("Close unused tabs.")
        recommendations.append("Enable Chrome Memory Saver "
                               "(Settings > Performance).")

    # Confidence: high when we have real tab data AND Chrome clearly
    # dominates; lower when we are estimating from process counts.
    share = analysis.get("share_of_system_cpu")
    confidence = 0.95 if deep else 0.70
    if share is not None:
        if share >= 60:
            confidence = min(0.98, confidence + 0.03)
        elif share < 25:
            confidence = max(0.40, confidence - 0.25)

    evidence = {
        "process_count": analysis["process_count"],
        "renderer_count": analysis["renderer_count"],
        "total_cpu_percent": cpu,
        "total_ram_mb": analysis["total_ram_mb"],
        "data_source": "devtools_protocol" if deep else "process_inspection",
    }
    if deep:
        evidence.update({
            "tab_count": analysis["tab_count"],
            "extension_count": analysis["extension_count"],
            "media_tab_count": analysis["media_tab_count"],
            "top_domains": dict(analysis["top_domains"]),
        })
    else:
        evidence["tab_count_estimated"] = analysis.get("tab_count_estimated")

    return {
        "responsible_process": "chrome.exe",
        "analyzer_used": "chrome",
        "analyzer_had_deep_data": deep,
        "explanation": " ".join(lines),
        "evidence": evidence,
        "confidence": round(confidence, 2),
        "explains_percent": share,
        "process_tree_pids": analysis["pids"],
        "recommendations": recommendations,
    }


# ================================================================
# Standalone test
# ================================================================

if __name__ == "__main__":
    # Prime CPU counters — the first reading of any process is always 0.
    for p in psutil.process_iter():
        try:
            p.cpu_percent(interval=None)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass

    import time
    psutil.cpu_percent(interval=None)
    print("sampling for 2 seconds...\n")
    time.sleep(2)

    sys_cpu = psutil.cpu_percent(interval=None)
    sys_ram = psutil.virtual_memory().percent

    analysis = analyze(sys_cpu, sys_ram)

    if analysis is None:
        print("Chrome is not running.")
        raise SystemExit

    print(f"system            : {sys_cpu:.1f}% CPU, {sys_ram:.1f}% RAM")
    print(f"analyzer depth    : {analysis['depth'].upper()}")
    if analysis["depth"] == "basic":
        print(f"                    (start Chrome with "
              f"--remote-debugging-port={DEBUG_PORT} for tab data)")
    print()
    print(f"chrome processes  : {analysis['process_count']}")
    print(f"  by kind         : {analysis['process_kinds']}")
    print(f"chrome CPU        : {analysis['total_cpu_percent']:.1f}%")
    print(f"chrome RAM        : {analysis['total_ram_mb']:.0f} MB")
    if analysis.get("share_of_system_cpu") is not None:
        print(f"share of load     : {analysis['share_of_system_cpu']:.0f}%")

    if analysis["depth"] == "deep":
        print(f"\ntabs              : {analysis['tab_count']}")
        print(f"extensions        : {analysis['extension_count']}")
        print(f"media tabs        : {analysis['media_tab_count']}")
        if analysis["top_domains"]:
            print("top domains       :")
            for domain, count in analysis["top_domains"]:
                print(f"    {count:>3}  {domain}")

    exp = explain(analysis)
    print("\n" + "-" * 60)
    print("EXPLANATION")
    print("-" * 60)
    print(exp["explanation"])
    print(f"\nconfidence: {exp['confidence']}")
    if exp["recommendations"]:
        print("\nrecommendations:")
        for r in exp["recommendations"]:
            print(f"  - {r}")
