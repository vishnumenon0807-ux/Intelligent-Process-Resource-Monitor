"""
generic_analyzer.py
-------------------
Fallback analyzer for any process without a dedicated module.

It cannot inspect application internals, so it works entirely from
what process_snapshots already recorded: the process tree, thread
counts, memory growth over recent samples, and how long the process
has been running.

Weaker explanations than the Chrome analyzer, but always available —
which is what keeps the root cause engine useful for the long tail
of processes nobody writes a specific analyzer for.

The returned dict matches the shape chrome_analyzer.explain()
produces, so the engine can treat every analyzer identically.
"""

from datetime import datetime, timezone


# Processes whose high usage is usually expected rather than
# anomalous. The engine records these as suppressed instead of
# raising an alert.
EXPECTED_LOAD = {
    "msmpeng.exe": ("antivirus", "Windows Defender is scanning"),
    "antimalwareservicehost.exe": ("antivirus", "antivirus scan in progress"),
    "onedrive.exe": ("cloud_sync", "OneDrive is syncing files"),
    "googledrivefs.exe": ("cloud_sync", "Google Drive is syncing"),
    "dropbox.exe": ("cloud_sync", "Dropbox is syncing"),
    "tiworker.exe": ("os_update", "Windows Update is installing components"),
    "trustedinstaller.exe": ("os_update", "Windows is servicing system files"),
    "searchindexer.exe": ("indexing", "Windows Search is indexing files"),
    "wuauclt.exe": ("os_update", "Windows Update is running"),
    "compattelrunner.exe": ("os_update", "Windows telemetry task is running"),
}

# Known heavy applications where load is normal during active use
KNOWN_HEAVY = {
    "code.exe": "VS Code",
    "devenv.exe": "Visual Studio",
    "pycharm64.exe": "PyCharm",
    "idea64.exe": "IntelliJ IDEA",
    "photoshop.exe": "Photoshop",
    "blender.exe": "Blender",
    "ffmpeg.exe": "FFmpeg",
    "docker desktop.exe": "Docker Desktop",
    "com.docker.backend.exe": "Docker",
    "python.exe": "a Python process",
    "node.exe": "a Node.js process",
    "java.exe": "a Java process",
}


def _humanize_runtime(create_time) -> str | None:
    if not create_time:
        return None
    if create_time.tzinfo is None:
        create_time = create_time.replace(tzinfo=timezone.utc)
    seconds = (datetime.now(timezone.utc) - create_time).total_seconds()
    if seconds < 0:
        return None
    if seconds < 90:
        return f"{int(seconds)} seconds"
    if seconds < 5400:
        return f"{int(seconds // 60)} minutes"
    if seconds < 172800:
        return f"{int(seconds // 3600)} hours"
    return f"{int(seconds // 86400)} days"


def analyze(tree: dict, system_cpu: float | None,
            memory_trend: dict | None = None) -> dict:
    """
    `tree` is the aggregated process tree from the engine:
        name, root_pid, pids, cpu_percent, ram_mb, process_count,
        thread_count, create_time

    `memory_trend` is optional: {"growth_mb": float, "samples": int}
    computed by the engine from recent snapshots.
    """
    name = (tree.get("name") or "").lower()
    display = KNOWN_HEAVY.get(name, tree.get("name") or "An unknown process")

    lines = []
    recommendations = []
    evidence = {
        "process_count": tree.get("process_count", 1),
        "thread_count": tree.get("thread_count"),
        "cpu_percent": round(tree.get("cpu_percent", 0.0), 2),
        "ram_mb": round(tree.get("ram_mb", 0.0), 1),
        "data_source": "process_inspection",
    }

    # --- expected-load processes ---
    expected = EXPECTED_LOAD.get(name)
    if expected:
        category, phrase = expected
        lines.append(f"{phrase.capitalize()}.")
        lines.append(f"It is using {tree['cpu_percent']:.1f}% CPU and "
                     f"{tree['ram_mb']:.0f} MB RAM.")
        lines.append("This is a scheduled background task, not a fault.")
        evidence["task_category"] = category
        evidence["is_expected"] = True
        recommendations.append("No action needed — this usually finishes on its own.")
        recommendations.append("If it runs at inconvenient times, reschedule it "
                               "in the application's settings.")
        return {
            "responsible_process": tree.get("name"),
            "analyzer_used": "background_task",
            "analyzer_had_deep_data": False,
            "explanation": " ".join(lines),
            "evidence": evidence,
            "confidence": 0.85,
            "recommendations": recommendations,
            "is_expected": True,
        }

    # --- general case ---
    lines.append(f"{display} is using {tree['cpu_percent']:.1f}% CPU "
                 f"and {tree['ram_mb']:.0f} MB RAM.")

    n_proc = tree.get("process_count", 1)
    if n_proc > 1:
        lines.append(f"It is running as {n_proc} processes.")

    threads = tree.get("thread_count")
    if threads and threads > 100:
        lines.append(f"It has {threads} threads active, which suggests "
                     f"heavy parallel work.")
        evidence["high_thread_count"] = True

    runtime = _humanize_runtime(tree.get("create_time"))
    if runtime:
        lines.append(f"It has been running for {runtime}.")
        evidence["runtime"] = runtime

    if memory_trend and memory_trend.get("growth_mb", 0) > 100:
        growth = memory_trend["growth_mb"]
        lines.append(f"Its memory use has grown by {growth:.0f} MB over "
                     f"the last {memory_trend['samples']} samples, which can "
                     f"indicate a leak or a large in-memory operation.")
        evidence["memory_growth_mb"] = round(growth, 1)
        recommendations.append("Watch its memory — restart it if growth continues.")

    lines.append("No application-specific analyzer is available for this "
                 "process, so the explanation is based on process metrics only.")

    if tree["cpu_percent"] > 40:
        recommendations.append(f"Close {display} if it is not in use.")
    if tree["ram_mb"] > 2048:
        recommendations.append(f"{display} is holding over 2 GB — restarting "
                               f"it will release that memory.")
    if not recommendations:
        recommendations.append("Monitor this process; no action needed yet.")

    share = None
    if system_cpu and system_cpu > 0:
        share = round(tree["cpu_percent"] / system_cpu * 100, 1)

    # Confidence is deliberately moderate: we identified the process
    # but cannot see inside it.
    confidence = 0.55
    if share is not None:
        if share >= 60:
            confidence = 0.70
        elif share < 25:
            confidence = 0.40

    return {
        "responsible_process": tree.get("name"),
        "analyzer_used": "generic",
        "analyzer_had_deep_data": False,
        "explanation": " ".join(lines),
        "evidence": evidence,
        "confidence": round(confidence, 2),
        "explains_percent": share,
        "recommendations": recommendations,
        "is_expected": False,
    }
