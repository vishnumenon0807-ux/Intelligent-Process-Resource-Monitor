"""
IPRM :: recommendation engine.

Turns (anomaly + root cause + process snapshot) into a Recommendation:
    what the issue is, why it is happening, and what the user can do.

Nothing in this module touches a process. It reads state and produces text.

Wire it into the RCA engine's output:

    rec = RECOMMENDER.advise(state)
    if rec:
        save(conn, rec)
"""

from __future__ import annotations

import fnmatch
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

# --------------------------------------------------------------------------
# Input
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class SystemState:
    """Everything the rules are allowed to look at."""

    # process under suspicion (None for host-wide findings)
    pid: int | None = None
    name: str = ""
    create_time: float | None = None
    cpu_percent: float = 0.0            # 0-100 normalised across all cores
    memory_mb: float = 0.0
    memory_growth_mb_per_min: float = 0.0
    io_read_mb_per_s: float = 0.0
    io_write_mb_per_s: float = 0.0
    child_count: int = 0
    thread_count: int = 0
    responding: bool = True
    uptime_minutes: float = 0.0

    # host
    host_cpu_percent: float = 0.0
    host_mem_percent: float = 0.0
    host_swap_percent: float = 0.0
    host_disk_busy_percent: float = 0.0
    process_count: int = 0

    # from the existing pipeline
    anomaly_id: int | None = None
    cause_label: str | None = None
    cause_text: str | None = None
    severity: float | None = None       # Isolation Forest score

    def as_metrics(self) -> dict[str, Any]:
        return {
            "cpu_percent": round(self.cpu_percent, 2),
            "memory_mb": round(self.memory_mb, 1),
            "memory_growth_mb_per_min": round(self.memory_growth_mb_per_min, 2),
            "child_count": self.child_count,
            "host_cpu_percent": round(self.host_cpu_percent, 1),
            "host_mem_percent": round(self.host_mem_percent, 1),
            "host_disk_busy_percent": round(self.host_disk_busy_percent, 1),
        }


# --------------------------------------------------------------------------
# Output
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class Recommendation:
    rule_id: str
    category: str                       # cpu | memory | disk | startup
    confidence: float
    title: str
    diagnosis: str
    cause: str
    suggestions: tuple[str, ...]
    state: SystemState
    primary_metric: str                 # which metric to recheck later


# --------------------------------------------------------------------------
# Rules
#
# Each rule owns a predicate and three templates. Templates are formatted
# against the SystemState, so `{name}` and `{cpu_percent:.0f}` work directly.
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class Rule:
    id: str
    category: str
    priority: int                       # lower wins when several match
    match: Callable[[SystemState], bool]
    title: str
    diagnosis: str
    cause: str
    suggestions: tuple[str, ...]
    base_confidence: float = 0.6
    primary_metric: str = "cpu_percent"


def _is(state: SystemState, *patterns: str) -> bool:
    n = (state.name or "").lower()
    return any(fnmatch.fnmatch(n, p) for p in patterns)


BROWSERS = ("chrome.exe", "msedge.exe", "brave.exe", "firefox.exe", "opera.exe")
BUILD_TOOLS = ("node.exe", "java.exe", "msbuild.exe", "cl.exe", "gcc.exe",
               "cargo.exe", "rustc.exe", "python.exe", "dotnet.exe")


RULES: tuple[Rule, ...] = (
    # ---------------------------------------------------------- browser
    Rule(
        id="browser_tab_sprawl",
        category="memory",
        priority=10,
        match=lambda s: _is(s, *BROWSERS) and s.child_count >= 15,
        title="{name} is running {child_count} renderer processes",
        diagnosis=(
            "{name} has spawned {child_count} child processes and is holding "
            "{memory_mb:.0f} MB. Each open tab, extension and iframe gets its own "
            "process under site isolation."
        ),
        cause=(
            "Tab count rather than any single heavy page. Memory scales roughly "
            "linearly with open tabs, and background tabs are not fully freed "
            "until the browser is under pressure."
        ),
        suggestions=(
            "Open the browser's own task manager (Shift+Esc in Chrome/Edge) to see "
            "which tabs are heaviest.",
            "Close or bookmark tabs you are not actively using.",
            "Enable Memory Saver: Settings > Performance > Memory Saver.",
            "Audit extensions -- they run continuously, not only when you use them.",
        ),
        base_confidence=0.8,
        primary_metric="memory_mb",
    ),
    Rule(
        id="browser_cpu_hog",
        category="cpu",
        priority=15,
        match=lambda s: _is(s, *BROWSERS) and s.cpu_percent >= 25,
        title="{name} is using {cpu_percent:.0f}% CPU",
        diagnosis=(
            "{name} is holding {cpu_percent:.0f}% of total CPU across "
            "{child_count} processes."
        ),
        cause=(
            "Usually one misbehaving tab -- an autoplaying video, a stuck ad script, "
            "an animation loop, or a page in a reload cycle. Site isolation means "
            "the cost shows up in a renderer child, not the parent."
        ),
        suggestions=(
            "Press Shift+Esc to open the browser task manager and sort by CPU.",
            "Close or reload the tab at the top of that list.",
            "If it returns immediately, test in an incognito window to rule out an "
            "extension.",
            "Check whether hardware acceleration is off: Settings > System.",
        ),
        base_confidence=0.75,
    ),

    # ----------------------------------------------------------- memory
    Rule(
        id="memory_leak_suspected",
        category="memory",
        priority=20,
        match=lambda s: (s.memory_growth_mb_per_min >= 5
                         and s.uptime_minutes >= 30
                         and s.memory_mb >= 1024),
        title="{name} memory is growing steadily",
        diagnosis=(
            "{name} is at {memory_mb:.0f} MB and has grown by roughly "
            "{memory_growth_mb_per_min:.1f} MB/min over the last window, with no "
            "plateau after {uptime_minutes:.0f} minutes of uptime."
        ),
        cause=(
            "Monotonic growth without a plateau is the signature of a leak rather "
            "than a cache. A cache stabilises once it fills; a leak does not. "
            "Confirming needs a longer window than one snapshot."
        ),
        suggestions=(
            "Restart {name} -- this reclaims the memory immediately and confirms "
            "growth restarts from a low baseline.",
            "Note how long it took to reach {memory_mb:.0f} MB; if the same curve "
            "reappears after restart, it is a leak and worth reporting upstream.",
            "Check whether the application has an update available; leaks are "
            "common bug-fix targets.",
        ),
        base_confidence=0.55,       # deliberately low -- one window is weak evidence
        primary_metric="memory_mb",
    ),
    Rule(
        id="host_memory_pressure",
        category="memory",
        priority=25,
        match=lambda s: s.host_mem_percent >= 88 or s.host_swap_percent >= 40,
        title="System memory is nearly exhausted",
        diagnosis=(
            "Physical memory is {host_mem_percent:.0f}% used with the page file at "
            "{host_swap_percent:.0f}%, across {process_count} running processes."
        ),
        cause=(
            "Total demand from all running applications exceeds installed RAM, so "
            "Windows is paging to disk. The slowdown you feel is disk latency "
            "standing in for memory access, not CPU load."
        ),
        suggestions=(
            "Sort by memory in Task Manager and close the largest applications you "
            "are not using.",
            "Restart any application that has been open for days -- long-lived "
            "processes accumulate the most.",
            "If this recurs at normal workloads, the machine is under-provisioned "
            "for how you use it rather than misbehaving.",
        ),
        base_confidence=0.85,
        primary_metric="host_mem_percent",
    ),

    # -------------------------------------------------------------- cpu
    Rule(
        id="windows_update_worker",
        category="cpu",
        priority=30,
        match=lambda s: _is(s, "tiworker.exe", "trustedinstaller.exe",
                            "mousocoreworker.exe", "usoclient.exe", "wuauclt.exe"),
        title="Windows Update is working in the background",
        diagnosis=(
            "{name} is at {cpu_percent:.0f}% CPU. This is a Windows Update "
            "component, not an application."
        ),
        cause=(
            "Update download, or servicing-stack cleanup of the component store. "
            "It is expected behaviour and it finishes on its own, but it is not "
            "scheduled around your working hours by default."
        ),
        suggestions=(
            "Leave it running -- interrupting servicing can leave the component "
            "store in a bad state.",
            "Set Active Hours so this happens outside your working day: "
            "Settings > Windows Update > Advanced options.",
            "If it runs for hours across multiple days, run "
            "`DISM /Online /Cleanup-Image /RestoreHealth` from an elevated prompt.",
        ),
        base_confidence=0.9,
    ),
    Rule(
        id="search_indexer",
        category="disk",
        priority=35,
        match=lambda s: _is(s, "searchindexer.exe", "searchprotocolhost.exe",
                            "searchfilterhost.exe"),
        title="Windows Search is rebuilding its index",
        diagnosis=(
            "{name} is at {cpu_percent:.0f}% CPU and "
            "{io_read_mb_per_s:.1f} MB/s read, with the disk "
            "{host_disk_busy_percent:.0f}% busy."
        ),
        cause=(
            "A full index rebuild, usually triggered by a Windows update, a new "
            "large folder being added to indexed locations, or a corrupted index. "
            "It is one-off but can run for hours."
        ),
        suggestions=(
            "Let it finish once -- interrupting it means starting over.",
            "Narrow what gets indexed: Indexing Options > Modify, and deselect "
            "large source-code or build directories.",
            "Add your project folders to the exclusion list -- build output "
            "generates thousands of files the indexer will chase.",
        ),
        base_confidence=0.85,
        primary_metric="host_disk_busy_percent",
    ),
    Rule(
        id="antivirus_scan",
        category="disk",
        priority=36,
        match=lambda s: _is(s, "msmpeng.exe", "nissrv.exe", "antimalwareservice*"),
        title="Antivirus scanning is consuming resources",
        diagnosis=(
            "{name} is at {cpu_percent:.0f}% CPU with "
            "{io_read_mb_per_s:.1f} MB/s read."
        ),
        cause=(
            "A scheduled or on-access scan. On-access scanning spikes hardest "
            "against directories with high file churn -- build outputs, "
            "node_modules, package caches."
        ),
        suggestions=(
            "Move scheduled scans to off-hours via Task Scheduler > Microsoft > "
            "Windows > Windows Defender.",
            "Exclude your build and dependency directories: Windows Security > "
            "Virus & threat protection > Manage settings > Exclusions.",
            "Do not disable real-time protection as a fix.",
        ),
        base_confidence=0.8,
    ),
    Rule(
        id="build_parallelism",
        category="cpu",
        priority=40,
        match=lambda s: _is(s, *BUILD_TOOLS) and s.cpu_percent >= 60,
        title="{name} is saturating the CPU",
        diagnosis=(
            "{name} is at {cpu_percent:.0f}% CPU across {thread_count} threads."
        ),
        cause=(
            "Build and compile tools default to using every available core. That "
            "is the fastest setting for the build and the worst one for everything "
            "else running at the same time."
        ),
        suggestions=(
            "If this is an intentional build, it is working as designed -- wait it out.",
            "To keep the machine usable during builds, cap parallelism: `-j N` for "
            "make, `/m:N` for MSBuild, `--jobs N` for cargo.",
            "Leave at least two cores free so the UI stays responsive.",
        ),
        base_confidence=0.7,
    ),
    Rule(
        id="process_not_responding",
        category="cpu",
        priority=45,
        match=lambda s: not s.responding,
        title="{name} has stopped responding",
        diagnosis=(
            "{name} (PID {pid}) is not pumping its message loop. CPU is at "
            "{cpu_percent:.0f}%."
        ),
        cause=(
            "Either a blocked main thread waiting on I/O or a lock, or an infinite "
            "loop. High CPU while unresponsive points to a loop; near-zero CPU "
            "points to a deadlock or a stalled network call."
        ),
        suggestions=(
            "Wait 30 seconds first -- many hangs are slow I/O and recover.",
            "If it has unsaved work, wait longer before forcing it closed.",
            "If it stays frozen, close it from Task Manager and reopen.",
        ),
        base_confidence=0.75,
    ),
    Rule(
        id="startup_bloat",
        category="startup",
        priority=50,
        match=lambda s: s.process_count >= 250 and s.host_cpu_percent >= 40,
        title="Unusually high number of running processes",
        diagnosis=(
            "{process_count} processes are running with the host at "
            "{host_cpu_percent:.0f}% CPU and {host_mem_percent:.0f}% memory."
        ),
        cause=(
            "Accumulated startup entries. Applications commonly add auto-start "
            "helpers, updaters and tray agents on install, and these are rarely "
            "removed afterwards."
        ),
        suggestions=(
            "Review Task Manager > Startup apps and disable anything with a high "
            "startup impact that you do not need at boot.",
            "Updaters and tray helpers are the usual candidates -- they can be "
            "started manually when needed.",
            "Disable rather than uninstall first, so you can reverse it.",
        ),
        base_confidence=0.6,
        primary_metric="host_cpu_percent",
    ),

    # -------------------------------------------------------- fallback
    Rule(
        id="generic_cpu_anomaly",
        category="cpu",
        priority=900,
        match=lambda s: s.cpu_percent >= 50,
        title="{name} is using {cpu_percent:.0f}% CPU",
        diagnosis=(
            "{name} (PID {pid}) is at {cpu_percent:.0f}% CPU and "
            "{memory_mb:.0f} MB, well above its own baseline."
        ),
        cause=(
            "No specific pattern matched, so the cause is not identified. The "
            "process has been running {uptime_minutes:.0f} minutes."
        ),
        suggestions=(
            "Check whether this matches something you just started -- an export, "
            "an import, a sync.",
            "If nothing you did explains it, restart the application.",
            "If it recurs at the same time each day, look for a scheduled task in "
            "Task Scheduler.",
        ),
        base_confidence=0.4,
    ),
)


# --------------------------------------------------------------------------
# Engine
# --------------------------------------------------------------------------
class Recommender:
    def __init__(self, rules: Sequence[Rule] = RULES) -> None:
        self.rules = tuple(sorted(rules, key=lambda r: r.priority))

    def advise(self, state: SystemState) -> Recommendation | None:
        for rule in self.rules:
            try:
                if not rule.match(state):
                    continue
            except Exception:
                continue                    # a broken rule must not kill the pass
            return self._build(rule, state)
        return None

    def advise_all(self, state: SystemState, limit: int = 3) -> list[Recommendation]:
        """Several findings can be true at once (leaking AND host under pressure)."""
        out: list[Recommendation] = []
        for rule in self.rules:
            if len(out) >= limit:
                break
            try:
                if rule.match(state):
                    out.append(self._build(rule, state))
            except Exception:
                continue
        return out

    def _build(self, rule: Rule, state: SystemState) -> Recommendation:
        ctx = {**state.__dict__}
        ctx["name"] = state.name or "This process"
        ctx["pid"] = state.pid if state.pid is not None else "?"

        def fmt(t: str) -> str:
            try:
                return t.format(**ctx)
            except (KeyError, ValueError, IndexError):
                return t

        return Recommendation(
            rule_id=rule.id,
            category=rule.category,
            confidence=self._confidence(rule, state),
            title=fmt(rule.title),
            diagnosis=fmt(rule.diagnosis),
            cause=fmt(rule.cause),
            suggestions=tuple(fmt(s) for s in rule.suggestions),
            state=state,
            primary_metric=rule.primary_metric,
        )

    @staticmethod
    def _confidence(rule: Rule, state: SystemState) -> float:
        """
        Blend the rule's own confidence with the detector's severity. A rule that
        pattern-matched cleanly on a process the detector was also confident about
        deserves a higher number than one that only just crossed a threshold.
        """
        c = rule.base_confidence
        if state.severity is not None:
            c = 0.7 * c + 0.3 * max(0.0, min(1.0, state.severity))
        return round(min(0.95, c), 2)


RECOMMENDER = Recommender()