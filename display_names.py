"""
display_names.py
----------------
Turns a raw process name into something a person recognises.

    chrome.exe    -> Google Chrome        (app)
    TiWorker.exe  -> Windows Update       (system)
    MsMpEng.exe   -> Microsoft Defender   (background)

This is a PRESENTATION layer only. Nothing here changes what is stored.
The database keeps the raw process name, because the dedupe check, the
knowledge base keys and the analyzers all match on it. Resolving the
display name at render time also means improving the table below fixes
every old advisory, not just new ones.

Three sources, in order of quality:

  1. The curated table. Hand-written, so the wording is yours. Use it
     for anything common, and for the cases where Windows' own name is
     unhelpful ("Antimalware Service Executable" means nothing to a
     normal user).

  2. The executable's FileDescription, read from its version resource.
     This is what Task Manager shows, and it covers everything
     installed without any maintenance from you. Needs pywin32.

  3. Cleaning up the filename. Drops .exe, splits camelCase and
     underscores, capitalises. Crude, but "Some App" beats
     "someApp.exe".

Install the optional dependency for step 2:
    python -m pip install pywin32
"""

import re
from functools import lru_cache

# ================================================================
# Kinds
# ================================================================
# Why this matters: the kind changes what the user should DO.
#   app        - they opened it, so closing or trimming it is fair advice
#   background - runs on its own, usually fine, sometimes worth pausing
#   system     - part of Windows, almost never safe to kill
#
# The TiWorker advisory already uses this idea when it says "this is a
# Windows Update component, not an application". The kind makes that
# available for every process instead of one hand-written case.

APP = "app"
BACKGROUND = "background"
SYSTEM = "system"
UNKNOWN = "unknown"


# ================================================================
# Curated table
# ================================================================

PROCESS_INFO = {
    # Browsers
    "chrome.exe": ("Google Chrome", APP),
    "msedge.exe": ("Microsoft Edge", APP),
    "firefox.exe": ("Firefox", APP),
    "brave.exe": ("Brave", APP),
    "opera.exe": ("Opera", APP),

    # Dev tools
    "code.exe": ("VS Code", APP),
    "devenv.exe": ("Visual Studio", APP),
    "pycharm64.exe": ("PyCharm", APP),
    "idea64.exe": ("IntelliJ IDEA", APP),
    "windowsterminal.exe": ("Windows Terminal", APP),
    "powershell.exe": ("PowerShell", APP),
    "pwsh.exe": ("PowerShell", APP),
    "cmd.exe": ("Command Prompt", APP),
    "python.exe": ("Python", APP),
    "pythonw.exe": ("Python", APP),
    "node.exe": ("Node.js", APP),
    "java.exe": ("Java", APP),
    "git.exe": ("Git", APP),
    "docker desktop.exe": ("Docker Desktop", APP),
    "com.docker.backend.exe": ("Docker", BACKGROUND),
    "postgres.exe": ("PostgreSQL", BACKGROUND),
    "mysqld.exe": ("MySQL", BACKGROUND),

    # Everyday apps
    "ms-teams.exe": ("Microsoft Teams", APP),
    "teams.exe": ("Microsoft Teams", APP),
    "slack.exe": ("Slack", APP),
    "discord.exe": ("Discord", APP),
    "zoom.exe": ("Zoom", APP),
    "spotify.exe": ("Spotify", APP),
    "whatsapp.exe": ("WhatsApp", APP),
    "notion.exe": ("Notion", APP),
    "obs64.exe": ("OBS Studio", APP),
    "vlc.exe": ("VLC", APP),
    "excel.exe": ("Excel", APP),
    "winword.exe": ("Word", APP),
    "powerpnt.exe": ("PowerPoint", APP),
    "outlook.exe": ("Outlook", APP),
    "steam.exe": ("Steam", APP),

    # Background services. Windows' own names for these are the worst
    # offenders, which is why they are all listed by hand.
    "msmpeng.exe": ("Microsoft Defender", BACKGROUND),
    "nissrv.exe": ("Microsoft Defender", BACKGROUND),
    "securityhealthservice.exe": ("Windows Security", BACKGROUND),
    "onedrive.exe": ("OneDrive", BACKGROUND),
    "searchindexer.exe": ("Windows Search indexing", BACKGROUND),
    "searchhost.exe": ("Windows Search", BACKGROUND),
    "wmiprvse.exe": ("Windows management service", BACKGROUND),
    "nvcontainer.exe": ("NVIDIA background service", BACKGROUND),
    "nvdisplay.container.exe": ("NVIDIA display service", BACKGROUND),
    "backgroundtaskhost.exe": ("Background app task", BACKGROUND),
    "runtimebroker.exe": ("Windows app permissions", BACKGROUND),
    "gamebar.exe": ("Xbox Game Bar", BACKGROUND),
    "gamebarftserver.exe": ("Xbox Game Bar", BACKGROUND),

    # Windows internals
    "tiworker.exe": ("Windows Update", SYSTEM),
    "trustedinstaller.exe": ("Windows Update installer", SYSTEM),
    "wuauclt.exe": ("Windows Update", SYSTEM),
    "usoclient.exe": ("Windows Update", SYSTEM),
    "svchost.exe": ("Windows system service", SYSTEM),
    "csrss.exe": ("Windows system process", SYSTEM),
    "lsass.exe": ("Windows security process", SYSTEM),
    "services.exe": ("Windows service manager", SYSTEM),
    "wininit.exe": ("Windows startup process", SYSTEM),
    "winlogon.exe": ("Windows sign-in", SYSTEM),
    "smss.exe": ("Windows session manager", SYSTEM),
    "dwm.exe": ("Desktop Window Manager", SYSTEM),
    "explorer.exe": ("Windows Explorer", SYSTEM),
    "conhost.exe": ("Console window host", SYSTEM),
    "ctfmon.exe": ("Windows text input", SYSTEM),
    "audiodg.exe": ("Windows audio", SYSTEM),
    "fontdrvhost.exe": ("Windows font service", SYSTEM),
    "spoolsv.exe": ("Print spooler", SYSTEM),
    "taskmgr.exe": ("Task Manager", SYSTEM),
    "systemsettings.exe": ("Settings", SYSTEM),
    "startmenuexperiencehost.exe": ("Start menu", SYSTEM),
    "shellexperiencehost.exe": ("Windows shell", SYSTEM),
    "memory compression": ("Windows memory compression", SYSTEM),
    "registry": ("Windows registry", SYSTEM),
    "system": ("Windows kernel", SYSTEM),
    "system idle process": ("Idle", SYSTEM),
}

# Windows reports some FileDescription values that are accurate and
# still useless to a normal reader. If one of these comes back, skip it
# and fall through to cleaning up the filename instead.
UNHELPFUL_DESCRIPTIONS = {
    "host process for windows services",
    "application",
    "windows application",
    "console application",
    "generic host process",
}


# ================================================================
# Windows version resource
# ================================================================

@lru_cache(maxsize=512)
def _file_description(exe_path):
    """
    The FileDescription field from an executable's version resource.

    Cached because the same handful of executables come round on every
    collection cycle, and this touches the filesystem.
    """
    if not exe_path:
        return None
    try:
        import win32api
    except ImportError:
        return None

    try:
        translations = win32api.GetFileVersionInfo(
            exe_path, "\\VarFileInfo\\Translation")
        if not translations:
            return None
        lang, codepage = translations[0]
        key = "\\StringFileInfo\\%04X%04X\\FileDescription" % (lang, codepage)
        description = win32api.GetFileVersionInfo(exe_path, key)
    except Exception:
        # Missing file, no version resource, access denied, malformed
        # resource. All of them mean "fall through to the next source".
        return None

    if not description:
        return None
    description = description.strip()
    if not description or description.lower() in UNHELPFUL_DESCRIPTIONS:
        return None
    return description


# ================================================================
# Filename cleanup
# ================================================================

def _humanise(process_name):
    """someApp_v2.exe -> Some App V2"""
    stem = re.sub(r"\.exe$", "", process_name, flags=re.IGNORECASE)
    stem = re.sub(r"[_\-.]+", " ", stem)
    # Split camelCase and PascalCase into separate words.
    stem = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", stem)
    words = [w for w in stem.split() if w]
    if not words:
        return process_name
    # Leave acronyms alone: NVIDIA stays NVIDIA, not Nvidia.
    return " ".join(w if w.isupper() else w.capitalize() for w in words)


# ================================================================
# Public API
# ================================================================

def describe(process_name, exe_path=None):
    """
    Everything the advisory layer needs about how to name a process.

    Returns a dict:
        raw     the original process name, unchanged
        name    what to show the user
        kind    app | background | system | unknown
        source  curated | metadata | filename  (useful while debugging)
    """
    if not process_name:
        return {"raw": process_name, "name": "Unknown process",
                "kind": UNKNOWN, "source": "filename"}

    key = process_name.strip().lower()

    if key in PROCESS_INFO:
        name, kind = PROCESS_INFO[key]
        return {"raw": process_name, "name": name,
                "kind": kind, "source": "curated"}

    description = _file_description(exe_path)
    if description:
        return {"raw": process_name, "name": description,
                "kind": UNKNOWN, "source": "metadata"}

    return {"raw": process_name, "name": _humanise(process_name),
            "kind": UNKNOWN, "source": "filename"}


def friendly_name(process_name, exe_path=None):
    """Just the display name, when that is all you need."""
    return describe(process_name, exe_path)["name"]


def process_kind(process_name):
    """app | background | system | unknown, from the curated table only."""
    return PROCESS_INFO.get(
        (process_name or "").strip().lower(), (None, UNKNOWN))[1]


def kind_note(kind):
    """
    One line of context for the card, so the user knows whether this is
    something they opened or something Windows is doing on its own.
    """
    return {
        APP: "An app you have open.",
        BACKGROUND: "Runs in the background on its own.",
        SYSTEM: "Part of Windows. Best left alone.",
    }.get(kind)


if __name__ == "__main__":
    samples = ["chrome.exe", "TiWorker.exe", "MsMpEng.exe", "Code.exe",
               "svchost.exe", "someRandomTool_v2.exe", "NVDisplay.Container.exe"]
    width = max(len(s) for s in samples)
    for s in samples:
        d = describe(s)
        print(f"{s:<{width}}  ->  {d['name']:<28} [{d['kind']}, {d['source']}]")