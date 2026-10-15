import subprocess

def check_status() -> None:
    dirty = subprocess.run(["git", "status", "--porcelain"], capture_output=True,
    text=True).stdout.strip()
    if dirty:
        raise SystemExit("Есть незакоммиченные изменения — итоговый запуск невозможен:\n" + dirty)

def get_commit_hash() -> str:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], 
        capture_output=True,
        text=True
    ).stdout.strip()
    return commit
