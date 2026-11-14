"""Check an explicit source release list, optionally building a ZIP from it.

This is a lightweight accidental-disclosure check, not a security guarantee.
Archive construction never recursively copies the working directory.
"""
import argparse
import hashlib
import ipaddress
from pathlib import Path
import re
import zipfile


FILES = (
    ".gitignore", "LICENSE", "README.md", "requirements.txt",
    "run.py", "analyze.py", "render.py",
    "adaptive_mpc/__init__.py", "adaptive_mpc/task.py",
    "adaptive_mpc/geometry.py", "adaptive_mpc/dynamics.py",
    "adaptive_mpc/controller.py", "adaptive_mpc/simulation.py",
    "tests/test_core.py", "tools/check_release.py",
    "docs/method.md", "docs/related_work.md", "docs/provenance.md",
    "docs/demo.gif",
    "results/summary.json", "results/analysis.json", "results/success.png",
)
LOCAL_DIRECTORIES = {".git", ".venv", "outputs", "dist", "__pycache__", ".pytest_cache"}
BINARY_FILES = {"docs/demo.gif", "results/success.png"}
SENSITIVE = {
    "private key": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    "GitHub token": re.compile(r"\b(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,})\b"),
    "cloud access key": re.compile(r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b"),
    "API key": re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b"),
    "personal filesystem path": re.compile(r"/(?:Users|home)/[^\s/]+/"),
    "email or login address": re.compile(r"\b[A-Za-z0-9_.+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
    "credential-bearing URL": re.compile(r"[a-z]+://[^\s/]+:[^\s/]+@", re.I),
    "assigned credential": re.compile(r"(?i)\b(?:api_key|password|access_token|secret_key)\s*=\s*['\"][^'\"]+['\"]"),
}
IPV4 = re.compile(r"(?<![\w.])(?:\d{1,3}\.){3}\d{1,3}(?![\w.])")


def inspect(root):
    errors, total = [], 0
    expected = set(FILES)
    for path in root.rglob("*"):
        relative = path.relative_to(root)
        if any(part in LOCAL_DIRECTORIES for part in relative.parts):
            continue
        if path.is_symlink():
            errors.append(f"Symlink is not allowed: {relative}")
        elif path.is_file() and relative.as_posix() not in expected:
            errors.append(f"Unexpected file: {relative}")
    for name in FILES:
        path = root / name
        if not path.is_file() or path.is_symlink():
            errors.append(f"Missing or unsafe release file: {name}")
            continue
        payload = path.read_bytes()
        total += len(payload)
        if name in BINARY_FILES:
            signature = b"GIF8" if name.endswith(".gif") else b"\x89PNG\r\n\x1a\n"
            if not payload.startswith(signature):
                errors.append(f"Unexpected binary format: {name}")
            continue
        try:
            text = payload.decode("utf-8")
        except UnicodeDecodeError:
            errors.append(f"Non-UTF-8 source: {name}")
            continue
        for label, pattern in SENSITIVE.items():
            if pattern.search(text):
                errors.append(f"Possible {label}: {name}")  # Never print the match.
        for match in IPV4.findall(text):
            try:
                ipaddress.ip_address(match)
            except ValueError:
                continue
            errors.append(f"Possible network address: {name}")
    return errors, total


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--archive", type=Path)
    args = p.parse_args()
    root = Path(__file__).resolve().parents[1]
    errors, total = inspect(root)
    if errors:
        raise SystemExit("Release check failed:\n" + "\n".join(errors))
    print(f"PASS: {len(FILES)} explicit release files, {total:,} bytes; no text-pattern findings.")
    if args.archive:
        if args.archive.exists():
            p.error("Archive already exists")
        args.archive.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(args.archive, "x", compression=zipfile.ZIP_DEFLATED) as archive:
            for name in FILES:
                archive.write(root / name, f"adaptive-contact-geometry-mpc/{name}")
        digest = hashlib.sha256(args.archive.read_bytes()).hexdigest()
        print(f"Archive SHA-256: {digest}")


if __name__ == "__main__":
    main()
