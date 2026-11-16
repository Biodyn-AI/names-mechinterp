"""Temporary forced SSH command exposing only approved, closed research archives.

No listener, shell, directory listing, account credential or scientific mutation.
The author installs a restricted ephemeral PUBLIC key separately on the owned pod.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import stat
import sys
import time

ALLOWED_ROOTS = frozenset((
    "/workspace/migration/names-native-release-development-20261116",
    "/workspace/migration/names-native-release-confirmation-20261116",
))
HARD_EXPIRY = datetime(2026, 11, 16, 21, 45, tzinfo=timezone.utc).timestamp()
NAME = re.compile(r"(?:names-confirmation-[a-z0-9]+(?:-[a-z0-9]+)*|names-development-part[0-9]{3})\.tar\.gz")


def no_links(path):
    for item in (*path.parents, path):
        if item.is_symlink():
            raise ValueError("Symbolic links are not approved archive paths")


def unique_pairs(pairs):
    value = dict(pairs)
    if len(value) != len(pairs):
        raise ValueError("Duplicate catalog keys")
    return value


def load_catalog(path, now):
    no_links(path)
    value = json.loads(path.read_text(), object_pairs_hook=unique_pairs)
    if not isinstance(value, dict) or set(value) != {"expires_utc", "allowed_roots", "archives"}:
        raise ValueError("Invalid catalog schema")
    expiry = value["expires_utc"]
    if not isinstance(expiry, str) or not expiry.endswith("Z"):
        raise ValueError("An explicit UTC expiry is required")
    expiry = datetime.fromisoformat(expiry[:-1] + "+00:00").timestamp()
    if not now < expiry <= HARD_EXPIRY:
        raise ValueError("Reader authorization expired or exceeds its deadline")
    roots = value["allowed_roots"]
    if not isinstance(roots, list) or not roots or any(root not in ALLOWED_ROOTS for root in roots):
        raise ValueError("Unapproved publication root")
    for root in roots:
        no_links(Path(root))
        if not Path(root).is_dir():
            raise ValueError("Publication root is not a directory")
    records = value["archives"]
    if not isinstance(records, list) or not 1 <= len(records) <= 32:
        raise ValueError("Invalid archive inventory")
    approved = {}
    for record in records:
        if not isinstance(record, dict) or set(record) != {"name", "path", "size_bytes", "sha256"}:
            raise ValueError("Invalid archive record")
        name, location, size, checksum = (record[key] for key in ("name", "path", "size_bytes", "sha256"))
        if (not isinstance(name, str) or not NAME.fullmatch(name) or name in approved
                or not isinstance(location, str) or type(size) is not int
                or not 0 < size < 2 * 1024**3 or not isinstance(checksum, str)
                or not re.fullmatch(r"[0-9a-f]{64}", checksum)):
            raise ValueError("Invalid archive identity")
        archive = Path(location)
        if not archive.is_absolute() or str(archive.parent) not in roots or archive.name != name:
            raise ValueError("Archive escaped its publication directory")
        no_links(archive)
        info = archive.stat()
        if not stat.S_ISREG(info.st_mode) or info.st_size != size:
            raise ValueError("Archive is not the expected closed regular file")
        approved[name] = record
    return approved, expiry


def serve(catalog, command, output, *, clock=time.time):
    approved, expiry = load_catalog(Path(catalog), clock())
    if command == "probe":
        output.write(b"names-native-reader-v1\n")
        return
    if not command.startswith("get ") or command[4:] not in approved:
        raise ValueError("Only exact approved archive requests are allowed")
    record = approved[command[4:]]
    descriptor = os.open(record["path"], os.O_RDONLY | os.O_NOFOLLOW)
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode) or info.st_size != record["size_bytes"]:
            raise ValueError("Archive changed before streaming")
        remaining, checksum = record["size_bytes"], hashlib.sha256()
        while remaining:
            if clock() >= expiry:
                raise ValueError("Reader deadline reached during transfer")
            content = os.read(descriptor, min(1024 * 1024, remaining))
            if not content:
                raise ValueError("Archive ended early")
            if output.write(content) != len(content):
                raise OSError("Incomplete archive output")
            checksum.update(content)
            remaining -= len(content)
        if os.read(descriptor, 1) or checksum.hexdigest() != record["sha256"]:
            raise ValueError("Archive bytes differ from the approved checksum")
        output.flush()
    finally:
        os.close(descriptor)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, required=True)
    args = parser.parse_args()
    try:
        # Interrupt even a stalled client's blocking write at the absolute expiry.
        _, expiry = load_catalog(args.catalog, time.time())
        signal.signal(signal.SIGALRM, expired_alarm)
        signal.setitimer(signal.ITIMER_REAL, max(0.001, expiry - time.time()))
        serve(args.catalog, os.environ.get("SSH_ORIGINAL_COMMAND", ""), sys.stdout.buffer)
    except (OSError, ValueError, TypeError, KeyError):
        print("Archive reader refused request.", file=sys.stderr)
        return 1
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
    return 0


def expired_alarm(signum, frame):
    raise ValueError("Reader authorization expired")


if __name__ == "__main__":
    raise SystemExit(main())
