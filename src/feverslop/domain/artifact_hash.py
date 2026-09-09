import hashlib
import json
from pathlib import Path


def is_sha256_hex(value: object) -> bool:
    """Return whether value is an exact lowercase SHA-256 hex digest."""
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def sha256_bytes(data: bytes) -> str:
    """Hash already-loaded content without performing filesystem I/O."""
    return hashlib.sha256(data).hexdigest()


def fingerprint_json(value: object, *, ensure_ascii: bool = False) -> str:
    """Hash compact, key-sorted JSON using the requested Unicode encoding."""
    payload = json.dumps(
        value,
        ensure_ascii=ensure_ascii,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
