"""Software measurement of the code a client is about to run.

A hardware TEE measures the loaded enclave image and signs the digest with a key
the host cannot reach. Without hardware, the next best thing is to measure the
*actual files* -- the installed ``trustfed`` package, the training script, and
anything else the pipeline loads -- and have the site's registered node key sign
that digest. The measurement is reproducible: the same release of the package and
the same script yield the same digest on any machine, so maintainers can publish
the digest of an approved pipeline and a verifier can allow-list it.

What this establishes: a registered node vouched, with its key, that it ran code
with this digest. What it does not establish: anything about a node whose
operator lies, since the operator holds the key. That residual risk is what the
Byzantine-resilient aggregation and the multi-site evaluation exist for.
"""

from __future__ import annotations

import hashlib
import importlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence

from trustfed.ledger.crypto import canonical_json

#: Scheme tag inside every manifest, so a verifier knows how the digest was made.
MEASUREMENT_SCHEME = "openmed-software-measurement/1"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def measure_tree(root: Path, *, suffixes: Sequence[str] = (".py",)) -> Dict[str, Any]:
    """Digest every file under ``root`` with one of ``suffixes``.

    The digest covers relative paths and contents, in sorted order, so it is
    independent of where the tree is installed and of file timestamps.
    Compiled bytecode and caches are excluded by the suffix filter.
    """
    root = Path(root).resolve()
    files = sorted(
        p for p in root.rglob("*") if p.is_file() and p.suffix in suffixes and "__pycache__" not in p.parts
    )
    h = hashlib.sha256()
    for path in files:
        rel = path.relative_to(root).as_posix()
        h.update(rel.encode("utf-8"))
        h.update(b"\0")
        h.update(sha256_file(path).encode("ascii"))
        h.update(b"\n")
    return {"root": root.name, "files": len(files), "sha256": h.hexdigest()}


def measure_package(name: str = "trustfed") -> Dict[str, Any]:
    """Digest an importable package's source tree."""
    module = importlib.import_module(name)
    file = getattr(module, "__file__", None)
    if not file:
        raise ValueError(f"package {name!r} has no file system location to measure")
    tree = measure_tree(Path(file).parent)
    tree.update(package=name, version=str(getattr(module, "__version__", "") or ""))
    return tree


def measure_file(path: Path) -> Dict[str, Any]:
    path = Path(path)
    return {"name": path.name, "sha256": sha256_file(path), "bytes": path.stat().st_size}


@dataclass(frozen=True)
class SoftwareMeasurement:
    """A measurement digest plus the manifest it was computed from."""

    measurement: str
    manifest: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {"measurement": self.measurement, "manifest": dict(self.manifest)}


def software_measurement(
    *,
    packages: Iterable[str] = ("trustfed",),
    script: Optional[Path] = None,
    extra_files: Iterable[Path] = (),
    labels: Optional[Mapping[str, Any]] = None,
) -> SoftwareMeasurement:
    """Measure the installed packages, the training script and extra files.

    The measurement is ``sha256(canonical_json(manifest))``. The manifest
    deliberately excludes machine-specific facts (Python build, hostname, paths)
    so that the same code yields the same digest everywhere; run-specific
    configuration goes in the quote's separate ``config_hash`` field.
    """
    manifest: Dict[str, Any] = {
        "scheme": MEASUREMENT_SCHEME,
        "packages": [measure_package(name) for name in packages],
        "script": measure_file(script) if script is not None else None,
        "extra_files": [measure_file(p) for p in extra_files],
        "labels": dict(labels or {}),
    }
    return SoftwareMeasurement(sha256_bytes(canonical_json(manifest)), manifest)


__all__ = [
    "MEASUREMENT_SCHEME",
    "SoftwareMeasurement",
    "measure_file",
    "measure_package",
    "measure_tree",
    "sha256_bytes",
    "sha256_file",
    "software_measurement",
]
