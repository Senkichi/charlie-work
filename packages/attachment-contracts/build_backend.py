"""Minimal PEP 517 build backend for charlie-work-attachment-contracts.

Both hatchling and setuptools reject ``..`` in package paths (paths must be
relative to the project root and cannot escape it).  This backend builds the
wheel and sdist directly from the single in-tree source at
``src/charlie_work/attachment_contracts/`` -- no second copy of the 13 modules
exists anywhere in this repo (issue #1544 Stage 1).

The backend implements just enough of PEP 517 (``build_wheel`` and
``build_sdist``) to produce a standards-compliant wheel from the source
directory.  Metadata is read from the ``[project]`` table in ``pyproject.toml``
via ``tomllib`` (stdlib, Python >=3.11).
"""

from __future__ import annotations

import base64
import csv
import hashlib
import io
import os
import re
import tarfile
import tomllib
import zipfile
from pathlib import Path

_HERE = Path(__file__).resolve().parent
# The source tree lives at ``src/charlie_work/attachment_contracts/`` but its
# location relative to this backend module depends on which layout the build
# is running from:
#
# * in-repo layout: this file is at ``packages/attachment-contracts/build_backend.py``
#   and the source is at ``<repo>/src/charlie_work/attachment_contracts/`` (two
#   levels up from ``_HERE``).
# * sdist layout: ``build_sdist`` places this file at ``<prefix>/build_backend.py``
#   and bundles the source at ``<prefix>/src/charlie_work/attachment_contracts/``
#   (one level down from ``_HERE``).  A wheel built FROM the sdist (the default
#   ``python -m build`` / ``uv build`` flow, and the standard PEP 517 isolated
#   build) runs the backend from the extracted sdist prefix, so the source is
#   ``_HERE / "src" / ...`` -- NOT ``_HERE.parent.parent / "src" / ...``.
#
# The prior code unconditionally used ``_HERE.parent.parent / "src" / ...``,
# which is correct for the in-repo layout but WRONG for the sdist layout: a
# wheel built from the sdist silently contained zero ``.py`` files because the
# resolved source directory did not exist (round-1 review finding).  Detecting
# the layout by existence picks the right one in both cases.
_SDIST_SRC_DIR = _HERE / "src" / "charlie_work" / "attachment_contracts"
_REPO_SRC_DIR = _HERE.parent.parent / "src" / "charlie_work" / "attachment_contracts"
_SRC_DIR = _SDIST_SRC_DIR if _SDIST_SRC_DIR.is_dir() else _REPO_SRC_DIR


def _read_project() -> dict:
    with (_HERE / "pyproject.toml").open("rb") as f:
        return tomllib.load(f)["project"]


def _normalize_name(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _wheel_filename(name: str, version: str) -> str:
    return f"{_normalize_name(name).replace('-', '_')}-{version}-py3-none-any.whl"


def _dist_info_dir(name: str, version: str) -> str:
    """Return the ``.dist-info`` directory name for the wheel.

    Per the wheel spec, the dist-info directory is
    ``{normalized_name_with_underscores}-{version}.dist-info`` -- the same
    name normalization as :func:`_wheel_filename`.  Deriving it from the
    project name/version (read via :func:`_read_project`) keeps the
    dist-info directory name in sync with the wheel filename on every
    version bump; a hardcoded constant would silently desync and produce a
    spec-non-compliant wheel (round-2 review finding).
    """
    return f"{_normalize_name(name).replace('-', '_')}-{version}.dist-info"


def _metadata_payload(project: dict) -> str:
    """Return the METADATA file content (RFC 822-style, per the wheel spec)."""
    lines = [
        "Metadata-Version: 2.1",
        f"Name: {project['name']}",
        f"Version: {project['version']}",
        f"Summary: {project['description']}",
        f"Requires-Python: {project['requires-python']}",
        f"License: {project['license']['text']}",
    ]
    for kw in project.get("keywords", []):
        # Keywords are a single comma-separated line.
        pass
    if project.get("keywords"):
        lines.append(f"Keywords: {','.join(project['keywords'])}")
    for cls in project.get("classifiers", []):
        lines.append(f"Classifier: {cls}")
    readme = project.get("readme", {})
    if isinstance(readme, dict) and "text" in readme:
        lines.append(f"Description-Content-Type: {readme.get('content-type', 'text/plain')}")
        lines.append("")
        lines.append(readme["text"])
    return "\n".join(lines) + "\n"


def _record_hash(data: bytes) -> tuple[str, str]:
    digest = hashlib.sha256(data).digest()
    return "sha256=" + base64.urlsafe_b64encode(digest).rstrip(b"=").decode(), str(len(data))


def build_wheel(
    wheel_directory: str,
    config_settings: dict | None = None,
    metadata_directory: str | None = None,
) -> str:
    """Build the wheel from ``src/charlie_work/attachment_contracts/*.py``."""
    project = _read_project()
    name = project["name"]
    version = project["version"]
    filename = _wheel_filename(name, version)
    dist_info = _dist_info_dir(name, version)
    wheel_path = os.path.join(wheel_directory, filename)

    records: list[list[str]] = []
    all_entries: list[tuple[str, bytes]] = []

    # Package .py files — sorted for reproducibility.
    for py_file in sorted(_SRC_DIR.glob("*.py")):
        arcname = f"charlie_work/attachment_contracts/{py_file.name}"
        data = py_file.read_bytes()
        all_entries.append((arcname, data))

    # dist-info metadata files.
    metadata = _metadata_payload(project).encode()
    all_entries.append((f"{dist_info}/METADATA", metadata))

    wheel_meta = (
        "Wheel-Version: 1.0\n"
        "Generator: charlie-work-attachment-contracts-build-backend (0.1.1)\n"
        "Root-Is-Purelib: true\n"
        "Tag: py3-none-any\n"
    ).encode()
    all_entries.append((f"{dist_info}/WHEEL", wheel_meta))

    license_text = project["license"]["text"]
    all_entries.append(
        (
            f"{dist_info}/LICENSE",
            f"MIT License\n\nCopyright (c) 2026 Senkichi\n\n{license_text}\n".encode(),
        )
    )

    # RECORD — written last, with hashes for all other entries plus a self-entry.
    for arcname, data in all_entries:
        h, size = _record_hash(data)
        records.append([arcname, h, size])
    records.append([f"{dist_info}/RECORD", "", ""])

    record_data = io.StringIO()
    writer = csv.writer(record_data, lineterminator="\n")
    writer.writerows(records)
    all_entries.append((f"{dist_info}/RECORD", record_data.getvalue().encode()))

    # Write the wheel ZIP.
    os.makedirs(wheel_directory, exist_ok=True)
    with zipfile.ZipFile(wheel_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for arcname, data in sorted(all_entries, key=lambda e: e[0]):
            info = zipfile.ZipInfo(arcname)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            zf.writestr(info, data)

    return filename


def build_sdist(
    sdist_directory: str,
    config_settings: dict | None = None,
) -> str:
    """Build the source distribution (a tarball of pyproject.toml + backend + sources).

    The sdist bundles ``pyproject.toml``, ``build_backend.py``, ``PKG-INFO``, and
    the source ``.py`` files under ``<prefix>/src/charlie_work/attachment_contracts/``.
    A wheel built FROM this sdist (the default ``python -m build`` / ``uv build``
    flow) extracts the prefix and runs ``build_wheel`` there, where
    ``_SRC_DIR`` detects the sdist layout (``_HERE / "src" / ...``) and finds the
    bundled source -- without the layout detection, the wheel would silently
    contain zero ``.py`` files (round-1 review finding).
    """
    project = _read_project()
    name = project["name"]
    version = project["version"]
    sdist_name = f"{_normalize_name(name).replace('-', '_')}-{version}.tar.gz"
    sdist_path = os.path.join(sdist_directory, sdist_name)
    os.makedirs(sdist_directory, exist_ok=True)

    prefix = f"{_normalize_name(name).replace('-', '_')}-{version}"
    with tarfile.open(sdist_path, "w:gz") as tar:
        # pyproject.toml
        _add_to_tar(tar, _HERE / "pyproject.toml", f"{prefix}/pyproject.toml")
        # build_backend.py (the PEP 517 backend)
        _add_to_tar(tar, _HERE / "build_backend.py", f"{prefix}/build_backend.py")
        # PKG-INFO (PEP 643 / core metadata)
        pkg_info = _metadata_payload(project).encode()
        info = tarfile.TarInfo(name=f"{prefix}/PKG-INFO")
        info.size = len(pkg_info)
        info.mode = 0o644
        tar.addfile(info, io.BytesIO(pkg_info))
        # Source .py files
        for py_file in sorted(_SRC_DIR.glob("*.py")):
            _add_to_tar(
                tar, py_file, f"{prefix}/src/charlie_work/attachment_contracts/{py_file.name}"
            )

    return sdist_name


def _add_to_tar(tar: tarfile.TarFile, path: Path, arcname: str) -> None:
    info = tarfile.TarInfo(name=arcname)
    info.size = path.stat().st_size
    info.mode = 0o644
    with path.open("rb") as f:
        tar.addfile(info, f)
