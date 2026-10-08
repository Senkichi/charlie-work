"""Per-process git repo templates behind ``_worktree_fixtures``' builders (HS-CW-4).

Each builder spawns 5-12 ``git`` processes per call (~113 ms on Windows). The
first call of a shape in a pytest process (each xdist worker is one) runs the
builder's own fresh path into a template directory; every later call copies
the template into the destination with ``shutil.copytree`` (~12 ms) and
rewrites the one value the copy inherits from the template's location -- the
``url =`` line git wrote into the config -- plus the same path inside reflog
messages. The copy path spawns no process.

Shapes: ``plain`` (``_init_repo``), ``bare`` (``_init_repo(bare=True)``),
``ebc`` (``_init_bare_remote_and_clone``), ``plain-origin``
(``_init_repo_with_origin``: a plain repo whose ``origin`` remote the build
sets to a placeholder URL, rewritten per copy like the clone shapes), and
``clone-of-<shape>`` (``_clone_repo`` from a repo this registry materialized
that nothing has changed since: same refs, same config, same git environment).

Copies of one shape share commit SHAs where fresh builds differ by
timestamp. A test that builds two "independent" repos and pushes, fetches or
merges between them passes ``fresh=True`` to the builder.

Fail modes: a template BUILD failure (including a failed invariant) emits one
``warnings.warn("test template disabled: <cause>")`` per process and that
shape takes the fresh path from then on; a wrong COPY raises
``AssertionError("git template copy mismatch: ...")`` (the test fails).
``CI_FLEET_TEST_REUSE=off`` forces the fresh path; it is read on every call.
"""

from __future__ import annotations

import atexit
import os
import re
import shutil
import stat
import subprocess
import sys
import tempfile
import warnings
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

REUSE_ENV = "CI_FLEET_TEST_REUSE"
# Written into the ``plain-origin`` template's config by ``git remote add`` at
# build time; every copy rewrites it to the caller's URL. Chosen to be
# ``_config_safe`` and to collide with no real remote URL.
_ORIGIN_URL_PLACEHOLDER = "https://git-template.invalid/origin.git"
_ENV_EXTRA = frozenset({"HOME", "USERPROFILE", "XDG_CONFIG_HOME"})
# A test that points HOME/GIT_* somewhere unique would otherwise build one
# template per test; past this many distinct environments, build fresh.
_MAX_ENVS = 8
# Captured at import: per-test fixtures redirect ``tempfile.gettempdir``, and a
# template must outlive the test that happened to build it.
_REAL_GETTEMPDIR = tempfile.gettempdir()
_UNSAFE_CONFIG_CHARS = frozenset('#;"\t\n\r')

EnvKey = tuple[tuple[str, str], ...]
RefsKey = tuple[tuple[str, bytes], ...]
InitBuilder = Callable[[Path, bool], None]
CloneBuilder = Callable[[Path, Path], None]
PairBuilder = Callable[[Path], tuple[Path, Path]]


def reuse_enabled() -> bool:
    return os.environ.get(REUSE_ENV, "").strip().lower() != "off"


def env_fingerprint() -> EnvKey:
    """The environment a builder's output depends on.

    ``GIT_CEILING_DIRECTORIES`` is excluded: ``conftest._isolate_git_env`` points
    it at each test's own basetemp, and it only bounds repository discovery,
    never what ``init``/``clone``/``commit`` write.
    """
    return tuple(
        sorted(
            (key, value)
            for key, value in os.environ.items()
            if key.upper() in _ENV_EXTRA
            or (key.upper().startswith("GIT_") and key.upper() != "GIT_CEILING_DIRECTORIES")
        )
    )


def _scratch_root() -> Path:
    """The shallow root ``wt_scratch`` uses (``_worktree_fixtures``)."""
    roots = []
    local_appdata = os.environ.get("LOCALAPPDATA")
    if local_appdata:
        roots.append(Path(local_appdata) / "Temp")
    roots.append(Path(_REAL_GETTEMPDIR))
    root = min(roots, key=lambda p: len(str(p))) / "charlie-wt-scratch"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _remove_tree(path: Path) -> None:
    """``rmtree`` that clears git's read-only object files on Windows; never raises."""

    def _retry(func: Callable[[str], object], target: str, _exc: object) -> None:
        os.chmod(target, stat.S_IWRITE)
        func(target)

    try:
        if sys.version_info >= (3, 12):
            shutil.rmtree(path, onexc=_retry)
        else:
            shutil.rmtree(path, onerror=_retry)
    except OSError:
        pass


def _forms(path: Path | str) -> tuple[str, str]:
    """(native, forward-slash) spellings; git records whichever it was handed."""
    native = str(path)
    return native, native.replace("\\", "/")


def _config_escape(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"')


def _config_safe(value: str) -> bool:
    """A value git writes unquoted, so a byte-level rewrite sees it verbatim."""
    return value == value.strip() and not (_UNSAFE_CONFIG_CHARS & set(value))


def _url_line(value: str) -> re.Pattern[bytes]:
    escaped = re.escape(_config_escape(value).encode("utf-8"))
    return re.compile(rb"url = " + escaped + rb"(?=\r?\n)")


def _recorded_form(config: bytes, source: Path) -> int:
    """Which spelling of ``source`` git wrote as a ``url =`` value (index into ``_forms``)."""
    for index, form in enumerate(_forms(source)):
        if _config_safe(form) and _url_line(form).search(config):
            return index
    raise RuntimeError(f"no 'url = {source}' line in the template config")


def _rewrite_url(config_path: Path, old: str, new: str) -> None:
    data = config_path.read_bytes()
    replacement = b"url = " + _config_escape(new).encode("utf-8")
    updated, count = _url_line(old).subn(lambda _match: replacement, data)
    if count != 1:
        raise AssertionError(
            f"git template copy mismatch: expected one 'url = {old}' in {config_path}, "
            f"found {count}"
        )
    config_path.write_bytes(updated)


def _rewrite_logs(gitdir: Path, old: str, new: str) -> None:
    """Re-point reflog messages (``clone: from <url>``) at the copy's own remote."""
    logs = gitdir / "logs"
    if not logs.is_dir():
        return
    pairs = [
        (old_form.encode("utf-8"), new_form.encode("utf-8"))
        for old_form, new_form in zip(_forms(old), _forms(new), strict=True)
    ]
    for log in logs.rglob("*"):
        if not log.is_file():
            continue
        data = log.read_bytes()
        updated = data
        for old_bytes, new_bytes in pairs:
            updated = updated.replace(old_bytes, new_bytes)
        if updated != data:
            log.write_bytes(updated)


def _copy_tree(src: Path, dst: Path) -> None:
    shutil.copytree(src, dst, dirs_exist_ok=True)


def _refs_fingerprint(gitdir: Path, *, with_config: bool = False) -> RefsKey:
    names = ["HEAD", "packed-refs", *(["config"] if with_config else [])]
    entries = [(name, (gitdir / name).read_bytes()) for name in names if (gitdir / name).is_file()]
    refs = gitdir / "refs"
    if refs.is_dir():
        entries.extend(
            sorted(
                (path.relative_to(gitdir).as_posix(), path.read_bytes())
                for path in refs.rglob("*")
                if path.is_file()
            )
        )
    return tuple(entries)


def _verify_copy(template_gitdir: Path, copy_gitdir: Path) -> None:
    if _refs_fingerprint(template_gitdir) != _refs_fingerprint(copy_gitdir):
        raise AssertionError(
            f"git template copy mismatch: refs in {copy_gitdir} differ from {template_gitdir}"
        )


def _check_invariants(gitdir: Path, *, identity: bool) -> None:
    if (gitdir / "worktrees").exists():
        raise RuntimeError(f"{gitdir} has linked worktrees (they embed absolute paths)")
    for sub in ("hooks", "info"):
        if not (gitdir / sub).is_dir():
            raise RuntimeError(f"{gitdir} has no {sub}/ directory")
    if identity:
        config = (gitdir / "config").read_text(encoding="utf-8")
        if "email = test@example.test" not in config or "name = Test User" not in config:
            raise RuntimeError(f"{gitdir} has no test committer identity")


def _eligible(dst: Path) -> bool:
    if not dst.is_absolute() or not all(_config_safe(form) for form in _forms(dst)):
        return False
    return not dst.exists() or (dst.is_dir() and not any(dst.iterdir()))


def _gitdir(repo: Path) -> Path:
    return repo / ".git" if (repo / ".git").is_dir() else repo


def _norm(path: Path) -> str:
    return os.path.normcase(os.path.abspath(path))


@dataclass(frozen=True)
class _Template:
    repo: Path
    url: str | None = None
    url_form: int = 0
    remote: Path | None = None


class TemplateRegistry:
    """Per-process templates keyed by ``(shape, env_fingerprint())``."""

    def __init__(self) -> None:
        self._templates: dict[tuple[str, EnvKey], _Template] = {}
        self._failed: set[tuple[str, EnvKey]] = set()
        self._warned = False
        self._pristine: dict[str, tuple[str, EnvKey, RefsKey]] = {}
        self.materialized: Counter[str] = Counter()

    def lookup(self, shape: str) -> _Template | None:
        return self._templates.get((shape, env_fingerprint()))

    def template(self, shape: str, build: Callable[[Path], _Template]) -> _Template | None:
        env = env_fingerprint()
        key = (shape, env)
        if key in self._templates:
            return self._templates[key]
        if key in self._failed:
            return None
        if len({env, *(known_env for _, known_env in self._templates)}) > _MAX_ENVS:
            return None
        try:
            root = Path(tempfile.mkdtemp(prefix="tpl-", dir=_scratch_root())).resolve()
            atexit.register(_remove_tree, root)
            built = build(root)
        except Exception as exc:  # any build failure falls back to the fresh path
            self._failed.add(key)
            if not self._warned:
                self._warned = True
                warnings.warn(f"test template disabled: {shape}: {exc}", stacklevel=3)
            return None
        self._templates[key] = built
        return built

    def mark_pristine(self, repo: Path, shape: str) -> None:
        fingerprint = _refs_fingerprint(_gitdir(repo), with_config=True)
        self._pristine[_norm(repo)] = (shape, env_fingerprint(), fingerprint)

    def pristine_shape(self, repo: Path) -> str | None:
        entry = self._pristine.get(_norm(repo))
        if entry is None or not repo.is_dir():
            return None
        shape, env, fingerprint = entry
        if env != env_fingerprint():
            return None
        if _refs_fingerprint(_gitdir(repo), with_config=True) != fingerprint:
            return None
        return shape


_REGISTRY = TemplateRegistry()


def init_repo(
    dst: Path, *, bare: bool, build: InitBuilder, registry: TemplateRegistry | None = None
) -> bool:
    """Materialize ``_init_repo``'s shape into ``dst``; False means "take the fresh path"."""
    registry = registry or _REGISTRY
    if not reuse_enabled() or not _eligible(dst):
        return False
    shape = "bare" if bare else "plain"

    def _build(root: Path) -> _Template:
        repo = root / ("remote.git" if bare else "repo")
        build(repo, bare)
        _check_invariants(_gitdir(repo), identity=not bare)
        if not bare:
            return _Template(repo=repo)
        source = repo.parent / f"{repo.name}-temp"
        form = _recorded_form((repo / "config").read_bytes(), source)
        return _Template(repo=repo, url=_forms(source)[form], url_form=form)

    template = registry.template(shape, _build)
    if template is None:
        return False
    _copy_tree(template.repo, dst)
    gitdir = _gitdir(dst)
    if template.url is not None:
        new_url = _forms(dst.parent / f"{dst.name}-temp")[template.url_form]
        _rewrite_url(gitdir / "config", template.url, new_url)
        _rewrite_logs(gitdir, template.url, new_url)
    _verify_copy(_gitdir(template.repo), gitdir)
    registry.materialized[shape] += 1
    registry.mark_pristine(dst, shape)
    return True


def init_repo_with_origin(
    dst: Path, url: str, *, build: InitBuilder, registry: TemplateRegistry | None = None
) -> bool:
    """Materialize a ``plain`` repo plus an ``origin`` remote at ``url``; False
    means "take the fresh path".

    The ``git remote add`` spawn is paid once per process inside the template
    build, against ``_ORIGIN_URL_PLACEHOLDER``; each copy then rewrites the
    ``url =`` line to ``url`` through the same machinery the clone shapes use
    for their per-copy remote path, so a materialized call spawns no process.
    """
    registry = registry or _REGISTRY
    if not reuse_enabled() or not _eligible(dst) or not _config_safe(url):
        return False
    shape = "plain-origin"

    def _build(root: Path) -> _Template:
        repo = root / "repo"
        build(repo, False)
        subprocess.run(
            ["git", "remote", "add", "origin", _ORIGIN_URL_PLACEHOLDER],
            cwd=repo,
            check=True,
            capture_output=True,
            text=True,
        )
        _check_invariants(_gitdir(repo), identity=True)
        return _Template(repo=repo, url=_ORIGIN_URL_PLACEHOLDER)

    template = registry.template(shape, _build)
    if template is None or template.url is None:
        return False
    _copy_tree(template.repo, dst)
    gitdir = _gitdir(dst)
    _rewrite_url(gitdir / "config", template.url, url)
    _rewrite_logs(gitdir, template.url, url)
    _verify_copy(_gitdir(template.repo), gitdir)
    registry.materialized[shape] += 1
    registry.mark_pristine(dst, shape)
    return True


def clone_repo(
    remote: Path, dst: Path, *, build: CloneBuilder, registry: TemplateRegistry | None = None
) -> bool:
    """Materialize a clone of an unchanged materialized repo; False means fresh path."""
    registry = registry or _REGISTRY
    if not reuse_enabled() or not remote.is_absolute() or not _eligible(dst):
        return False
    if not all(_config_safe(form) for form in _forms(remote)):
        return False
    remote_shape = registry.pristine_shape(remote)
    origin = registry.lookup(remote_shape) if remote_shape is not None else None
    if origin is None:
        return False
    origin_repo = origin.remote if remote_shape == "ebc" else origin.repo
    if origin_repo is None:
        return False
    shape = f"clone-of-{remote_shape}"

    def _build(root: Path) -> _Template:
        repo = root / "clone"
        build(origin_repo, repo)
        _check_invariants(repo / ".git", identity=True)
        form = _recorded_form((repo / ".git" / "config").read_bytes(), origin_repo)
        return _Template(repo=repo, url=_forms(origin_repo)[form], url_form=form)

    template = registry.template(shape, _build)
    if template is None or template.url is None:
        return False
    _copy_tree(template.repo, dst)
    new_url = _forms(remote)[template.url_form]
    _rewrite_url(dst / ".git" / "config", template.url, new_url)
    _rewrite_logs(dst / ".git", template.url, new_url)
    _verify_copy(template.repo / ".git", dst / ".git")
    registry.materialized[shape] += 1
    registry.mark_pristine(dst, shape)
    return True


def bare_remote_and_clone(
    base: Path, *, build: PairBuilder, registry: TemplateRegistry | None = None
) -> tuple[Path, Path] | None:
    """Materialize ``_init_bare_remote_and_clone(base)``; None means "take the fresh path"."""
    registry = registry or _REGISTRY
    remote_dst, clone_dst = base / "remote", base / "clone"
    if not reuse_enabled() or not (_eligible(remote_dst) and _eligible(clone_dst)):
        return None

    def _build(root: Path) -> _Template:
        remote, clone = build(root)
        _check_invariants(remote, identity=False)
        _check_invariants(clone / ".git", identity=True)
        form = _recorded_form((clone / ".git" / "config").read_bytes(), remote)
        return _Template(repo=clone, url=_forms(remote)[form], url_form=form, remote=remote)

    template = registry.template("ebc", _build)
    if template is None or template.url is None or template.remote is None:
        return None
    _copy_tree(template.remote, remote_dst)
    _copy_tree(template.repo, clone_dst)
    new_url = _forms(remote_dst)[template.url_form]
    _rewrite_url(clone_dst / ".git" / "config", template.url, new_url)
    _rewrite_logs(clone_dst / ".git", template.url, new_url)
    _verify_copy(template.remote, remote_dst)
    _verify_copy(template.repo / ".git", clone_dst / ".git")
    registry.materialized["ebc"] += 1
    registry.mark_pristine(remote_dst, "ebc")
    return remote_dst, clone_dst
