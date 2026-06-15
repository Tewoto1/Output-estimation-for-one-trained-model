"""_compat.py -- let the vendored kprop import on Python < 3.12 WITHOUT editing it.

The vendored ``..kprop`` library is copied verbatim from the ARC paper repo and
must stay byte-for-byte pristine (see the repo README: "vendored, unmodified").
It targets Python >= 3.12 and uses two constructs older interpreters reject:

  * PEP 695 type-alias statements   ``type Samples = Float[Tensor, "samples n"]``
    (incl. the one generic alias    ``type SetPartition[T] = tuple[frozenset[T], ...]``)
  * ``from typing import Self``      (``typing.Self`` only exists on 3.11+)

Rather than patch those into the vendored files (which is exactly the change we
reverted), this shim -- owned by ``skprop``, the spike-aware predictor that
depends on kprop -- installs a source-rewriting import hook that downgrades the
syntax *in memory* as each kprop module is loaded. The files on disk are never
touched.

On Python >= 3.12 ``install()`` is a no-op: the vendored code parses natively and
this module is never even imported (``Mecha_preds.cumulants.__init__`` guards the
call behind a version check). So this has zero effect on the repo's real runtime
(the project .venv is 3.13); it only makes the 3.10/3.11 sandbox able to run
skprop end-to-end.
"""
from __future__ import annotations

import importlib.abc
import importlib.machinery
import importlib.util
import os
import re
import sys

# Fully-qualified name + on-disk directory of the vendored kprop package.
# __name__ == "Mecha_preds.cumulants.skprop._compat"  ->  pkg "Mecha_preds.cumulants"
_KPROP_PKG = __name__.rsplit(".", 2)[0] + ".kprop"
_KPROP_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "kprop")

# A PEP 695 alias line:  <indent> type NAME [params]? = RHS
_TYPE_RE = re.compile(
    r"^(?P<indent>[ \t]*)type[ \t]+(?P<name>[A-Za-z_]\w*)[ \t]*"
    r"(?P<params>\[[^\]]*\])?[ \t]*=(?P<rhs>.*)$"
)
# Leading subscriptable origin of an RHS, e.g. "tuple" in "tuple[frozenset[T], ...]".
_ORIGIN_RE = re.compile(r"\s*([A-Za-z_][\w.]*)\s*\[")


def _downgrade_source(src: str) -> str:
    """Rewrite PEP 695 ``type`` alias statements to plain assignments.

    ``type NAME = RHS``            -> ``NAME = RHS``                  (keeps RHS)
    ``type NAME[T,..] = ORIGIN[..]`` -> ``NAME = ORIGIN``            (stays subscriptable,
                                                                      matching the upstream
                                                                      3.10 fix ``SetPartition = tuple``)
    ``type NAME[T,..] = RHS``      -> ``T = TypeVar('T'); .. ; NAME = RHS``  (fallback)
    Lines that don't start a ``type`` alias are returned unchanged, so docstrings
    or comments mentioning the word "type" are never affected.
    """
    out = []
    for line in src.split("\n"):
        m = _TYPE_RE.match(line)
        if m is None:
            out.append(line)
            continue
        indent, name, params, rhs = (
            m.group("indent"), m.group("name"), m.group("params"), m.group("rhs"),
        )
        if not params:
            out.append("%s%s =%s" % (indent, name, rhs))
            continue
        origin = _ORIGIN_RE.match(rhs)
        if origin:  # generic alias over a subscriptable base -> bind the bare base
            out.append("%s%s = %s" % (indent, name, origin.group(1)))
        else:        # no subscriptable origin: declare the type params, keep the RHS
            tvs = [p.split(":")[0].strip() for p in params[1:-1].split(",") if p.strip()]
            pre = "".join("%s = __import__('typing').TypeVar('%s'); " % (t, t) for t in tvs)
            out.append("%s%s%s =%s" % (indent, pre, name, rhs))
    return "\n".join(out)


def _patch_typing_self() -> None:
    """Ensure ``from typing import Self`` resolves on interpreters < 3.11."""
    import typing
    if hasattr(typing, "Self"):
        return
    try:
        import typing_extensions
        typing.Self = typing_extensions.Self  # type: ignore[attr-defined]
    except Exception:  # last-resort placeholder so the import line succeeds
        from typing import TypeVar
        typing.Self = TypeVar("Self")  # type: ignore[attr-defined]


class _RewriteLoader(importlib.machinery.SourceFileLoader):
    """SourceFileLoader that downgrades PEP 695 syntax before compiling."""

    def source_to_code(self, data, path, *, _optimize=-1):  # type: ignore[override]
        src = importlib.util.decode_source(data)
        src = _downgrade_source(src)
        return compile(src, path, "exec", dont_inherit=True, optimize=_optimize)


class _KpropFinder(importlib.abc.MetaPathFinder):
    """Intercept imports of the vendored kprop package and load via the rewriter."""

    def find_spec(self, fullname, path=None, target=None):
        if fullname != _KPROP_PKG and not fullname.startswith(_KPROP_PKG + "."):
            return None
        sub = fullname[len(_KPROP_PKG):].lstrip(".")
        parts = sub.split(".") if sub else []
        pkg_init = os.path.join(_KPROP_DIR, *parts, "__init__.py")
        mod_py = (os.path.join(_KPROP_DIR, *parts) + ".py") if parts else None
        if os.path.isfile(pkg_init):
            return importlib.util.spec_from_file_location(
                fullname, pkg_init, loader=_RewriteLoader(fullname, pkg_init),
                submodule_search_locations=[os.path.join(_KPROP_DIR, *parts)],
            )
        if mod_py and os.path.isfile(mod_py):
            return importlib.util.spec_from_file_location(
                fullname, mod_py, loader=_RewriteLoader(fullname, mod_py),
            )
        return None


_installed = False


def install() -> None:
    """Install the kprop import shim. No-op on Python >= 3.12 and idempotent."""
    global _installed
    if _installed or sys.version_info >= (3, 12):
        return
    _patch_typing_self()
    sys.meta_path.insert(0, _KpropFinder())
    _installed = True
