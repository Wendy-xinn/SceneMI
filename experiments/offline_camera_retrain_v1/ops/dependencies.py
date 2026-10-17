"""Explicit source entry points and their repository-local static imports.

Runtime-selected implementations belong in ``entries`` too. External source
files are admitted by an explicit file path rather than a directory scan.
"""
from __future__ import annotations

import ast
import hashlib
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
PROTOCOL = "explicit-entry-import-closure-v1"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def dependency_closure(entries, *, root=PROJECT_ROOT, extra_files=(), import_roots=()):
    """Hash explicit entries and recursively resolve their local AST imports."""
    root = Path(root).resolve()
    roots = [root, *(Path(value).resolve() for value in import_roots)]

    def resolve(value):
        path = Path(value)
        return (root / path if not path.is_absolute() else path).resolve()

    pending, found = [resolve(path) for path in entries], set()
    if not pending:
        raise ValueError("Source seal needs explicit entry points")
    while pending:
        path = pending.pop()
        if path in found:
            continue
        if not path.is_file():
            raise ValueError("Missing explicit dependency: " + str(path))
        package_root = next((base for base in roots if path.is_relative_to(base)), None)
        if package_root is None:
            raise ValueError("External files must use extra_files: " + str(path))
        found.add(path)
        if path.suffix != ".py":
            continue
        package = list(path.relative_to(package_root).with_suffix("").parts[:-1])
        for node in ast.walk(ast.parse(path.read_text(), filename=str(path))):
            modules = []
            if isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                prefix = package[:len(package) - node.level + 1] if node.level else []
                base = prefix + (node.module.split(".") if node.module else [])
                modules = [".".join(base)] + [".".join(base + [a.name]) for a in node.names]
            for module in modules:
                for search_root in roots:
                    base = search_root.joinpath(*module.split("."))
                    for candidate in (base.with_suffix(".py"), base / "__init__.py"):
                        if candidate.is_file() and candidate.resolve().is_relative_to(search_root):
                            pending.append(candidate.resolve())
    for value in extra_files:
        path = resolve(value)
        if not path.is_file():
            raise ValueError("Missing explicit external dependency: " + str(path))
        found.add(path)
    return {str(path): sha256(path) for path in sorted(found)}


def configured_sources(options):
    if options.get("protocol") != PROTOCOL:
        raise ValueError("Unknown explicit source-seal protocol")
    return dependency_closure(options["entries"], extra_files=options.get("extra_files", ()),
                              import_roots=options.get("import_roots", ()))
