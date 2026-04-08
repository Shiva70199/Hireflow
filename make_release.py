from __future__ import annotations

import shutil
import zipfile
from pathlib import Path


def _should_skip(path: Path, root: Path) -> bool:
    rel = path.relative_to(root)
    parts = rel.parts
    if parts[0] in {"dist", "__pycache__", ".git"}:
        return True
    if "__pycache__" in parts:
        return True
    if path.suffix == ".pyc":
        return True
    return False


def main() -> None:
    root = Path(__file__).resolve().parent
    release_dir = root / "dist"
    release_dir.mkdir(exist_ok=True)
    out_zip = release_dir / "hireflow-multiagent-env.zip"
    if out_zip.exists():
        out_zip.unlink()

    with zipfile.ZipFile(out_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for p in root.rglob("*"):
            if not p.is_file():
                continue
            if _should_skip(p, root):
                continue
            arc = p.relative_to(root).as_posix()
            zf.write(p, arcname=arc)

    print(f"Release archive generated: {out_zip}")


if __name__ == "__main__":
    main()
