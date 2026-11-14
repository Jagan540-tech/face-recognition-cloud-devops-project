#!/usr/bin/env python3
"""Fail if TensorFlow (or Keras / tflite / other TF-family packages) is installed, or reachable
through the dependency tree of requirements.txt, or gets imported by the app.

    python scripts/verify_no_tensorflow.py
Run it in a CLEAN environment (fresh venv or the Docker image); a venv you reused from the old
version of this project may still contain TensorFlow.
"""
import re
import sys
from importlib import metadata
from pathlib import Path

FORBIDDEN = re.compile(r"^(tensorflow|tensorflow[-_].*|tf[-_].*|keras|keras[-_].*|tflite[-_].*|"
                       r"mtcnn|dlib|dlib[-_]bin|torch|torchvision|onnxruntime.*|tensorboard.*)$", re.I)


def norm(name):
    return re.sub(r"[-_.]+", "-", name).lower()


def req_name(line):
    m = re.match(r"\s*([A-Za-z0-9][A-Za-z0-9._-]*)", line)
    return norm(m.group(1)) if m else None


def applies(requirement):
    """Does this requirement line apply to THIS environment (extras and python/platform markers)?"""
    if ";" not in requirement:
        return True
    try:
        from packaging.markers import Marker  # installed with gunicorn; part of the runtime tree
        return Marker(requirement.split(";", 1)[1].strip()).evaluate({"extra": ""})
    except Exception:
        return "extra ==" not in requirement


def walk(roots):
    seen, stack, missing = {}, list(roots), []
    while stack:
        name = stack.pop()
        if name in seen:
            continue
        try:
            dist = metadata.distribution(name)
        except metadata.PackageNotFoundError:
            missing.append(name)
            continue
        seen[name] = dist.version
        for r in dist.requires or []:
            if not applies(r):        # optional extras / other-platform markers are not installed
                continue
            n = req_name(r)
            if n:
                stack.append(n)
    return seen, missing


def main():
    req = Path(__file__).resolve().parent.parent / "requirements.txt"
    roots = [req_name(l) for l in req.read_text().splitlines()
             if l.strip() and not l.strip().startswith(("#", "-"))]
    tree, missing = walk([r for r in roots if r])
    print("Runtime dependency tree (requirements.txt):")
    for n, v in sorted(tree.items()):
        print(f"  {n}=={v}")
    problems = []
    if missing:
        problems.append(f"requirements not installed here: {', '.join(missing)}")
    bad_tree = [n for n in tree if FORBIDDEN.match(n)]
    if bad_tree:
        problems.append(f"forbidden packages in the dependency tree: {bad_tree}")
    installed = sorted({norm(d.metadata['Name']) for d in metadata.distributions() if d.metadata['Name']})
    bad_env = [n for n in installed if FORBIDDEN.match(n)]
    if bad_env:
        problems.append(f"forbidden packages installed in this environment: {bad_env}")
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    import app  # noqa: F401  (importing the app must not pull in TensorFlow)
    bad_mod = sorted(m for m in sys.modules if m.split(".")[0] in {"tensorflow", "keras", "tflite_runtime", "torch"})
    if bad_mod:
        problems.append(f"forbidden modules imported by the app: {bad_mod[:5]}")
    if problems:
        print("\nFAIL")
        for p in problems:
            print(" -", p)
        sys.exit(1)
    print(f"\nPASS: no TensorFlow/Keras/PyTorch/onnxruntime/dlib/mtcnn in {len(tree)} runtime packages, "
          f"{len(installed)} installed packages, or imported modules.")


if __name__ == "__main__":
    main()
