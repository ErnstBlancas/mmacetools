"""Convert a whole-model file between macefork and upstream mace.

A whole-model file is a zip archive whose data.pkl records each class by module
path, e.g. `macefork.modules.models.ScaleShiftMACE`. Both packages have the same
classes, so this detects which one the model was saved with, rewrites those
module paths to the other one and copies every other entry unchanged.

    python fork2mace.py finetune.model [converted.model]
"""

import argparse
import pickletools
import zipfile
from pathlib import Path

FORK = "macefork"
MACE = "mace"


def in_package(module: str, package: str) -> bool:
    return module == package or module.startswith(package + ".")


def find_package(data: bytes) -> str:
    """Return the package (macefork or mace) the pickled classes come from."""
    for op, arg, pos in pickletools.genops(data):
        if op.name == "GLOBAL":
            module = arg.split(" ", 1)[0]
            for package in (FORK, MACE):
                if in_package(module, package):
                    return package
    raise ValueError("no macefork or mace classes found in the model")


def rename_modules(data: bytes, old: str, new: str) -> bytes:
    """Rewrite `<old>.*` class references in a pickle stream to `<new>.*`."""
    out = bytearray()
    ops = list(pickletools.genops(data))
    for i, (op, arg, pos) in enumerate(ops):
        end = ops[i + 1][2] if i + 1 < len(ops) else len(data)
        chunk = data[pos:end]
        if op.name == "GLOBAL":  # torch saves with protocol 2: "c<module>\n<name>\n"
            module, name = arg.split(" ", 1)
            if in_package(module, old):
                module = new + module[len(old) :]
                chunk = f"c{module}\n{name}\n".encode()
        out += chunk
    if f"{old}.".encode() in out:
        raise ValueError(f"'{old}' still referenced after conversion, cannot convert")
    return bytes(out)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("model", help="model file saved by macefork or mace")
    parser.add_argument(
        "output",
        nargs="?",
        help="converted model (default: <model>_mace.model or <model>_macefork.model)",
    )
    args = parser.parse_args()

    model = Path(args.model)
    with zipfile.ZipFile(model) as src:
        pkl = next(name for name in src.namelist() if name.endswith("/data.pkl"))
        old = find_package(src.read(pkl))
    new = MACE if old == FORK else FORK

    output = Path(args.output or model.with_name(f"{model.stem}_{new}.model"))
    if output.resolve() == model.resolve():
        raise SystemExit("output must differ from the input model")

    with zipfile.ZipFile(model) as src, zipfile.ZipFile(output, "w") as dst:
        for entry in src.infolist():
            data = src.read(entry)
            if entry.filename.endswith("/data.pkl"):
                data = rename_modules(data, old, new)
            dst.writestr(entry, data)
    print(f"{old} -> {new}: written {output}")


if __name__ == "__main__":
    main()
