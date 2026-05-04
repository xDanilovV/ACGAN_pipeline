from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from acgan_pipeline.data.mea_loader import _culture_type_for_path, _label_for_path


def inspect_fermentation_folder(root: str | Path, label_mode: str = "class") -> dict[str, object]:
    root = Path(root)
    files = sorted(root.rglob("*.mea"))
    class_counts = Counter()
    culture_counts = Counter()
    batch_counts = defaultdict(int)
    examples = []
    for path in files:
        label = _label_for_path(path, root, None, label_mode=label_mode)
        culture_type = _culture_type_for_path(path, root) or "unknown"
        class_counts[label] += 1
        culture_counts[culture_type] += 1
        batch_counts[str(path.parent.relative_to(root))] += 1
        if len(examples) < 10:
            examples.append(
                {
                    "path": str(path),
                    "label": label,
                    "culture_type": culture_type,
                    "batch": path.parent.name,
                }
            )
    return {
        "root": str(root),
        "label_mode": label_mode,
        "num_files": len(files),
        "class_counts": dict(sorted(class_counts.items())),
        "culture_counts": dict(sorted(culture_counts.items())),
        "num_batches": len(batch_counts),
        "examples": examples,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect fermentation .mea folder layout.")
    parser.add_argument("root", type=str)
    parser.add_argument("--label-mode", choices=["class", "culture_type"], default="class")
    args = parser.parse_args()
    print(json.dumps(inspect_fermentation_folder(args.root, args.label_mode), indent=2))


if __name__ == "__main__":
    main()
