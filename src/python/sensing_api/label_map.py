"""Mapping from model class labels to physical objects and coordinates.

A fingerprinting classifier predicts a *label* (the name of a dataset folder,
for example ``empty`` or ``zone_1``). The Digital Twin needs *objects with
coordinates*. This module turns one into the other using a JSON file, so the
coordinates can be changed without retraining or touching code.

File format (``config/sensing_label_map.json``)::

    {
      "schema_version": "sensing_label_map_v1",
      "frame_id": "world",
      "units": "m",
      "labels": {
        "empty":  [],
        "zone_1": [{"object_type": "person",
                    "position": {"x": 1.0, "y": 2.0, "z": 0.0},
                    "yaw_deg": 0.0}],
        "zone_2": {"object_type": "person",
                   "position": {"x": 3.0, "y": 2.0}}
      }
    }

* An empty list means "no object present" for that label.
* A single object may be written as a dict instead of a one-element list.
* ``z`` and ``yaw_deg`` default to 0; ``object_type`` defaults to
  ``default_object_type`` (``person``).

Check a map against a trained checkpoint::

    python -m sensing_api.label_map config/sensing_label_map.json \
        --model results/ssb_sensing_analysis/model_fisher_logreg.pt
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

SCHEMA_VERSION = "sensing_label_map_v1"


class LabelMapError(ValueError):
    """Raised when a label map is malformed or does not cover the model classes."""


@dataclass(frozen=True)
class MappedObject:
    object_type: str
    x: float
    y: float
    z: float = 0.0
    yaw_deg: float = 0.0

    def as_dict(self) -> dict:
        return asdict(self)


def _finite_number(value: Any, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise LabelMapError(f"{where} must be a finite number, got {value!r}")
    return float(value)


def _parse_object(raw: Any, where: str, default_type: str) -> MappedObject:
    if not isinstance(raw, dict):
        raise LabelMapError(f"{where} must be an object, got {type(raw).__name__}")
    position = raw.get("position")
    if not isinstance(position, dict):
        raise LabelMapError(f"{where}.position must be an object with x and y")
    for axis in ("x", "y"):
        if axis not in position:
            raise LabelMapError(f"{where}.position.{axis} is required")
    object_type = raw.get("object_type", default_type)
    if not isinstance(object_type, str) or not object_type.strip():
        raise LabelMapError(f"{where}.object_type must be a non-empty string")
    return MappedObject(
        object_type=object_type.strip(),
        x=_finite_number(position["x"], f"{where}.position.x"),
        y=_finite_number(position["y"], f"{where}.position.y"),
        z=_finite_number(position.get("z", 0.0), f"{where}.position.z"),
        yaw_deg=_finite_number(raw.get("yaw_deg", 0.0), f"{where}.yaw_deg"),
    )


@dataclass(frozen=True)
class LabelMap:
    frame_id: str
    units: str
    labels: dict[str, tuple[MappedObject, ...]]
    source: str = ""

    @classmethod
    def from_dict(cls, data: dict, source: str = "") -> "LabelMap":
        if not isinstance(data, dict):
            raise LabelMapError("Label map root must be a JSON object")
        version = data.get("schema_version", SCHEMA_VERSION)
        if version != SCHEMA_VERSION:
            raise LabelMapError(f"Unsupported schema_version {version!r}; expected {SCHEMA_VERSION!r}")
        raw_labels = data.get("labels")
        if not isinstance(raw_labels, dict) or not raw_labels:
            raise LabelMapError("'labels' must be a non-empty object")
        default_type = str(data.get("default_object_type", "person"))

        labels: dict[str, tuple[MappedObject, ...]] = {}
        for label, raw in raw_labels.items():
            items = [raw] if isinstance(raw, dict) else raw
            if not isinstance(items, list):
                raise LabelMapError(f"labels.{label} must be a list of objects, an object or []")
            labels[str(label)] = tuple(
                _parse_object(item, f"labels.{label}[{i}]", default_type) for i, item in enumerate(items)
            )
        return cls(
            frame_id=str(data.get("frame_id", "world")),
            units=str(data.get("units", "m")),
            labels=labels,
            source=source,
        )

    @classmethod
    def load(cls, path: str | Path) -> "LabelMap":
        path = Path(path)
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise LabelMapError(f"Label map not found: {path}") from exc
        except json.JSONDecodeError as exc:
            raise LabelMapError(f"Invalid JSON in {path}: {exc}") from exc
        return cls.from_dict(data, source=str(path))

    def missing_labels(self, classes: Iterable[str]) -> list[str]:
        return [name for name in classes if name not in self.labels]

    def require_labels(self, classes: Iterable[str]) -> None:
        """Fail if any model class has no entry, so nothing is placed at a made-up position."""
        missing = self.missing_labels(classes)
        if missing:
            where = f" in {self.source}" if self.source else ""
            raise LabelMapError(f"Labels without coordinates{where}: {missing}. Add them to the label map.")

    def objects_for(self, label: str) -> tuple[MappedObject, ...]:
        if label not in self.labels:
            raise LabelMapError(f"Label {label!r} is not in the label map")
        return self.labels[label]

    def empty_labels(self) -> set[str]:
        return {label for label, objects in self.labels.items() if not objects}


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate a sensing label map.")
    parser.add_argument("label_map", type=Path)
    parser.add_argument("--model", type=Path, help="Fisher checkpoint whose classes must all be mapped")
    args = parser.parse_args()

    label_map = LabelMap.load(args.label_map)
    print(f"{args.label_map}: frame={label_map.frame_id} units={label_map.units}")
    for label, objects in label_map.labels.items():
        text = ", ".join(f"{o.object_type}@({o.x:g},{o.y:g},{o.z:g})" for o in objects) or "no objects"
        print(f"  {label}: {text}")
    if args.model:
        import torch

        classes = [str(c) for c in torch.load(args.model, map_location="cpu", weights_only=False)["classes"]]
        label_map.require_labels(classes)
        print(f"OK: every class of {args.model} is mapped {classes}")


if __name__ == "__main__":
    main()
