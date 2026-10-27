"""Convert sensing detections into a simple Mitsuba 3 XML scene for Sionna.

The Sensing API publishes object type + coordinates; this module is one
possible consumer-side representation. It is used by the online pipeline for
an optional local XML file and by ``consume_sensing_api.py --mitsuba-xml``.

Each detected person becomes a body ellipsoid + head sphere sharing the BSDF
id ``human_body_standard``. Plain Mitsuba BSDFs cannot encode radio
conductivity, so the radio material is written as an XML comment that the
Sionna side maps to a ``RadioMaterial``.
"""

from __future__ import annotations

from dataclasses import dataclass
from xml.dom import minidom
import xml.etree.ElementTree as ET


@dataclass(frozen=True)
class HumanProxyGeometry:
    """Standard human proxy, in metres."""
    body_radius_x_m: float = 0.28
    body_radius_y_m: float = 0.18
    body_half_height_m: float = 0.62
    body_center_z_m: float = 0.92
    head_radius_m: float = 0.12
    head_center_z_m: float = 1.62


@dataclass(frozen=True)
class HumanRadioMaterial:
    name: str = "human_body_standard"
    relative_permittivity: float = 38.0
    relative_permeability: float = 1.0
    conductivity_s_per_m: float = 1.46


def _fmt(value: float) -> str:
    return f"{float(value):.9g}"


def _vec3(values) -> str:
    return ",".join(_fmt(v) for v in values)


def _add_transform(parent: ET.Element, translation, yaw_deg: float, scale) -> None:
    transform = ET.SubElement(parent, "transform", {"name": "to_world"})
    ET.SubElement(transform, "scale", {"value": _vec3(scale)})
    if abs(float(yaw_deg)) > 1e-9:
        ET.SubElement(transform, "rotate", {"z": "1", "angle": _fmt(yaw_deg)})
    ET.SubElement(transform, "translate", {"value": _vec3(translation)})


def _safe_id(text: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "_-" else "_" for ch in str(text))


def make_mitsuba_scene_xml(
    detections: list[dict],
    *,
    timestamp_utc: str = "",
    frequency_hz: float | None = None,
    geometry: HumanProxyGeometry = HumanProxyGeometry(),
    material: HumanRadioMaterial = HumanRadioMaterial(),
) -> str:
    """Return a Mitsuba XML scene with one human proxy per ``person`` detection.

    ``detections`` use the Sensing API shape:
    ``{"object_id", "object_type", "position": {"x","y","z"}, "yaw_deg", "confidence"}``.
    """
    scene = ET.Element("scene", {"version": "3.0.0"})
    scene.append(ET.Comment("Generated from sensing detections (sensing_api.mitsuba)"))
    scene.append(ET.Comment(f"timestamp_utc={timestamp_utc}"))
    radio = (
        f"SIONNA_RADIO_MATERIAL name={material.name} "
        f"relative_permittivity={_fmt(material.relative_permittivity)} "
        f"relative_permeability={_fmt(material.relative_permeability)} "
        f"conductivity_s_per_m={_fmt(material.conductivity_s_per_m)}"
    )
    if frequency_hz is not None:
        radio += f" frequency_hz={_fmt(frequency_hz)}"
    scene.append(ET.Comment(radio))

    bsdf = ET.SubElement(scene, "bsdf", {"type": "diffuse", "id": material.name})
    ET.SubElement(bsdf, "rgb", {"name": "reflectance", "value": "0.65,0.44,0.36"})

    people = [d for d in detections if d.get("object_type") == "person"]
    skipped = len(detections) - len(people)
    if skipped:
        scene.append(ET.Comment(f"{skipped} non-person detection(s) not represented"))
    if not people:
        scene.append(ET.Comment("No person detected: no human shape exported."))

    for index, det in enumerate(people):
        pos = det.get("position") or {}
        base = [float(pos.get("x", 0.0)), float(pos.get("y", 0.0)), float(pos.get("z", 0.0))]
        yaw = float(det.get("yaw_deg", 0.0))
        object_id = det.get("object_id") or f"person_{index}"
        if det.get("sensor_id"):  # several sensors may each report a person_0
            object_id = f"{det['sensor_id']}_{object_id}"
        object_id = _safe_id(object_id)
        scene.append(ET.Comment(f"{object_id} confidence={_fmt(det.get('confidence', 0.0))}"))

        body = ET.SubElement(scene, "shape", {"type": "sphere", "id": f"{object_id}_body"})
        _add_transform(
            body,
            translation=[base[0], base[1], base[2] + geometry.body_center_z_m],
            yaw_deg=yaw,
            scale=[geometry.body_radius_x_m, geometry.body_radius_y_m, geometry.body_half_height_m],
        )
        ET.SubElement(body, "ref", {"id": material.name})

        head = ET.SubElement(scene, "shape", {"type": "sphere", "id": f"{object_id}_head"})
        _add_transform(
            head,
            translation=[base[0], base[1], base[2] + geometry.head_center_z_m],
            yaw_deg=yaw,
            scale=[geometry.head_radius_m] * 3,
        )
        ET.SubElement(head, "ref", {"id": material.name})

    return minidom.parseString(ET.tostring(scene, encoding="utf-8")).toprettyxml(indent="    ")
