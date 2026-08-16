"""Sequence-data provenance and adapter utilities for Step 9.

The adapter parses traceable downloaded source files without executing them.
It does not fabricate material properties, labels, or optimization results.
"""

from __future__ import annotations

import ast
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.clt import LaminaMaterial
from src.stacking_sequence import normalize_sequence, validate_sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SEQUENCE_ROOT = PROJECT_ROOT / "data" / "sequence"
TU_DELFT_SOURCE = SEQUENCE_ROOT / "tu_delft_zenodo_15864524"

REQUIRED_STIFFNESS_FIELDS = ("E11", "E22", "G12", "nu12", "ply_thickness")
REQUIRED_STRENGTH_FIELDS = ("Xt", "Xc", "Yt", "Yc", "S")


@dataclass(frozen=True)
class SequenceSourceRecord:
    """Parsed sequence-source summary."""

    file_name: str
    sequence: list[int]
    ply_count: int
    material_properties: dict[str, float]
    load_case: dict[str, float]
    reference_outputs: dict[str, float]
    missing_strength_properties: list[str]
    clt_compatible: bool


def load_json(path: Path) -> dict[str, Any]:
    """Load JSON file."""
    return json.loads(path.read_text(encoding="utf-8"))


def load_source_metadata(source_dir: Path = TU_DELFT_SOURCE) -> dict[str, Any]:
    """Load source-specific provenance metadata."""
    return load_json(source_dir / "metadata.json")


def validate_source_metadata(metadata: dict[str, Any]) -> list[str]:
    """Return missing required provenance metadata fields."""
    required = [
        "source_name",
        "doi",
        "url",
        "authors",
        "license",
        "dataset_type",
        "raw_files",
        "transformations_performed",
        "limitations",
    ]
    return [field for field in required if not metadata.get(field)]


def list_raw_source_files(source_dir: Path = TU_DELFT_SOURCE) -> list[Path]:
    """List preserved raw Python source files."""
    return sorted((source_dir / "raw").glob("*.py"))


def parse_sequence(value: object) -> list[int]:
    """Parse stacking sequence from list-like object or string."""
    if isinstance(value, str):
        cleaned = value.strip().replace("[", "").replace("]", "")
        if not cleaned:
            raise ValueError("sequence string is empty")
        parts = [part.strip() for part in cleaned.replace("/", ",").split(",")]
        sequence = [int(float(part.replace("+", ""))) for part in parts if part]
    else:
        sequence = list(normalize_sequence(value))
    result = validate_sequence(sequence, allowed_angles=(-90, -45, 0, 45, 90))
    if not result.valid:
        raise ValueError("; ".join(result.reasons))
    return sequence


def validate_sequence_length(sequence: list[int], expected_ply_count: int) -> bool:
    """Return True when sequence has expected ply count."""
    return len(sequence) == expected_ply_count


def extract_python_source_record(path: Path) -> SequenceSourceRecord:
    """Parse constants, ref stack, loads, and reference outputs from source file."""
    namespace = _safe_assignment_namespace(path.read_text(encoding="utf-8", errors="ignore"))
    sequence = parse_sequence(namespace["ref_stack"])
    material = {
        key: float(namespace[key])
        for key in REQUIRED_STIFFNESS_FIELDS
        if key in namespace
    }
    load_case = {
        "load_Nxx_lb_per_in": float(namespace.get("load_Nxx", 0.0)),
        "load_Nyy_lb_per_in": float(namespace.get("load_Nyy", 0.0)),
    }
    outputs = _extract_reference_outputs(path.read_text(encoding="utf-8", errors="ignore"))
    missing_stiffness = [field for field in REQUIRED_STIFFNESS_FIELDS if field not in material]
    missing_strength = [field for field in REQUIRED_STRENGTH_FIELDS if field not in namespace]
    return SequenceSourceRecord(
        file_name=path.name,
        sequence=sequence,
        ply_count=len(sequence),
        material_properties=material,
        load_case=load_case,
        reference_outputs=outputs,
        missing_strength_properties=missing_strength,
        clt_compatible=not missing_stiffness,
    )


def parse_material_card(record: SequenceSourceRecord) -> LaminaMaterial:
    """Convert parsed source stiffness values to LaminaMaterial with SI units."""
    missing = detect_missing_material_properties(record)
    if missing["stiffness"]:
        raise ValueError(f"Missing stiffness properties: {missing['stiffness']}")
    props = record.material_properties
    return LaminaMaterial(
        name=f"TU Delft parsed lamina from {record.file_name}",
        e1_pa=props["E11"],
        e2_pa=props["E22"],
        g12_pa=props["G12"],
        nu12=props["nu12"],
        ply_thickness_m=props["ply_thickness"],
    )


def detect_missing_material_properties(record: SequenceSourceRecord) -> dict[str, list[str]]:
    """Return missing stiffness and strength fields for CLT/failure usage."""
    return {
        "stiffness": [
            field for field in REQUIRED_STIFFNESS_FIELDS if field not in record.material_properties
        ],
        "strength": list(record.missing_strength_properties),
    }


def lb_per_in_to_n_per_m(value: float) -> float:
    """Convert line load from lb/in to N/m."""
    if not math.isfinite(float(value)):
        raise ValueError("load value must be finite")
    return float(value) * 4.448222 / (25.4 / 1000.0)


def assess_clt_compatibility(record: SequenceSourceRecord) -> dict[str, Any]:
    """Assess record compatibility with current CLT evaluator."""
    missing = detect_missing_material_properties(record)
    return {
        "stiffness_available": not missing["stiffness"],
        "strength_allowables_available": not missing["strength"],
        "sequence_available": bool(record.sequence),
        "load_case_available": bool(record.load_case),
        "reference_outputs_available": bool(record.reference_outputs),
        "compatible_for_stiffness_clt": not missing["stiffness"] and bool(record.sequence),
        "compatible_for_failure_optimization": (
            not missing["stiffness"]
            and not missing["strength"]
            and bool(record.sequence)
            and bool(record.load_case)
        ),
        "missing": missing,
    }


def build_sequence_inventory(source_dir: Path = TU_DELFT_SOURCE) -> dict[str, Any]:
    """Build dataset inventory entry from preserved source files."""
    metadata = load_source_metadata(source_dir)
    records = [extract_python_source_record(path) for path in list_raw_source_files(source_dir)]
    ply_counts = sorted({record.ply_count for record in records})
    all_outputs = sorted({key for record in records for key in record.reference_outputs})
    compatibility = [assess_clt_compatibility(record) for record in records]
    return {
        "name": metadata["source_name"],
        "source": metadata["url"],
        "doi": metadata["doi"],
        "provenance": {
            "authors": metadata["authors"],
            "publisher": metadata["publisher"],
            "license": metadata["license"],
            "download_date": metadata["download_date"],
        },
        "type": metadata["dataset_type"],
        "fields": {
            "records": len(records),
            "ply_counts": ply_counts,
            "material_properties": list(REQUIRED_STIFFNESS_FIELDS),
            "missing_strength_properties": list(REQUIRED_STRENGTH_FIELDS),
            "outputs": all_outputs,
        },
        "usable_for_clt": all(item["compatible_for_stiffness_clt"] for item in compatibility),
        "usable_for_sequence_ml": False,
        "limitations": metadata["limitations"],
        "status": "B - PARTIALLY USABLE",
    }


def _extract_reference_outputs(text: str) -> dict[str, float]:
    """Extract asserted reference buckling/failure loads from comments."""
    outputs: dict[str, float] = {}
    patterns = {
        "reference_failure_load_lambda_cs": r"np\.isclose\(ref_lambda_cs,\s*([0-9.]+)",
        "reference_buckling_load_lambda_cb": r"np\.isclose\(ref_lambda_cb,\s*([0-9.]+)",
    }
    for key, pattern in patterns.items():
        match = re.search(pattern, text)
        if match:
            outputs[key] = float(match.group(1))
    return outputs


def _safe_assignment_namespace(text: str) -> dict[str, Any]:
    """Evaluate safe top-level assignments needed for source inspection."""
    namespace: dict[str, Any] = {}
    tree = ast.parse(text)
    for node in tree.body:
        if isinstance(node, ast.Assign):
            try:
                value = _safe_eval(node.value, namespace)
            except ValueError:
                continue
            for target in node.targets:
                if isinstance(target, ast.Name):
                    namespace[target.id] = value
    return namespace


def _safe_eval(node: ast.AST, namespace: dict[str, Any]) -> Any:
    """Evaluate numeric/list expressions without executing source code."""
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.List):
        return [_safe_eval(item, namespace) for item in node.elts]
    if isinstance(node, ast.Name) and node.id in namespace:
        return namespace[node.id]
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        value = _safe_eval(node.operand, namespace)
        return value if isinstance(node.op, ast.UAdd) else -value
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        return _safe_eval(node.left, namespace) + _safe_eval(node.right, namespace)
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mult):
        left = _safe_eval(node.left, namespace)
        right = _safe_eval(node.right, namespace)
        return left * right
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
        return _safe_eval(node.left, namespace) / _safe_eval(node.right, namespace)
    if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name):
        value = namespace[node.value.id]
        if isinstance(node.slice, ast.Slice):
            step = _safe_eval(node.slice.step, namespace) if node.slice.step else None
            return value[slice(None, None, step)]
    raise ValueError(f"unsupported expression: {ast.dump(node)}")
