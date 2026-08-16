"""Tests for Step 9 sequence-data provenance adapter."""

from __future__ import annotations

import unittest

from src.clt import LaminaMaterial
from src.sequence_data import (
    TU_DELFT_SOURCE,
    assess_clt_compatibility,
    build_sequence_inventory,
    detect_missing_material_properties,
    extract_python_source_record,
    lb_per_in_to_n_per_m,
    list_raw_source_files,
    load_source_metadata,
    parse_material_card,
    parse_sequence,
    validate_sequence_length,
    validate_source_metadata,
)


class Step9SequenceDataTest(unittest.TestCase):
    """Validate downloaded sequence-source metadata and adapter behavior."""

    @classmethod
    def setUpClass(cls) -> None:
        """Load representative TU Delft/Zenodo source record."""
        cls.metadata = load_source_metadata()
        cls.raw_files = list_raw_source_files()
        cls.record = extract_python_source_record(cls.raw_files[0])

    def test_dataset_loading(self) -> None:
        """Raw source files and metadata load from preserved folder."""
        self.assertTrue(TU_DELFT_SOURCE.exists())
        self.assertGreaterEqual(len(self.raw_files), 32)
        self.assertEqual(self.metadata["doi"], "10.5281/zenodo.15864524")

    def test_metadata_validation(self) -> None:
        """Required provenance metadata is present."""
        self.assertEqual(validate_source_metadata(self.metadata), [])
        self.assertEqual(self.metadata["license"], "mit-license")

    def test_sequence_parsing(self) -> None:
        """Sequence parser accepts string/list forms and preserves order."""
        self.assertEqual(parse_sequence("[0, +45, -45, 90]"), [0, 45, -45, 90])
        self.assertEqual(parse_sequence([90, 45, -45, 0]), [90, 45, -45, 0])

    def test_material_property_parsing(self) -> None:
        """Adapter extracts stiffness material card from raw code constants."""
        material = parse_material_card(self.record)
        self.assertIsInstance(material, LaminaMaterial)
        self.assertAlmostEqual(material.e1_pa, 127.55e9)
        self.assertAlmostEqual(material.e2_pa, 13.03e9)
        self.assertAlmostEqual(material.g12_pa, 6.41e9)
        self.assertAlmostEqual(material.nu12, 0.3)
        self.assertAlmostEqual(material.ply_thickness_m, 0.000127)

    def test_unit_conversion(self) -> None:
        """Line-load conversion lb/in -> N/m is deterministic."""
        self.assertAlmostEqual(lb_per_in_to_n_per_m(1.0), 175.1268503937008)

    def test_missing_property_detection(self) -> None:
        """Strength allowables are reported missing, not fabricated."""
        missing = detect_missing_material_properties(self.record)
        self.assertEqual(missing["stiffness"], [])
        self.assertEqual(missing["strength"], ["Xt", "Xc", "Yt", "Yc", "S"])

    def test_sequence_length_validation(self) -> None:
        """Parsed source sequence length matches 48 or 64-ply cases."""
        self.assertIn(self.record.ply_count, {48, 64})
        self.assertTrue(validate_sequence_length(self.record.sequence, self.record.ply_count))
        self.assertFalse(validate_sequence_length(self.record.sequence, self.record.ply_count + 1))

    def test_clt_compatibility(self) -> None:
        """Source is stiffness-CLT compatible but not failure-optimization ready."""
        compatibility = assess_clt_compatibility(self.record)
        self.assertTrue(compatibility["compatible_for_stiffness_clt"])
        self.assertFalse(compatibility["compatible_for_failure_optimization"])
        self.assertFalse(compatibility["strength_allowables_available"])

    def test_dataset_inventory(self) -> None:
        """Inventory classifies source as partially usable."""
        inventory = build_sequence_inventory()
        self.assertEqual(inventory["status"], "B - PARTIALLY USABLE")
        self.assertTrue(inventory["usable_for_clt"])
        self.assertFalse(inventory["usable_for_sequence_ml"])


if __name__ == "__main__":
    unittest.main()
