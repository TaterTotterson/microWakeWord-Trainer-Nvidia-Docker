import hashlib
import struct
import tempfile
import unittest
from pathlib import Path

from shared_feature_bank import FEATURE_BANK_FILENAME, ensure_shared_negative_feature_bank


def write_npy(path: Path) -> None:
    header = b"{'descr': '<f2', 'fortran_order': False, 'shape': (2, 3), }"
    padding = (16 - ((10 + len(header) + 1) % 16)) % 16
    payload = b"\x93NUMPY\x01\x00" + struct.pack("<H", len(header) + padding + 1)
    path.write_bytes(payload + header + b" " * padding + b"\n" + b"\0" * 12)


class SharedFeatureBankTests(unittest.TestCase):
    def test_adopts_existing_bank_without_copying(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            legacy = root / "openwakeword" / "assets" / FEATURE_BANK_FILENAME
            legacy.parent.mkdir(parents=True)
            write_npy(legacy)
            digest = hashlib.sha256(legacy.read_bytes()).hexdigest()
            shared = ensure_shared_negative_feature_bank(
                root, legacy_candidates=[legacy], expected_size=legacy.stat().st_size,
                expected_sha256=digest, expected_dtype="<f2", expected_shape=(2, 3),
            )
            self.assertTrue(shared.is_file())
            self.assertFalse(legacy.exists())


if __name__ == "__main__":
    unittest.main()
