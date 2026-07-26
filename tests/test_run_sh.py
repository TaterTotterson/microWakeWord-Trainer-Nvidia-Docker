from __future__ import annotations

import os
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
RUN_SH = REPO_ROOT / "run.sh"


def _cuda_path_probe() -> str:
    source = RUN_SH.read_text(encoding="utf-8")
    match = re.search(
        r'WHISPER_CUDA_LIBRARY_PATH="\$\("\$\{PY\}" - <<\'PY\'\n(?P<probe>.*?)\nPY\n\)"',
        source,
        flags=re.DOTALL,
    )
    if match is None:
        raise AssertionError("Could not locate the CUDA library path probe in run.sh")
    return match.group("probe")


class RunShCudaLibraryPathTests(unittest.TestCase):
    def _run_probe(self, python_path: Path) -> subprocess.CompletedProcess[str]:
        env = dict(os.environ)
        env["PYTHONPATH"] = str(python_path)
        return subprocess.run(
            [sys.executable, "-S", "-"],
            input=_cuda_path_probe(),
            text=True,
            capture_output=True,
            check=False,
            env=env,
        )

    def test_namespace_cuda_packages_do_not_require_module_file(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            cublas_lib = root / "nvidia" / "cublas" / "lib"
            cudnn_lib = root / "nvidia" / "cudnn" / "lib"
            cublas_lib.mkdir(parents=True)
            cudnn_lib.mkdir(parents=True)

            result = self._run_probe(root)

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(
                result.stdout.strip().split(":"),
                [str(cublas_lib.resolve()), str(cudnn_lib.resolve())],
            )

    def test_missing_cuda_packages_return_an_empty_path(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            result = self._run_probe(Path(temp_dir))

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), "")

    def test_parakeet_uses_cuda_onnxruntime_package(self) -> None:
        source = RUN_SH.read_text(encoding="utf-8")

        self.assertIn('"onnx-asr[hub]>=0.12.0"', source)
        self.assertIn('"onnxruntime-gpu[cuda,cudnn]<1.27"', source)
        self.assertIn('"CUDAExecutionProvider"', source)


if __name__ == "__main__":
    unittest.main()
