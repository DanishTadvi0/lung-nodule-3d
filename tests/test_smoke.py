"""End-to-end smoke test on synthetic data. Run:  pytest -q  (or python tests/test_smoke.py)"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _run(*args):
    r = subprocess.run([sys.executable, *args], cwd=ROOT, capture_output=True, text=True)
    print(r.stdout[-2000:]); print(r.stderr[-2000:])
    assert r.returncode == 0, f"command failed: {' '.join(args)}"
    return r


def test_pipeline(tmp_path=None):
    _run("scripts/make_synthetic_data.py", "--n", "120", "--patch-vox", "48")
    _run("-m", "src.engine.train",
         "--train.epochs", "2", "--train.num_workers", "0",
         "--train.batch_size", "8", "--data.patch_size", "40",
         "--data.raw_patch_size", "48", "--model.depth", "10",
         "--output.run_name", "smoke")
    assert (ROOT / "artifacts" / "smoke" / "best.pt").exists()
    _run("-m", "src.engine.evaluate", "--run", "artifacts/smoke")


if __name__ == "__main__":
    test_pipeline()
    print("\nSMOKE TEST PASSED")
