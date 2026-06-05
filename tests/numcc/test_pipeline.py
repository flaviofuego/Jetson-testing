import sys
import subprocess
from pathlib import Path


def test_pipeline_argparse_help():
    pipeline = Path(__file__).parent.parent.parent / "numcc" / "pipeline.py"
    result = subprocess.run(
        [sys.executable, str(pipeline), "--help"],
        capture_output=True, text=True
    )
    assert result.returncode == 0
    assert "--depth" in result.stdout
    assert "--intrinsics" in result.stdout
    assert "--n-input" in result.stdout
    assert "--n-output" in result.stdout
    assert "--name" in result.stdout
    assert "--output" in result.stdout
