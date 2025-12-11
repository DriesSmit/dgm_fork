import json
import os
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional

# Root of the gigaplay repo; override via env var if needed.
GIGAPLAY_ROOT = Path(os.environ.get("GIGAPLAY_ROOT", "/home/dries/Develop/TufaLabs/Repos/Hackathon/repo/gigaplay")).resolve()
DEFAULT_CONFIG = GIGAPLAY_ROOT / "configs" / "config.toml"
DEFAULT_ALGO = "ss_dgm"
DEFAULT_ENV = "simple_sine"


def tool_info() -> dict[str, Any]:
    return {
        "name": "gigaplay_eval",
        "description": (
            "Evaluate the gigaplay hardcoded DGM agent (ss_dgm) using run_manager.py and return averaged metrics.\n"
            "Edits are assumed to be applied only to algos/hardcoded/ss_dgm.py. "
            "This tool runs the gigaplay eval command and reads eval_metrics.json from the latest run."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "algorithm": {"type": "string", "description": "Algorithm key to evaluate", "default": DEFAULT_ALGO},
                "environment": {"type": "string", "description": "Environment key", "default": DEFAULT_ENV},
                "config_file": {"type": "string", "description": "Path to gigaplay config.toml", "default": str(DEFAULT_CONFIG)},
                "seed": {"type": "integer", "description": "Optional seed override"},
            },
            "required": [],
        },
    }


def _latest_run_dir(runs_root: Path) -> Optional[Path]:
    if not runs_root.exists():
        return None
    dirs = [p for p in runs_root.iterdir() if p.is_dir()]
    if not dirs:
        return None
    return max(dirs, key=lambda p: p.stat().st_mtime)


def _load_metrics(run_dir: Path) -> Dict[str, Any]:
    metrics_path = run_dir / "eval_metrics.json"
    if not metrics_path.exists():
        raise FileNotFoundError(f"eval_metrics.json not found in {run_dir}")
    with metrics_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def tool_function(
    algorithm: str = DEFAULT_ALGO,
    environment: str = DEFAULT_ENV,
    config_file: str = str(DEFAULT_CONFIG),
    seed: Optional[int] = None,
) -> str:
    """
    Run gigaplay eval and return averaged metrics as JSON.
    """
    if not GIGAPLAY_ROOT.exists():
        return f"Error: gigaplay root not found at {GIGAPLAY_ROOT}"

    runs_root = GIGAPLAY_ROOT / "runs"
    before = _latest_run_dir(runs_root)

    cmd = [
        "uv",
        "run",
        "python",
        "run_manager.py",
        "--config-file",
        config_file,
        "eval",
        "--algorithm",
        algorithm,
        "--environment",
        environment,
    ]
    if seed is not None:
        cmd.extend(["--seed", str(int(seed))])

    env = os.environ.copy()
    env.setdefault("UV_PROJECT_ENVIRONMENT", str(GIGAPLAY_ROOT / ".venv-linux-x86_64"))

    try:
        subprocess.run(cmd, cwd=GIGAPLAY_ROOT, env=env, check=True, text=True)
    except subprocess.CalledProcessError as exc:
        return f"Error running eval: {exc}"

    after = _latest_run_dir(runs_root)
    if after is None or before == after:
        return "Error: no new run directory detected under runs/"

    try:
        metrics = _load_metrics(after)
    except Exception as exc:
        return f"Error loading metrics from {after}: {exc}"

    payload = {
        "run_dir": str(after),
        "metrics": metrics,
    }
    return json.dumps(payload, indent=2)


if __name__ == "__main__":
    print(tool_function())
