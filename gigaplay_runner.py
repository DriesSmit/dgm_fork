from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional


def _latest_run_dir(runs_root: Path) -> Optional[Path]:
    if not runs_root.exists():
        return None
    dirs = [p for p in runs_root.iterdir() if p.is_dir()]
    if not dirs:
        return None
    return max(dirs, key=lambda p: p.stat().st_mtime)


def _load_summaries(run_dir: Path) -> List[Dict[str, Any]]:
    path = run_dir / "summaries.json"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found (expected rollout summaries)")
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise TypeError(f"{path} expected list of summaries")
    return data


def _aggregate(summaries: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    if not summaries:
        raise ValueError("No summaries to aggregate")
    total_rewards = [float(s["total_reward"]) for s in summaries]
    final_net_worths = [float(s["final_net_worth"]) for s in summaries]
    def stats(vals: List[float]) -> Dict[str, float]:
        return {
            "min": float(min(vals)),
            "max": float(max(vals)),
            "avg": float(sum(vals) / len(vals)),
        }
    return {
        "total_reward": stats(total_rewards),
        "final_net_worth": stats(final_net_worths),
    }


def _append_metrics(record: Dict[str, Any], output_dir: Path) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.jsonl"
    current_idx = 0
    if metrics_path.exists():
        with metrics_path.open("r", encoding="utf-8") as handle:
            for current_idx, _ in enumerate(handle, start=1):
                pass
    step = current_idx
    with metrics_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record))
        handle.write("\n")
    return step


def _log_tensorboard(record: Dict[str, Any], step: int, tb_dir: Path) -> None:
    try:
        from torch.utils.tensorboard import SummaryWriter
    except Exception:
        print("torch.utils.tensorboard not available; skipping TensorBoard logging.")
        return
    tb_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(tb_dir))
    metrics = record.get("metrics", {})
    tr = metrics.get("total_reward", {})
    nw = metrics.get("final_net_worth", {})
    writer.add_scalar("total_reward/min", tr.get("min", 0.0), step)
    writer.add_scalar("total_reward/avg", tr.get("avg", 0.0), step)
    writer.add_scalar("total_reward/max", tr.get("max", 0.0), step)
    writer.add_scalar("final_net_worth/min", nw.get("min", 0.0), step)
    writer.add_scalar("final_net_worth/avg", nw.get("avg", 0.0), step)
    writer.add_scalar("final_net_worth/max", nw.get("max", 0.0), step)
    writer.flush()
    writer.close()


def run_eval(
    *,
    gigaplay_root: Path,
    algorithm: str,
    environment: str,
    config_file: Path,
    seed: Optional[int],
    uv_env: Optional[Path],
    output_dir: Path,
) -> Dict[str, Any]:
    runs_root = gigaplay_root / "runs"
    before = _latest_run_dir(runs_root)

    cmd = [
        "uv",
        "run",
        "python",
        "run_manager.py",
        "--config-file",
        str(config_file),
        "eval",
        "--algorithm",
        algorithm,
        "--environment",
        environment,
    ]
    if seed is not None:
        cmd.extend(["--seed", str(int(seed))])

    env = os.environ.copy()
    if uv_env is not None:
        env["UV_PROJECT_ENVIRONMENT"] = str(uv_env)

    subprocess.run(cmd, cwd=gigaplay_root, env=env, check=True)

    after = _latest_run_dir(runs_root)
    if after is None or after == before:
        raise RuntimeError("No new run directory detected under runs/")

    summaries = _load_summaries(after)
    metrics = _aggregate(summaries)

    record = {
        "run_dir": str(after),
        "algorithm": algorithm,
        "environment": environment,
        "seed": seed,
        "rollouts": len(summaries),
        "metrics": metrics,
    }

    step = _append_metrics(record, output_dir)
    _log_tensorboard(record, step, output_dir / "tensorboard")
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Gigaplay eval via DGM harness and log metrics.")
    parser.add_argument("--gigaplay-root", type=Path, default=Path("/home/dries/Develop/TufaLabs/Repos/Hackathon/repo/gigaplay"))
    parser.add_argument("--algorithm", type=str, default="ss_dgm")
    parser.add_argument("--environment", type=str, default="simple_sine")
    parser.add_argument("--config-file", type=Path, default=Path("configs/config.toml"))
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--uv-env", type=Path, default=None, help="Path to UV venv (e.g., gigaplay .venv-*)")
    parser.add_argument("--output-dir", type=Path, default=Path("dgm/output_gigaplay"))
    args = parser.parse_args()

    gigaplay_root = args.gigaplay_root.resolve()
    config_file = (gigaplay_root / args.config_file) if not args.config_file.is_absolute() else args.config_file
    uv_env = args.uv_env.resolve() if args.uv_env is not None else None
    output_dir = args.output_dir if args.output_dir.is_absolute() else Path.cwd() / args.output_dir

    record = run_eval(
        gigaplay_root=gigaplay_root,
        algorithm=args.algorithm,
        environment=args.environment,
        config_file=config_file,
        seed=args.seed,
        uv_env=uv_env,
        output_dir=output_dir,
    )
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
