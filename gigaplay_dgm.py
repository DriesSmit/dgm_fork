from __future__ import annotations

import argparse
import json
import os
import random
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

from gigaplay_runner import run_eval


def _default_root() -> Path:
    repo_default = Path(__file__).resolve().parent.parent / "repo" / "gigaplay"
    env_default = Path(os.environ.get("GIGAPLAY_ROOT", repo_default))
    return env_default


def _sample_params() -> Dict[str, float]:
    # Lightweight random search over a small parameter space.
    return {
        "threshold": random.uniform(-1e-3, 1e-3),
        "deadband": random.uniform(0.0, 5e-4),
        "scale": random.uniform(0.5, 1.0),
    }


def _evaluate_candidate(
    *,
    params: Dict[str, float],
    gigaplay_root: Path,
    config_file: Path,
    algorithm: str,
    environment: str,
    uv_env: Optional[Path],
    output_dir: Path,
    seed: Optional[int],
) -> Dict[str, object]:
    env_overrides = {
        "GIGAPLAY_DGM_THRESHOLD": params["threshold"],
        "GIGAPLAY_DGM_DEADBAND": params["deadband"],
        "GIGAPLAY_DGM_SCALE": params["scale"],
    }
    record = run_eval(
        gigaplay_root=gigaplay_root,
        algorithm=algorithm,
        environment=environment,
        config_file=config_file,
        seed=seed,
        uv_env=uv_env,
        output_dir=output_dir,
        env_overrides=env_overrides,
    )
    record["params"] = params
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description="Simple Gigaplay DGM loop: random search over ss_dgm params.")
    parser.add_argument("--gigaplay-root", type=Path, default=_default_root())
    parser.add_argument("--config-file", type=Path, default=Path("configs/config.toml"))
    parser.add_argument("--algorithm", type=str, default="ss_dgm")
    parser.add_argument("--environment", type=str, default="simple_sine")
    parser.add_argument("--uv-env", type=Path, default=None, help="Path to UV venv (e.g., gigaplay .venv-*)")
    parser.add_argument("--output-dir", type=Path, default=Path("dgm/output_gigaplay"))
    parser.add_argument("--trials", type=int, default=8, help="Number of random candidates to try (includes baseline)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(int(args.seed))

    gigaplay_root = args.gigaplay_root.resolve()
    config_file = (gigaplay_root / args.config_file) if not args.config_file.is_absolute() else args.config_file
    uv_env = args.uv_env.resolve() if args.uv_env is not None else None

    # Create an optimization run directory.
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    optim_dir = args.output_dir if args.output_dir.is_absolute() else Path.cwd() / args.output_dir
    optim_dir = optim_dir / "optim" / stamp
    optim_dir.mkdir(parents=True, exist_ok=True)
    log_path = optim_dir / "records.jsonl"

    # Evaluate baseline params first (no overrides beyond defaults).
    baseline_params = {"threshold": 0.0, "deadband": 0.0, "scale": 1.0}
    best_record = _evaluate_candidate(
        params=baseline_params,
        gigaplay_root=gigaplay_root,
        config_file=config_file,
        algorithm=args.algorithm,
        environment=args.environment,
        uv_env=uv_env,
        output_dir=optim_dir,
        seed=args.seed,
    )
    best_score = best_record["metrics"]["total_reward"]["avg"]

    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(best_record) + "\n")

    remaining_trials = max(0, args.trials - 1)
    for _ in range(remaining_trials):
        params = _sample_params()
        record = _evaluate_candidate(
            params=params,
            gigaplay_root=gigaplay_root,
            config_file=config_file,
            algorithm=args.algorithm,
            environment=args.environment,
            uv_env=uv_env,
            output_dir=optim_dir,
            seed=args.seed,
        )
        score = record["metrics"]["total_reward"]["avg"]

        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")

        if score > best_score:
            best_score = score
            best_record = record

    # Write best record and params for easy inspection.
    (optim_dir / "best.json").write_text(json.dumps(best_record, indent=2))
    print(json.dumps({"best_avg_total_reward": best_score, "best_params": best_record["params"]}, indent=2))


if __name__ == "__main__":
    import os

    main()
