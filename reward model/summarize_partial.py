import argparse
import math
from pathlib import Path

import pandas as pd

FRM_DIR = Path(__file__).resolve().parent
PARTIAL_PATH = FRM_DIR / "frm_intervention_results_partial.csv"
N_ROUNDS = 40
GROUPS = ["Tufa", "Aima", "Reku", "Weki"]
SECTORS = ["HC_LW", "HC_HW", "LC_HW", "LC_LW"]


def entropy_base2(counts: list[int]) -> float:
    total = sum(counts)
    if total == 0:
        return 0.0
    probs = [c / total for c in counts if c > 0]
    return -sum(p * math.log2(p) for p in probs)


def run_si(run_df: pd.DataFrame) -> float:
    group_entropies = []
    for group in GROUPS:
        g_df = run_df[run_df["chosen_group"] == group]
        counts = [len(g_df[g_df["sector"] == s]) for s in SECTORS]
        group_entropies.append(entropy_base2(counts))
    return 2.0 - sum(group_entropies) / len(group_entropies)


def main() -> None:
    parser = argparse.ArgumentParser(description="Provisional SI from partial FRM results.")
    parser.add_argument(
        "--path",
        type=Path,
        default=PARTIAL_PATH,
        help=f"Partial results CSV (default: {PARTIAL_PATH.name})",
    )
    parser.add_argument(
        "--save",
        type=Path,
        default=None,
        help="Optional path to write a one-row summary CSV",
    )
    args = parser.parse_args()

    if not args.path.exists():
        raise SystemExit(f"Not found: {args.path}")

    df = pd.read_csv(args.path)
    complete = df.groupby("run_id").filter(lambda g: len(g) >= N_ROUNDS)
    if complete.empty:
        raise SystemExit(f"No complete games ({N_ROUNDS} rounds) in {args.path.name}")

    run_sis = [(int(rid), run_si(g)) for rid, g in complete.groupby("run_id")]
    mean_si = sum(si for _, si in run_sis) / len(run_sis)

    print(f"Source: {args.path}")
    print(f"Complete games: {len(run_sis)}")
    print(f"Mean SI: {mean_si:.3f}\n")
    print("Per-run SI:")
    for run_id, si in sorted(run_sis):
        print(f"  run {run_id:>2}: {si:.3f}")

    if args.save:
        summary = pd.DataFrame([{
            "mean_si": mean_si,
            "runs_completed": len(run_sis),
            "run_sis": [si for _, si in sorted(run_sis)],
        }])
        summary.to_csv(args.save, index=False)
        print(f"\nSaved summary to {args.save}")


if __name__ == "__main__":
    main()
