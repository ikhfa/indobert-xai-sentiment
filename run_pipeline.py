#!/usr/bin/env python
"""
Full pipeline runner for IndoBERT XAI Sentiment Analysis.

Runs all stages sequentially:
  1. Train IndoBERT on SmSA sentiment dataset
  2. Evaluate XAI faithfulness metrics
  3. Generate visualizations and comparison plots

Usage:
    python run_pipeline.py                  # Run full pipeline
    python run_pipeline.py --stage train    # Run only training
    python run_pipeline.py --stage xai      # Run only XAI evaluation
    python run_pipeline.py --stage viz      # Run only visualization
    python run_pipeline.py --skip train     # Skip training (use existing checkpoint)

With uv:
    uv run run_pipeline.py
    uv run run_pipeline.py --stage train
"""

import argparse
import sys
import time


def _fmt_duration(seconds: float) -> str:
    """Format seconds into human-readable duration."""
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h:
        return f"{h}h {m}m {s}s"
    if m:
        return f"{m}m {s}s"
    return f"{s}s"


def run_train():
    print("\n" + "=" * 60)
    print("STAGE 1/3: Training IndoBERT on SmSA dataset")
    print("=" * 60 + "\n")

    from src.train import train
    train()


def run_xai():
    print("\n" + "=" * 60)
    print("STAGE 2/3: Evaluating XAI faithfulness metrics")
    print("=" * 60 + "\n")

    from src.evaluate_xai import evaluate_all_methods
    from src.model import load_best_model
    from src.dataset import build_datasets
    import config

    model, tokenizer = load_best_model(attn_implementation="eager")
    _, _, test_ds = build_datasets(tokenizer)
    df, jaccard = evaluate_all_methods(model, tokenizer, test_ds.texts)

    print("\nXAI Evaluation Results:")
    print(df.to_string())
    print(f"\nResults saved to {config.RESULTS_DIR}")


def run_viz():
    print("\n" + "=" * 60)
    print("STAGE 3/3: Generating visualizations")
    print("=" * 60 + "\n")

    from src.visualize import generate_report
    from src.model import load_best_model
    import config

    model, tokenizer = load_best_model(attn_implementation="eager")
    generate_report(model, tokenizer)
    print(f"\nPlots saved to {config.PLOTS_DIR}")


STAGES = {
    "train": run_train,
    "xai": run_xai,
    "viz": run_viz,
}


def main():
    parser = argparse.ArgumentParser(
        description="Run the IndoBERT XAI Sentiment Analysis pipeline.",
    )
    parser.add_argument(
        "--stage",
        choices=list(STAGES.keys()),
        help="Run only a specific stage (default: run all).",
    )
    parser.add_argument(
        "--skip",
        choices=list(STAGES.keys()),
        nargs="+",
        default=[],
        help="Skip one or more stages (e.g. --skip train).",
    )
    args = parser.parse_args()

    if args.stage:
        stages = [args.stage]
    else:
        stages = [s for s in STAGES if s not in args.skip]

    if not stages:
        print("Nothing to run — all stages were skipped.")
        sys.exit(0)

    print("IndoBERT XAI Sentiment Analysis — Full Pipeline")
    print(f"Stages to run: {', '.join(stages)}")

    t_start = time.time()

    for name in stages:
        t_stage = time.time()
        STAGES[name]()
        print(f"\n[{name}] completed in {_fmt_duration(time.time() - t_stage)}")

    total = time.time() - t_start
    print("\n" + "=" * 60)
    print(f"Pipeline finished in {_fmt_duration(total)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
