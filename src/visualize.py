"""
Visualization and report generation for IndoBERT XAI Sentiment Analysis.

Produces:
- Attention heatmaps
- LIME bar charts
- IG token attribution plots
- Side-by-side comparison plots for all three methods
- Metric comparison radar chart

Usage:
    python src/visualize.py
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
import config
from src.dataset import normalize_indonesian
from src.model import load_best_model, predict_proba


# ---------------------------------------------------------------------------
# Side-by-Side Comparison Plot
# ---------------------------------------------------------------------------

def plot_method_comparison(
    tokens_attn: List[str],
    attn_importance: np.ndarray,
    lime_weights: Dict[str, float],
    tokens_ig: List[str],
    ig_importance: np.ndarray,
    text: str,
    pred_label: str,
    top_k: int = 10,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Three-panel comparison: top-K tokens per XAI method for the same input.

    Parameters
    ----------
    tokens_attn : List[str]
        Tokens from attention model output.
    attn_importance : np.ndarray
        Per-token importance from attention rollout.
    lime_weights : Dict[str, float]
        Token → weight from LIME.
    tokens_ig : List[str]
        Tokens from IG output.
    ig_importance : np.ndarray
        Per-token importance from IG (signed).
    text : str
        Original input text.
    pred_label : str
        Predicted sentiment label.
    top_k : int
        Number of top tokens to show per method.
    save_path : str, optional
        If provided, saves figure to this path.

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, max(5, top_k * 0.45)))
    fig.suptitle(
        f"XAI Method Comparison | Prediction: {pred_label}\n\"{text[:80]}{'...' if len(text) > 80 else ''}\"",
        fontsize=11, wrap=True
    )

    # --- Panel 1: Attention ---
    _plot_top_k_bar(
        axes[0],
        tokens_attn,
        attn_importance,
        top_k=top_k,
        title="Attention Rollout",
        color="#3498db",
    )

    # --- Panel 2: LIME ---
    # Convert LIME weights to token-aligned array using clean tokens
    def _lime_score(tok: str) -> float:
        clean = tok.replace("##", "")
        return abs(lime_weights.get(clean, 0.0))

    lime_imp = np.array([_lime_score(t) for t in tokens_attn])
    _plot_top_k_bar(
        axes[1],
        tokens_attn,
        lime_imp,
        top_k=top_k,
        title="LIME",
        color="#e67e22",
    )

    # --- Panel 3: IG ---
    ig_imp_abs = np.abs(ig_importance)
    _plot_top_k_bar(
        axes[2],
        tokens_ig,
        ig_imp_abs,
        top_k=top_k,
        title="Integrated Gradients",
        color="#27ae60",
    )

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def _plot_top_k_bar(
    ax: plt.Axes,
    tokens: List[str],
    importance: np.ndarray,
    top_k: int,
    title: str,
    color: str,
) -> None:
    """Helper: horizontal bar chart of top-K tokens on given Axes."""
    # Filter specials
    filtered = [
        (tok, float(imp))
        for tok, imp in zip(tokens, importance)
        if tok not in ("[CLS]", "[SEP]", "[PAD]")
    ]
    filtered.sort(key=lambda x: x[1], reverse=True)
    filtered = filtered[:top_k]

    if not filtered:
        ax.text(0.5, 0.5, "No tokens", ha="center", transform=ax.transAxes)
        ax.set_title(title)
        return

    labels = [f[0].replace("##", "") for f in filtered]
    values = [f[1] for f in filtered]

    y_pos = range(len(labels))
    ax.barh(list(y_pos), values, color=color, alpha=0.8)
    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Importance")
    ax.set_title(title, fontweight="bold")
    ax.invert_yaxis()


# ---------------------------------------------------------------------------
# Radar Chart for Metrics
# ---------------------------------------------------------------------------

def plot_metrics_radar(
    df_metrics: pd.DataFrame,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Radar chart comparing AOPC, Sufficiency (|val|), and Comprehensiveness
    across three XAI methods.

    Parameters
    ----------
    df_metrics : pd.DataFrame
        Output from evaluate_xai.evaluate_all_methods — index = method names,
        columns include {metric}_mean.
    save_path : str, optional
        If provided, saves figure to this path.

    Returns
    -------
    matplotlib.figure.Figure
    """
    methods = df_metrics.index.tolist()
    categories = ["AOPC", "Comprehensiveness", "1 - |Sufficiency|"]
    n_cats = len(categories)
    angles = np.linspace(0, 2 * np.pi, n_cats, endpoint=False).tolist()
    angles += angles[:1]  # close the loop

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    colors = ["#3498db", "#e67e22", "#27ae60"]

    for method, color in zip(methods, colors):
        row = df_metrics.loc[method]
        aopc = float(row.get("aopc_mean", 0))
        comp = float(row.get("comprehensiveness_mean", 0))
        suff_inv = max(0, 1 - abs(float(row.get("sufficiency_mean", 0))))
        values = [aopc, comp, suff_inv]
        values += values[:1]
        ax.plot(angles, values, "o-", linewidth=2, color=color, label=method)
        ax.fill(angles, values, alpha=0.15, color=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=11)
    ax.set_title("XAI Faithfulness Comparison", fontsize=13, pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


# ---------------------------------------------------------------------------
# Full Report Generation
# ---------------------------------------------------------------------------

EXAMPLE_TEXTS = [
    # Formal positive
    "Produk ini memiliki kualitas yang sangat baik dan harga yang terjangkau.",
    # Informal slang negative
    "Ih nyebelin banget sih, pesanan gue salah mulu. Kapok deh beli disini!",
    # With emoticons
    "Pelayanannya ramah banget :) Pasti balik lagi deh ke sini :D",
    # Mixed Indonesian-English
    "The food was amazing! Tempatnya juga cozy dan Instagrammable banget.",
    # Neutral / ambiguous
    "Biasa aja sih, tidak terlalu buruk tapi juga tidak istimewa.",
]


def generate_report(
    model,
    tokenizer,
    texts: Optional[List[str]] = None,
    save_dir: Optional[Path] = None,
) -> None:
    """
    Generate full visualization report for all example texts.

    For each text: attention heatmap, LIME bar chart, IG bar chart, comparison plot.
    Also generates radar chart if metrics CSV exists.

    Parameters
    ----------
    model : PreTrainedModel
    tokenizer : PreTrainedTokenizerBase
    texts : List[str], optional
        Texts to visualize. Defaults to built-in EXAMPLE_TEXTS.
    save_dir : Path, optional
        Directory to save plots. Defaults to config.PLOTS_DIR.
    """
    from src.xai.attention_viz import (
        get_attention_weights,
        get_token_importance_from_attention,
        visualize_token_importance,
    )
    from src.xai.lime_explainer import IndoBERTLimeExplainer, plot_lime_explanation
    from src.xai.ig_explainer import compute_ig_attributions, visualize_ig_attributions

    if texts is None:
        texts = EXAMPLE_TEXTS
    if save_dir is None:
        save_dir = config.PLOTS_DIR

    save_dir.mkdir(parents=True, exist_ok=True)
    lime_exp = IndoBERTLimeExplainer(model, tokenizer, num_samples=300)

    for idx, text in enumerate(texts):
        print(f"\n[{idx + 1}/{len(texts)}] Text: {text[:60]}...")
        safe_name = f"sample_{idx + 1:02d}"

        try:
            # Attention
            attn_data = get_attention_weights(model, tokenizer, text)
            attn_imp = get_token_importance_from_attention(attn_data, strategy="rollout")
            probs = torch.softmax(torch.tensor(attn_data["logits"]), dim=-1).numpy()[0]
            pred_class = int(probs.argmax())
            pred_label = config.LABEL_MAP[pred_class]

            fig_attn = visualize_token_importance(
                attn_data["tokens"], attn_imp,
                title=f"Attention Rollout — {pred_label}",
                prediction_label=pred_label,
            )
            fig_attn.savefig(save_dir / f"{safe_name}_attention.png", dpi=150, bbox_inches="tight")
            plt.close(fig_attn)
            print(f"  Attention plot saved.")

            # LIME
            lime_explanation = lime_exp.explain_instance(text)
            lime_weights = lime_exp.get_lime_feature_weights(lime_explanation)
            fig_lime = plot_lime_explanation(lime_explanation, title=f"LIME — {pred_label}")
            fig_lime.savefig(save_dir / f"{safe_name}_lime.png", dpi=150, bbox_inches="tight")
            plt.close(fig_lime)
            print(f"  LIME plot saved.")

            # IG
            ig_data = compute_ig_attributions(model, tokenizer, text, target_class=pred_class)
            fig_ig = visualize_ig_attributions(
                ig_data["tokens"], ig_data["attributions_signed"],
                title=f"Integrated Gradients — {pred_label}",
            )
            fig_ig.savefig(save_dir / f"{safe_name}_ig.png", dpi=150, bbox_inches="tight")
            plt.close(fig_ig)
            print(f"  IG plot saved.")

            # Comparison
            fig_cmp = plot_method_comparison(
                attn_data["tokens"], attn_imp,
                lime_weights,
                ig_data["tokens"], ig_data["attributions_signed"],
                text=text,
                pred_label=pred_label,
            )
            fig_cmp.savefig(save_dir / f"{safe_name}_comparison.png", dpi=150, bbox_inches="tight")
            plt.close(fig_cmp)
            print(f"  Comparison plot saved.")

        except Exception as e:
            print(f"  [Warning] Error processing sample {idx + 1}: {e}")
            continue

    # Radar chart (if evaluation CSV exists)
    metrics_csv = config.RESULTS_DIR / "xai_evaluation.csv"
    if metrics_csv.exists():
        df_metrics = pd.read_csv(metrics_csv, index_col=0)
        fig_radar = plot_metrics_radar(df_metrics, save_path=str(save_dir / "radar_metrics.png"))
        plt.close(fig_radar)
        print("\nRadar chart saved.")

    print(f"\nAll plots saved to: {save_dir}")


if __name__ == "__main__":
    print(f"Device: {config.DEVICE}")
    model, tokenizer = load_best_model(attn_implementation="eager")
    generate_report(model, tokenizer)
