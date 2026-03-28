"""
Attention Visualization for IndoBERT.

Extracts multi-head attention weights and aggregates them into per-token
importance scores using three strategies: last-layer mean, last-4-layers mean,
and attention rollout (Abnar & Zuidema, 2020).
"""

from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
import config
from src.dataset import normalize_indonesian


# ---------------------------------------------------------------------------
# Attention Extraction
# ---------------------------------------------------------------------------

def get_attention_weights(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    text: str,
) -> Dict[str, np.ndarray]:
    """
    Extract raw attention weights for all layers and heads.

    Parameters
    ----------
    model : PreTrainedModel
        Fine-tuned IndoBERT model.
    tokenizer : PreTrainedTokenizerBase
        Corresponding tokenizer.
    text : str
        Input text.

    Returns
    -------
    dict
        Keys: 'attentions' → np.ndarray of shape (num_layers, num_heads, seq_len, seq_len),
              'tokens' → List[str] of subword tokens,
              'input_ids' → np.ndarray of token IDs.
    """
    normalized = normalize_indonesian(text)
    enc = tokenizer(
        normalized,
        max_length=config.MAX_SEQ_LEN,
        truncation=True,
        return_tensors="pt",
    )
    enc = {k: v.to(config.DEVICE) for k, v in enc.items()}

    model.eval()
    with torch.no_grad():
        outputs = model(**enc, output_attentions=True)

    # outputs.attentions: tuple of (num_heads, seq_len, seq_len) per layer
    attentions = np.stack(
        [layer[0].cpu().numpy() for layer in outputs.attentions], axis=0
    )  # (num_layers, num_heads, seq_len, seq_len)

    tokens = tokenizer.convert_ids_to_tokens(enc["input_ids"][0].cpu().tolist())
    input_ids = enc["input_ids"][0].cpu().numpy()

    return {
        "attentions": attentions,
        "tokens": tokens,
        "input_ids": input_ids,
        "logits": outputs.logits.cpu().numpy(),
    }


# ---------------------------------------------------------------------------
# Aggregation Strategies
# ---------------------------------------------------------------------------

def _last_layer_mean(attentions: np.ndarray) -> np.ndarray:
    """Mean across all heads of the last layer. Shape: (seq_len, seq_len)."""
    return attentions[-1].mean(axis=0)


def _last_4_layers_mean(attentions: np.ndarray) -> np.ndarray:
    """Mean across all heads of the last 4 layers. Shape: (seq_len, seq_len)."""
    return attentions[-4:].mean(axis=(0, 1))


def _attention_rollout(attentions: np.ndarray) -> np.ndarray:
    """
    Attention rollout (Abnar & Zuidema, 2020).

    Recursively multiplies attention matrices across layers, adding the
    residual identity connection and re-normalising at each step.

    Parameters
    ----------
    attentions : np.ndarray
        Shape (num_layers, num_heads, seq_len, seq_len).

    Returns
    -------
    np.ndarray
        Shape (seq_len, seq_len) — accumulated attention from all layers.
    """
    num_layers, num_heads, seq_len, _ = attentions.shape
    rollout = np.eye(seq_len)

    for layer_idx in range(num_layers):
        # Average heads
        attn = attentions[layer_idx].mean(axis=0)  # (seq_len, seq_len)
        # Add residual and re-normalise rows
        attn = attn + np.eye(seq_len)
        attn = attn / (attn.sum(axis=-1, keepdims=True) + 1e-9)
        rollout = attn @ rollout

    return rollout


_STRATEGIES = {
    "last_layer": _last_layer_mean,
    "last_4_layers": _last_4_layers_mean,
    "rollout": _attention_rollout,
}


def get_token_importance_from_attention(
    attention_data: Dict,
    strategy: str = "rollout",
) -> np.ndarray:
    """
    Compute per-token importance scores from attention weights.

    The [CLS] token's attention to each other token is used as importance,
    since [CLS] aggregates global information for classification.

    Parameters
    ----------
    attention_data : dict
        Output of ``get_attention_weights``.
    strategy : str
        Aggregation strategy: 'last_layer', 'last_4_layers', or 'rollout'.

    Returns
    -------
    np.ndarray
        Shape (seq_len,) — importance score per token, normalized to [0, 1].
    """
    if strategy not in _STRATEGIES:
        raise ValueError(f"Unknown strategy '{strategy}'. Choose from {list(_STRATEGIES)}")

    attentions = attention_data["attentions"]
    aggregated = _STRATEGIES[strategy](attentions)  # (seq_len, seq_len)

    # [CLS] is at index 0 — use its attention row
    cls_attention = aggregated[0]  # (seq_len,)

    # Normalize
    min_v, max_v = cls_attention.min(), cls_attention.max()
    if max_v - min_v > 1e-9:
        cls_attention = (cls_attention - min_v) / (max_v - min_v)

    return cls_attention


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def visualize_attention_heatmap(
    tokens: List[str],
    weights: np.ndarray,
    title: str = "Attention Heatmap",
    layer_idx: Optional[int] = None,
    head_idx: Optional[int] = None,
) -> plt.Figure:
    """
    Visualize attention as a heatmap over token pairs.

    Parameters
    ----------
    tokens : List[str]
        Token strings.
    weights : np.ndarray
        Shape (seq_len, seq_len) or (seq_len,) for importance vector.
    title : str
        Plot title.
    layer_idx : int, optional
        For labeling only.
    head_idx : int, optional
        For labeling only.

    Returns
    -------
    matplotlib.figure.Figure
    """
    if weights.ndim == 1:
        # Bar chart for 1D importance
        fig, ax = plt.subplots(figsize=(max(8, len(tokens) * 0.6), 3))
        colors = plt.cm.RdYlGn(weights)
        ax.bar(range(len(tokens)), weights, color=colors)
        ax.set_xticks(range(len(tokens)))
        ax.set_xticklabels(tokens, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("Importance")
        ax.set_title(title)
        plt.tight_layout()
        return fig

    # 2D heatmap
    seq_len = len(tokens)
    w = weights[:seq_len, :seq_len]

    fig, ax = plt.subplots(figsize=(max(8, seq_len * 0.5), max(6, seq_len * 0.5)))
    im = ax.imshow(w, cmap="Blues", aspect="auto", vmin=0)
    ax.set_xticks(range(seq_len))
    ax.set_xticklabels(tokens, rotation=90, fontsize=7)
    ax.set_yticks(range(seq_len))
    ax.set_yticklabels(tokens, fontsize=7)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    subtitle = ""
    if layer_idx is not None:
        subtitle += f" L{layer_idx}"
    if head_idx is not None:
        subtitle += f" H{head_idx}"
    ax.set_title(f"{title}{subtitle}")
    plt.tight_layout()
    return fig


def visualize_token_importance(
    tokens: List[str],
    importance: np.ndarray,
    title: str = "Token Importance (Attention)",
    top_k: int = config.ATTENTION_TOP_K,
    prediction_label: Optional[str] = None,
) -> plt.Figure:
    """
    Bar chart of token importance with color coding by rank.

    Positive sentiment tokens shown in green, neutral in gray, top tokens highlighted.

    Parameters
    ----------
    tokens : List[str]
        Token strings.
    importance : np.ndarray
        Shape (seq_len,) importance scores.
    title : str
        Plot title.
    top_k : int
        Number of top tokens to highlight.
    prediction_label : str, optional
        Predicted sentiment label for title annotation.

    Returns
    -------
    matplotlib.figure.Figure
    """
    # Filter out special tokens for display
    display_tokens = []
    display_imp = []
    for tok, imp in zip(tokens, importance):
        if tok not in ("[CLS]", "[SEP]", "[PAD]"):
            display_tokens.append(tok)
            display_imp.append(imp)

    display_imp_arr = np.array(display_imp)
    top_indices = set(np.argsort(display_imp_arr)[-top_k:])
    colors = ["#e74c3c" if i in top_indices else "#95a5a6" for i in range(len(display_tokens))]

    fig, ax = plt.subplots(figsize=(max(8, len(display_tokens) * 0.5), 4))
    ax.barh(range(len(display_tokens)), display_imp_arr, color=colors)
    ax.set_yticks(range(len(display_tokens)))
    ax.set_yticklabels(display_tokens, fontsize=8)
    ax.set_xlabel("Importance Score")
    full_title = title
    if prediction_label:
        full_title += f" [Pred: {prediction_label}]"
    ax.set_title(full_title)
    ax.invert_yaxis()
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Top-K Token Extraction
# ---------------------------------------------------------------------------

def get_top_k_tokens(
    tokens: List[str],
    importance: np.ndarray,
    k: int = config.ATTENTION_TOP_K,
) -> List[Tuple[str, float]]:
    """
    Return the top-K most important tokens with their scores.

    Parameters
    ----------
    tokens : List[str]
        Token strings.
    importance : np.ndarray
        Importance scores per token.
    k : int
        Number of top tokens to return.

    Returns
    -------
    List[Tuple[str, float]]
        Sorted list of (token, score) pairs, descending by score.
    """
    filtered = [
        (tok, float(imp))
        for tok, imp in zip(tokens, importance)
        if tok not in ("[CLS]", "[SEP]", "[PAD]")
    ]
    return sorted(filtered, key=lambda x: x[1], reverse=True)[:k]


if __name__ == "__main__":
    from src.model import load_best_model

    model, tokenizer = load_best_model()
    text = "Aplikasi ini sangat bagus dan mudah digunakan! :)"

    print("Extracting attention weights...")
    attn_data = get_attention_weights(model, tokenizer, text)
    print(f"Tokens: {attn_data['tokens']}")
    print(f"Attention shape: {attn_data['attentions'].shape}")

    importance = get_token_importance_from_attention(attn_data, strategy="rollout")
    top_k = get_top_k_tokens(attn_data["tokens"], importance)
    print(f"\nTop-{config.ATTENTION_TOP_K} tokens (rollout):")
    for tok, score in top_k:
        print(f"  {tok}: {score:.4f}")

    fig = visualize_token_importance(attn_data["tokens"], importance, "Rollout Attention")
    fig.savefig(str(config.PLOTS_DIR / "attention_test.png"), dpi=150, bbox_inches="tight")
    print(f"Plot saved.")
