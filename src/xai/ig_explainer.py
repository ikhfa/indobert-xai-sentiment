"""
Integrated Gradients explainer for IndoBERT via Captum.

Computes attributions w.r.t. input token embeddings using LayerIntegratedGradients
with a zero-embedding baseline and 50 integration steps.
"""

from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from captum.attr import IntegratedGradients, LayerIntegratedGradients
from transformers import PreTrainedModel, PreTrainedTokenizerBase

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
import config
from src.dataset import normalize_indonesian


# ---------------------------------------------------------------------------
# Forward pass wrapper for Captum
# ---------------------------------------------------------------------------

class _IGModelWrapper(nn.Module):
    """
    Thin wrapper so Captum receives (input_embeds,) → logits[:, target_class].

    Captum requires a function that takes the input tensor being attributed
    and returns a scalar (or per-sample) output.
    """

    def __init__(
        self,
        model: PreTrainedModel,
        target_class: int,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor],
    ) -> None:
        super().__init__()
        self.model = model
        self.target_class = target_class
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids

    def forward(self, input_embeds: torch.Tensor) -> torch.Tensor:
        kwargs = dict(
            inputs_embeds=input_embeds,
            attention_mask=self.attention_mask,
        )
        if self.token_type_ids is not None:
            kwargs["token_type_ids"] = self.token_type_ids
        outputs = self.model(**kwargs)
        return outputs.logits[:, self.target_class]


# ---------------------------------------------------------------------------
# Core Attribution Function
# ---------------------------------------------------------------------------

def compute_ig_attributions(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    text: str,
    target_class: Optional[int] = None,
    n_steps: int = config.IG_STEPS,
) -> Dict:
    """
    Compute Integrated Gradients attributions for each input token.

    Uses LayerIntegratedGradients on the embedding layer with a zero-vector
    baseline (equivalent to padding token embedding).

    Parameters
    ----------
    model : PreTrainedModel
        Fine-tuned IndoBERT model.
    tokenizer : PreTrainedTokenizerBase
        Corresponding tokenizer.
    text : str
        Input text.
    target_class : int, optional
        Class to explain. If None, uses the predicted class.
    n_steps : int
        Number of integration steps (default 50).

    Returns
    -------
    dict
        Keys:
        - 'attributions': np.ndarray of shape (seq_len,) — L2 norm of attribution vectors
        - 'attributions_raw': np.ndarray of shape (seq_len, embed_dim) — raw attributions
        - 'tokens': List[str]
        - 'target_class': int
        - 'predicted_class': int
        - 'probs': np.ndarray of shape (num_labels,)
    """
    model.eval()
    normalized = normalize_indonesian(text)

    enc = tokenizer(
        normalized,
        max_length=config.MAX_SEQ_LEN,
        truncation=True,
        return_tensors="pt",
    )
    input_ids = enc["input_ids"].to(config.DEVICE)
    attention_mask = enc["attention_mask"].to(config.DEVICE)
    token_type_ids = enc.get("token_type_ids", None)
    if token_type_ids is not None:
        token_type_ids = token_type_ids.to(config.DEVICE)

    # Get prediction
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        probs = torch.softmax(outputs.logits, dim=-1)[0].cpu().numpy()
        predicted_class = int(probs.argmax())

    if target_class is None:
        target_class = predicted_class

    # Get embedding layer
    embedding_layer = model.bert.embeddings if hasattr(model, "bert") else \
        model.roberta.embeddings if hasattr(model, "roberta") else \
        list(model.children())[0].embeddings

    # LayerIntegratedGradients
    lig = LayerIntegratedGradients(
        lambda input_ids, token_type_ids, attention_mask: model(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
        ).logits[:, target_class],
        embedding_layer,
    )

    # Baseline: all [PAD] token IDs
    baseline_ids = torch.zeros_like(input_ids)

    attributions, delta = lig.attribute(
        inputs=input_ids,
        additional_forward_args=(token_type_ids, attention_mask),
        baselines=baseline_ids,
        n_steps=n_steps,
        return_convergence_delta=True,
        internal_batch_size=8,
    )

    # attributions: (1, seq_len, embed_dim) → squeeze → (seq_len, embed_dim)
    attr_np = attributions[0].detach().cpu().numpy()  # (seq_len, embed_dim)

    # Summarize per token: L2 norm across embedding dim
    token_scores = np.linalg.norm(attr_np, axis=-1)  # (seq_len,)

    # Normalize to [-1, 1] preserving sign by using sum instead of norm for sign info
    attr_sum = attr_np.sum(axis=-1)  # (seq_len,) — signed
    max_abs = np.abs(attr_sum).max()
    if max_abs > 1e-9:
        attr_signed_norm = attr_sum / max_abs
    else:
        attr_signed_norm = attr_sum

    tokens = tokenizer.convert_ids_to_tokens(input_ids[0].cpu().tolist())

    return {
        "attributions": token_scores,
        "attributions_signed": attr_signed_norm,
        "attributions_raw": attr_np,
        "tokens": tokens,
        "target_class": target_class,
        "predicted_class": predicted_class,
        "probs": probs,
        "convergence_delta": delta.item(),
    }


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def visualize_ig_attributions(
    tokens: List[str],
    attributions: np.ndarray,
    title: str = "Integrated Gradients",
    use_signed: bool = True,
) -> plt.Figure:
    """
    Color-coded bar chart of IG token attributions.

    Positive attributions (support prediction) are green,
    negative (oppose prediction) are red.

    Parameters
    ----------
    tokens : List[str]
        Token strings.
    attributions : np.ndarray
        Shape (seq_len,) — signed or unsigned attribution scores.
    title : str
        Plot title.
    use_signed : bool
        If True, color-code by sign; if False, use a sequential colormap.

    Returns
    -------
    matplotlib.figure.Figure
    """
    # Filter special tokens
    display_tokens = []
    display_attrs = []
    for tok, attr in zip(tokens, attributions):
        if tok not in ("[CLS]", "[SEP]", "[PAD]"):
            display_tokens.append(tok)
            display_attrs.append(attr)

    vals = np.array(display_attrs)

    if use_signed:
        colors = ["#2ecc71" if v >= 0 else "#e74c3c" for v in vals]
    else:
        norm = plt.Normalize(vmin=vals.min(), vmax=vals.max())
        colors = [cm.RdYlGn(norm(v)) for v in vals]

    fig, ax = plt.subplots(figsize=(8, max(4, len(display_tokens) * 0.4)))
    y_pos = range(len(display_tokens))
    ax.barh(list(y_pos), vals, color=colors)
    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(display_tokens, fontsize=9)
    if use_signed:
        ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Attribution Score")
    ax.set_title(title)
    ax.invert_yaxis()
    plt.tight_layout()
    return fig


def visualize_ig_html(
    tokens: List[str],
    attributions: np.ndarray,
    title: str = "IG Token Attribution",
) -> str:
    """
    Generate inline HTML with color-coded token spans for IG attributions.

    Tokens are colored from red (negative) to green (positive).

    Parameters
    ----------
    tokens : List[str]
        Token strings.
    attributions : np.ndarray
        Shape (seq_len,) signed attribution scores, normalized to [-1, 1].
    title : str
        Section title.

    Returns
    -------
    str
        HTML string.
    """

    def _color(val: float) -> str:
        # val in [-1, 1] → RGB
        if val >= 0:
            intensity = int(val * 200)
            return f"rgb({255 - intensity}, 255, {255 - intensity})"
        else:
            intensity = int(abs(val) * 200)
            return f"rgb(255, {255 - intensity}, {255 - intensity})"

    parts = [f"<h3>{title}</h3><p style='font-family:monospace;font-size:14px;'>"]
    for tok, attr in zip(tokens, attributions):
        if tok in ("[CLS]", "[SEP]", "[PAD]"):
            continue
        display_tok = tok.replace("##", "")
        bg = _color(float(attr))
        parts.append(
            f"<span style='background:{bg};padding:2px 4px;margin:1px;border-radius:3px;' "
            f"title='attr={attr:.3f}'>{display_tok}</span>"
        )
    parts.append("</p>")
    return "".join(parts)


# ---------------------------------------------------------------------------
# Summary Table
# ---------------------------------------------------------------------------

def summarize_token_scores(
    tokens: List[str],
    attributions: np.ndarray,
) -> pd.DataFrame:
    """
    Build a DataFrame summarizing per-token IG attribution scores.

    Parameters
    ----------
    tokens : List[str]
        Token strings.
    attributions : np.ndarray
        Shape (seq_len,) attribution scores.

    Returns
    -------
    pd.DataFrame
        Columns: token, attribution, rank.
        Rows sorted by |attribution| descending.
        Special tokens ([CLS], [SEP], [PAD]) excluded.
    """
    rows = []
    for tok, attr in zip(tokens, attributions):
        if tok in ("[CLS]", "[SEP]", "[PAD]"):
            continue
        rows.append({"token": tok, "attribution": float(attr)})

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df = df.sort_values("attribution", key=lambda s: s.abs(), ascending=False)
    df["rank"] = range(1, len(df) + 1)
    return df.reset_index(drop=True)


if __name__ == "__main__":
    from src.model import load_best_model

    model, tokenizer = load_best_model()
    text = "Makanannya enak banget, tempatnya juga nyaman. Recommended!"

    print("Computing Integrated Gradients...")
    ig_data = compute_ig_attributions(model, tokenizer, text)
    print(f"Tokens: {ig_data['tokens']}")
    print(f"Predicted: {config.LABEL_MAP[ig_data['predicted_class']]}")
    print(f"Convergence delta: {ig_data['convergence_delta']:.4f}")

    df = summarize_token_scores(ig_data["tokens"], ig_data["attributions_signed"])
    print("\nTop tokens:")
    print(df.head(10).to_string(index=False))

    fig = visualize_ig_attributions(
        ig_data["tokens"], ig_data["attributions_signed"], title="IG Test"
    )
    fig.savefig(str(config.PLOTS_DIR / "ig_test.png"), dpi=150, bbox_inches="tight")
    print("Plot saved.")
