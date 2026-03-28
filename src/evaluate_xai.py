"""
Faithfulness evaluation metrics for XAI methods.

Implements AOPC, Sufficiency, Comprehensiveness, and Token Overlap (Jaccard)
for comparing Attention Visualization, LIME, and Integrated Gradients.

Usage:
    python src/evaluate_xai.py
"""

import json
import random
import sys
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import PreTrainedModel, PreTrainedTokenizerBase

sys.path.insert(0, str(Path(__file__).parent.parent))
import config
from src.dataset import normalize_indonesian
from src.model import load_best_model


# ---------------------------------------------------------------------------
# Masking Utilities
# ---------------------------------------------------------------------------

def _get_word_positions(tokens: List[str]) -> List[int]:
    """Return indices of non-special tokens."""
    return [
        i for i, t in enumerate(tokens)
        if t not in ("[CLS]", "[SEP]", "[PAD]")
    ]


def _mask_tokens(
    input_ids: torch.Tensor,
    mask_indices: List[int],
    mask_token_id: int,
) -> torch.Tensor:
    """
    Replace specified token positions with mask_token_id.

    Parameters
    ----------
    input_ids : torch.Tensor
        Shape (1, seq_len).
    mask_indices : List[int]
        Positions to mask.
    mask_token_id : int
        Token ID to use as mask (typically tokenizer.mask_token_id or pad).

    Returns
    -------
    torch.Tensor
        Masked input_ids of same shape.
    """
    masked = input_ids.clone()
    for idx in mask_indices:
        masked[0, idx] = mask_token_id
    return masked


def _get_prob(
    model: PreTrainedModel,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    token_type_ids: Optional[torch.Tensor],
    target_class: int,
) -> float:
    """Run model forward pass and return probability of target_class."""
    model.eval()
    kwargs = dict(input_ids=input_ids, attention_mask=attention_mask)
    if token_type_ids is not None:
        kwargs["token_type_ids"] = token_type_ids
    with torch.no_grad():
        logits = model(**kwargs).logits
    probs = torch.softmax(logits, dim=-1)[0].cpu().numpy()
    return float(probs[target_class])


# ---------------------------------------------------------------------------
# Metric Implementations
# ---------------------------------------------------------------------------

def aopc_score(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    text: str,
    importance: np.ndarray,
    tokens: List[str],
    target_class: int,
    k_ratios: List[float] = config.AOPC_K_RATIOS,
) -> float:
    """
    Area Over the Perturbation Curve (AOPC).

    Masks top-K% tokens (by importance) and measures the cumulative drop
    in predicted probability for the target class.

    AOPC = (1/K) * sum_k [ P(y|x) - P(y|x_mask_k) ]

    Higher values indicate a more faithful explanation.

    Parameters
    ----------
    model : PreTrainedModel
    tokenizer : PreTrainedTokenizerBase
    text : str
    importance : np.ndarray
        Per-token importance scores (length = seq_len).
    tokens : List[str]
        Token strings corresponding to importance array.
    target_class : int
    k_ratios : List[float]
        Fractions of tokens to mask at each step.

    Returns
    -------
    float
        AOPC score.
    """
    normalized = normalize_indonesian(text)
    enc = tokenizer(
        normalized,
        max_length=config.MAX_SEQ_LEN,
        truncation=True,
        padding="max_length",
        return_tensors="pt",
    )
    input_ids = enc["input_ids"].to(config.DEVICE)
    attention_mask = enc["attention_mask"].to(config.DEVICE)
    token_type_ids = enc.get("token_type_ids", None)
    if token_type_ids is not None:
        token_type_ids = token_type_ids.to(config.DEVICE)

    mask_token_id = tokenizer.mask_token_id or tokenizer.pad_token_id

    # Baseline probability
    p_orig = _get_prob(model, input_ids, attention_mask, token_type_ids, target_class)

    # Get word token positions sorted by importance (descending)
    word_positions = _get_word_positions(tokens)
    word_imp = [(pos, float(importance[pos])) for pos in word_positions]
    word_imp.sort(key=lambda x: x[1], reverse=True)

    n_words = len(word_positions)
    drops = []

    for ratio in k_ratios:
        k = max(1, int(np.ceil(ratio * n_words)))
        top_k_positions = [pos for pos, _ in word_imp[:k]]
        masked_ids = _mask_tokens(input_ids, top_k_positions, mask_token_id)
        p_masked = _get_prob(model, masked_ids, attention_mask, token_type_ids, target_class)
        drops.append(p_orig - p_masked)

    return float(np.mean(drops))


def sufficiency_score(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    text: str,
    importance: np.ndarray,
    tokens: List[str],
    target_class: int,
    k_ratio: float = 0.2,
) -> float:
    """
    Sufficiency: keep only top-K important tokens, mask the rest.

    sufficiency = P(y | all_tokens) - P(y | important_tokens_only)

    Closer to 0 means the top-K tokens are sufficient to preserve the prediction.

    Parameters
    ----------
    model : PreTrainedModel
    tokenizer : PreTrainedTokenizerBase
    text : str
    importance : np.ndarray
    tokens : List[str]
    target_class : int
    k_ratio : float
        Fraction of tokens to keep.

    Returns
    -------
    float
        Sufficiency score (lower magnitude = more sufficient).
    """
    normalized = normalize_indonesian(text)
    enc = tokenizer(
        normalized,
        max_length=config.MAX_SEQ_LEN,
        truncation=True,
        padding="max_length",
        return_tensors="pt",
    )
    input_ids = enc["input_ids"].to(config.DEVICE)
    attention_mask = enc["attention_mask"].to(config.DEVICE)
    token_type_ids = enc.get("token_type_ids", None)
    if token_type_ids is not None:
        token_type_ids = token_type_ids.to(config.DEVICE)

    mask_token_id = tokenizer.mask_token_id or tokenizer.pad_token_id

    p_full = _get_prob(model, input_ids, attention_mask, token_type_ids, target_class)

    word_positions = _get_word_positions(tokens)
    word_imp = sorted([(pos, float(importance[pos])) for pos in word_positions],
                      key=lambda x: x[1], reverse=True)
    k = max(1, int(np.ceil(k_ratio * len(word_positions))))
    top_k_set = {pos for pos, _ in word_imp[:k]}
    # Mask everything EXCEPT top-K
    to_mask = [pos for pos in word_positions if pos not in top_k_set]

    masked_ids = _mask_tokens(input_ids, to_mask, mask_token_id)
    p_suff = _get_prob(model, masked_ids, attention_mask, token_type_ids, target_class)

    return float(p_full - p_suff)


def comprehensiveness_score(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    text: str,
    importance: np.ndarray,
    tokens: List[str],
    target_class: int,
    k_ratio: float = 0.2,
) -> float:
    """
    Comprehensiveness: remove top-K important tokens, keep the rest.

    comprehensiveness = P(y | all_tokens) - P(y | remaining_tokens)

    Higher values mean the explanation captures truly important tokens.

    Parameters
    ----------
    model : PreTrainedModel
    tokenizer : PreTrainedTokenizerBase
    text : str
    importance : np.ndarray
    tokens : List[str]
    target_class : int
    k_ratio : float
        Fraction of tokens to remove.

    Returns
    -------
    float
        Comprehensiveness score (higher = more comprehensive).
    """
    normalized = normalize_indonesian(text)
    enc = tokenizer(
        normalized,
        max_length=config.MAX_SEQ_LEN,
        truncation=True,
        padding="max_length",
        return_tensors="pt",
    )
    input_ids = enc["input_ids"].to(config.DEVICE)
    attention_mask = enc["attention_mask"].to(config.DEVICE)
    token_type_ids = enc.get("token_type_ids", None)
    if token_type_ids is not None:
        token_type_ids = token_type_ids.to(config.DEVICE)

    mask_token_id = tokenizer.mask_token_id or tokenizer.pad_token_id

    p_full = _get_prob(model, input_ids, attention_mask, token_type_ids, target_class)

    word_positions = _get_word_positions(tokens)
    word_imp = sorted([(pos, float(importance[pos])) for pos in word_positions],
                      key=lambda x: x[1], reverse=True)
    k = max(1, int(np.ceil(k_ratio * len(word_positions))))
    top_k_positions = [pos for pos, _ in word_imp[:k]]

    masked_ids = _mask_tokens(input_ids, top_k_positions, mask_token_id)
    p_comp = _get_prob(model, masked_ids, attention_mask, token_type_ids, target_class)

    return float(p_full - p_comp)


def token_overlap_jaccard(
    tokens_a: List[str],
    tokens_b: List[str],
) -> float:
    """
    Jaccard similarity between two sets of top-K tokens.

    Parameters
    ----------
    tokens_a : List[str]
        Top-K tokens from method A.
    tokens_b : List[str]
        Top-K tokens from method B.

    Returns
    -------
    float
        Jaccard similarity in [0, 1].
    """
    set_a = set(tokens_a)
    set_b = set(tokens_b)
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    if union == 0:
        return 0.0
    return float(intersection / union)


# ---------------------------------------------------------------------------
# Full Evaluation Pipeline
# ---------------------------------------------------------------------------

def evaluate_all_methods(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    texts: List[str],
    n_samples: int = 50,
    top_k_overlap: int = 10,
) -> pd.DataFrame:
    """
    Evaluate all three XAI methods on a sample of texts.

    For each text, computes per-method AOPC, Sufficiency, and Comprehensiveness.
    Also computes pairwise Jaccard overlaps.

    Parameters
    ----------
    model : PreTrainedModel
    tokenizer : PreTrainedTokenizerBase
    texts : List[str]
        Pool of texts to sample from.
    n_samples : int
        Number of texts to evaluate (random sample).
    top_k_overlap : int
        K for Jaccard overlap computation.

    Returns
    -------
    pd.DataFrame
        Rows = methods, columns = metrics (mean ± std across samples).
    """
    from src.xai.attention_viz import get_attention_weights, get_token_importance_from_attention
    from src.xai.lime_explainer import IndoBERTLimeExplainer
    from src.xai.ig_explainer import compute_ig_attributions

    random.seed(config.RANDOM_SEED)
    np.random.seed(config.RANDOM_SEED)

    sample_texts = random.sample(texts, min(n_samples, len(texts)))

    lime_exp = IndoBERTLimeExplainer(model, tokenizer, num_samples=200)

    results = {
        "attention": {"aopc": [], "sufficiency": [], "comprehensiveness": []},
        "lime": {"aopc": [], "sufficiency": [], "comprehensiveness": []},
        "ig": {"aopc": [], "sufficiency": [], "comprehensiveness": []},
    }
    jaccard = {
        "attn_lime": [], "attn_ig": [], "lime_ig": []
    }

    for text in tqdm(sample_texts, desc="Evaluating XAI faithfulness"):
        try:
            # --- Attention ---
            attn_data = get_attention_weights(model, tokenizer, text)
            attn_imp = get_token_importance_from_attention(attn_data, strategy="rollout")
            attn_tokens = attn_data["tokens"]
            pred_class = int(np.argmax(torch.softmax(
                torch.tensor(attn_data["logits"]), dim=-1
            ).numpy()))

            results["attention"]["aopc"].append(
                aopc_score(model, tokenizer, text, attn_imp, attn_tokens, pred_class)
            )
            results["attention"]["sufficiency"].append(
                sufficiency_score(model, tokenizer, text, attn_imp, attn_tokens, pred_class)
            )
            results["attention"]["comprehensiveness"].append(
                comprehensiveness_score(model, tokenizer, text, attn_imp, attn_tokens, pred_class)
            )

            # --- LIME ---
            lime_explanation = lime_exp.explain_instance(text, num_samples=200)
            lime_weights = lime_exp.get_lime_feature_weights(lime_explanation)
            # Build importance array aligned to tokens
            lime_imp = np.zeros(len(attn_tokens))
            for i, tok in enumerate(attn_tokens):
                clean_tok = tok.replace("##", "")
                if clean_tok in lime_weights:
                    lime_imp[i] = abs(lime_weights[clean_tok])

            results["lime"]["aopc"].append(
                aopc_score(model, tokenizer, text, lime_imp, attn_tokens, pred_class)
            )
            results["lime"]["sufficiency"].append(
                sufficiency_score(model, tokenizer, text, lime_imp, attn_tokens, pred_class)
            )
            results["lime"]["comprehensiveness"].append(
                comprehensiveness_score(model, tokenizer, text, lime_imp, attn_tokens, pred_class)
            )

            # --- IG ---
            ig_data = compute_ig_attributions(model, tokenizer, text, target_class=pred_class)
            ig_imp = np.abs(ig_data["attributions_signed"])

            results["ig"]["aopc"].append(
                aopc_score(model, tokenizer, text, ig_imp, ig_data["tokens"], pred_class)
            )
            results["ig"]["sufficiency"].append(
                sufficiency_score(model, tokenizer, text, ig_imp, ig_data["tokens"], pred_class)
            )
            results["ig"]["comprehensiveness"].append(
                comprehensiveness_score(model, tokenizer, text, ig_imp, ig_data["tokens"], pred_class)
            )

            # --- Jaccard Overlap ---
            def _top_k_toks(tokens, imp, k):
                word_pos = _get_word_positions(tokens)
                ranked = sorted(word_pos, key=lambda i: imp[i], reverse=True)
                return [tokens[i] for i in ranked[:k]]

            attn_top = _top_k_toks(attn_tokens, attn_imp, top_k_overlap)
            lime_top = _top_k_toks(attn_tokens, lime_imp, top_k_overlap)
            ig_top = _top_k_toks(ig_data["tokens"], ig_imp, top_k_overlap)

            jaccard["attn_lime"].append(token_overlap_jaccard(attn_top, lime_top))
            jaccard["attn_ig"].append(token_overlap_jaccard(attn_top, ig_top))
            jaccard["lime_ig"].append(token_overlap_jaccard(lime_top, ig_top))

        except Exception as e:
            print(f"  [Warning] Skipped sample due to error: {e}")
            continue

    # Build summary DataFrame
    rows = []
    for method in ["attention", "lime", "ig"]:
        row = {"method": method}
        for metric in ["aopc", "sufficiency", "comprehensiveness"]:
            vals = results[method][metric]
            row[f"{metric}_mean"] = float(np.mean(vals)) if vals else float("nan")
            row[f"{metric}_std"] = float(np.std(vals)) if vals else float("nan")
        rows.append(row)

    df = pd.DataFrame(rows).set_index("method")

    # Jaccard summary
    jac_row = {
        "attn_lime_jaccard": float(np.mean(jaccard["attn_lime"])),
        "attn_ig_jaccard": float(np.mean(jaccard["attn_ig"])),
        "lime_ig_jaccard": float(np.mean(jaccard["lime_ig"])),
    }

    return df, jac_row


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def main() -> None:
    """Load model, run evaluation on test set sample, save results."""
    print(f"Device: {config.DEVICE}")
    model, tokenizer = load_best_model()

    # Load test texts
    from datasets import load_dataset
    dataset = load_dataset(config.DATASET_NAME, config.DATASET_SUBSET)
    test_texts = [str(ex["text"]) for ex in dataset["test"]]

    print(f"Evaluating on {min(50, len(test_texts))} test samples...")
    df, jaccard = evaluate_all_methods(model, tokenizer, test_texts, n_samples=50)

    print("\n=== Faithfulness Metrics ===")
    print(df.to_string())
    print("\n=== Token Overlap (Jaccard) ===")
    for k, v in jaccard.items():
        print(f"  {k}: {v:.4f}")

    # Save results
    results_path = config.RESULTS_DIR / "xai_evaluation.csv"
    df.to_csv(results_path)
    print(f"\nResults saved to {results_path}")

    jaccard_path = config.RESULTS_DIR / "jaccard_overlap.json"
    with open(jaccard_path, "w") as f:
        json.dump(jaccard, f, indent=2)
    print(f"Jaccard results saved to {jaccard_path}")


if __name__ == "__main__":
    main()
