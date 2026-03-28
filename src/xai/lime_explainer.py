"""
LIME-Text explainer for IndoBERT sentiment classification.

Wraps the model as a predict_proba function and uses LIME to generate
local linear explanations for individual predictions.
"""

import re
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from lime.lime_text import LimeTextExplainer
from transformers import PreTrainedModel, PreTrainedTokenizerBase

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
import config
from src.dataset import normalize_indonesian


# ---------------------------------------------------------------------------
# Indonesian-aware tokenizer for LIME
# ---------------------------------------------------------------------------

# Common Indonesian affixes for boundary detection
_AFFIX_PREFIXES = re.compile(
    r"^(me[mn]?|ber|ke|ter|pe[nm]?|di|se)(?=[a-z])", re.IGNORECASE
)
_AFFIX_SUFFIXES = re.compile(
    r"(kan|an|i|nya|ku|mu|lah|kah|tah|pun)$", re.IGNORECASE
)


def indonesian_word_tokenizer(text: str) -> List[str]:
    """
    Split text into words for LIME, respecting Indonesian affixes.

    Splits on whitespace and punctuation while treating emoticons as single tokens.
    Affixed words are kept whole (not split on affixes) to preserve morphology.

    Parameters
    ----------
    text : str
        Input text.

    Returns
    -------
    List[str]
        Word tokens.
    """
    # Preserve emoticons
    emoticon_pat = re.compile(
        r"(?::\)|:\(|:D|:P|;\)|:o|:O|<3|>\.<|:\||:\/|B\)|:'\(|"
        r"\^_?\^|\*_?\*|T_?T|>_?<|~_?~|=_?=|o_?o|O_?O)"
    )
    tokens = []
    last = 0
    for m in emoticon_pat.finditer(text):
        chunk = text[last: m.start()].strip()
        if chunk:
            tokens.extend(re.split(r"[\s,.!?;:()\[\]{}'\"]+", chunk))
        tokens.append(m.group())
        last = m.end()
    remainder = text[last:].strip()
    if remainder:
        tokens.extend(re.split(r"[\s,.!?;:()\[\]{}'\"]+", remainder))

    return [t for t in tokens if t]


# ---------------------------------------------------------------------------
# LIME Wrapper
# ---------------------------------------------------------------------------

class IndoBERTLimeExplainer:
    """
    LIME-Text explainer wrapping an IndoBERT sentiment model.

    Parameters
    ----------
    model : PreTrainedModel
        Fine-tuned IndoBERT model.
    tokenizer : PreTrainedTokenizerBase
        Corresponding tokenizer.
    num_features : int
        Max features (tokens) returned by LIME.
    num_samples : int
        Number of perturbed samples for LIME fitting.
    """

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        num_features: int = config.LIME_NUM_FEATURES,
        num_samples: int = config.LIME_NUM_SAMPLES,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.num_features = num_features
        self.num_samples = num_samples

        self.explainer = LimeTextExplainer(
            class_names=config.LABEL_NAMES,
            split_expression=indonesian_word_tokenizer,
            random_state=config.RANDOM_SEED,
        )
        self.model.eval()

    def _predict_proba(self, texts: List[str]) -> np.ndarray:
        """
        Predict class probabilities for a batch of texts.

        This is the callable passed to LimeTextExplainer.

        Parameters
        ----------
        texts : List[str]
            Perturbed text variants from LIME.

        Returns
        -------
        np.ndarray
            Shape (N, num_labels) probability array.
        """
        all_probs = []
        batch_size = 32

        for i in range(0, len(texts), batch_size):
            batch = [normalize_indonesian(t) for t in texts[i: i + batch_size]]
            enc = self.tokenizer(
                batch,
                max_length=config.MAX_SEQ_LEN,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            enc = {k: v.to(config.DEVICE) for k, v in enc.items()}
            with torch.no_grad():
                logits = self.model(**enc).logits
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            all_probs.append(probs)

        return np.vstack(all_probs)

    def explain_instance(
        self,
        text: str,
        num_features: Optional[int] = None,
        num_samples: Optional[int] = None,
        label: Optional[int] = None,
    ):
        """
        Generate LIME explanation for a single text instance.

        Parameters
        ----------
        text : str
            Input text to explain.
        num_features : int, optional
            Override number of features (defaults to self.num_features).
        num_samples : int, optional
            Override number of samples (defaults to self.num_samples).
        label : int, optional
            Class label to explain. If None, explains the predicted class.

        Returns
        -------
        lime.explanation.Explanation
            LIME explanation object.
        """
        n_feat = num_features or self.num_features
        n_samp = num_samples or self.num_samples

        # Determine label to explain
        if label is None:
            probs = self._predict_proba([text])[0]
            label = int(probs.argmax())

        exp = self.explainer.explain_instance(
            text,
            self._predict_proba,
            num_features=n_feat,
            num_samples=n_samp,
            labels=[label],
        )
        exp._target_label = label  # store for downstream use
        return exp

    def get_lime_feature_weights(
        self,
        exp,
        label: Optional[int] = None,
    ) -> Dict[str, float]:
        """
        Extract token → weight mapping from a LIME explanation.

        Parameters
        ----------
        exp : lime.explanation.Explanation
            Output of ``explain_instance``.
        label : int, optional
            Target label. Defaults to the predicted label stored in exp.

        Returns
        -------
        Dict[str, float]
            Mapping of token string to LIME weight (positive = supports prediction).
        """
        target = label if label is not None else getattr(exp, "_target_label", 0)
        return dict(exp.as_list(label=target))

    def plot_lime_explanation(
        self,
        exp,
        label: Optional[int] = None,
        title: str = "LIME Explanation",
    ) -> plt.Figure:
        """
        Horizontal bar chart of LIME feature weights, color-coded by direction.

        Parameters
        ----------
        exp : lime.explanation.Explanation
            Output of ``explain_instance``.
        label : int, optional
            Target label. Defaults to predicted label.
        title : str
            Plot title.

        Returns
        -------
        matplotlib.figure.Figure
        """
        target = label if label is not None else getattr(exp, "_target_label", 0)
        weights = exp.as_list(label=target)

        if not weights:
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, "No features", ha="center")
            return fig

        tokens = [w[0] for w in weights]
        values = [w[1] for w in weights]
        colors = ["#2ecc71" if v > 0 else "#e74c3c" for v in values]

        fig, ax = plt.subplots(figsize=(8, max(4, len(tokens) * 0.4)))
        y_pos = range(len(tokens))
        ax.barh(list(y_pos), values, color=colors)
        ax.set_yticks(list(y_pos))
        ax.set_yticklabels(tokens, fontsize=9)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_xlabel("LIME Weight")
        label_name = config.LABEL_MAP.get(target, str(target))
        ax.set_title(f"{title} [Class: {label_name}]")
        ax.invert_yaxis()
        plt.tight_layout()
        return fig


# ---------------------------------------------------------------------------
# Module-Level Convenience Functions
# ---------------------------------------------------------------------------

def explain_instance(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    text: str,
    num_features: int = config.LIME_NUM_FEATURES,
    num_samples: int = config.LIME_NUM_SAMPLES,
    label: Optional[int] = None,
):
    """
    Convenience wrapper: build explainer and explain one instance.

    Parameters
    ----------
    model : PreTrainedModel
    tokenizer : PreTrainedTokenizerBase
    text : str
    num_features : int
    num_samples : int
    label : int, optional

    Returns
    -------
    lime.explanation.Explanation
    """
    explainer = IndoBERTLimeExplainer(model, tokenizer, num_features, num_samples)
    return explainer.explain_instance(text, label=label)


def plot_lime_explanation(
    exp,
    label: Optional[int] = None,
    title: str = "LIME Explanation",
) -> plt.Figure:
    """
    Plot a previously computed LIME explanation.

    Parameters
    ----------
    exp : lime.explanation.Explanation
    label : int, optional
    title : str

    Returns
    -------
    matplotlib.figure.Figure
    """
    target = label if label is not None else getattr(exp, "_target_label", 0)
    weights = exp.as_list(label=target)

    if not weights:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No features", ha="center")
        return fig

    tokens = [w[0] for w in weights]
    values = [w[1] for w in weights]
    colors = ["#2ecc71" if v > 0 else "#e74c3c" for v in values]

    fig, ax = plt.subplots(figsize=(8, max(4, len(tokens) * 0.4)))
    y_pos = range(len(tokens))
    ax.barh(list(y_pos), values, color=colors)
    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(tokens, fontsize=9)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("LIME Weight")
    label_name = config.LABEL_MAP.get(target, str(target))
    ax.set_title(f"{title} [Class: {label_name}]")
    ax.invert_yaxis()
    plt.tight_layout()
    return fig


def get_lime_feature_weights(exp, label: Optional[int] = None) -> Dict[str, float]:
    """
    Extract feature weights dict from a LIME explanation.

    Parameters
    ----------
    exp : lime.explanation.Explanation
    label : int, optional

    Returns
    -------
    Dict[str, float]
    """
    target = label if label is not None else getattr(exp, "_target_label", 0)
    return dict(exp.as_list(label=target))


if __name__ == "__main__":
    from src.model import load_best_model

    model, tokenizer = load_best_model()
    text = "Aplikasinya lemot banget, sering crash. Sangat mengecewakan!"

    print("Running LIME explanation...")
    lime_exp = explain_instance(model, tokenizer, text, num_samples=100)
    weights = get_lime_feature_weights(lime_exp)
    print("Feature weights:")
    for token, weight in sorted(weights.items(), key=lambda x: abs(x[1]), reverse=True):
        print(f"  {token}: {weight:.4f}")

    fig = plot_lime_explanation(lime_exp, title="LIME Test")
    fig.savefig(str(config.PLOTS_DIR / "lime_test.png"), dpi=150, bbox_inches="tight")
    print("Plot saved.")
