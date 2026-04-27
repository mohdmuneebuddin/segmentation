import numpy as np

import matplotlib.pyplot as plt

from sklearn.metrics import (

    roc_curve, roc_auc_score,

    precision_recall_curve, average_precision_score,

    confusion_matrix,

    precision_score, recall_score, f1_score, accuracy_score,

)
 
 
def plot_evaluation(y_true, y_scores, threshold, save_dir="plots"):

    """

    Produce a complete, non-redundant set of evaluation plots for binary

    anomaly detection.
 
    Args:

        y_true:    array-like of 0/1 labels (0 = good, 1 = defective)

        y_scores:  array-like of continuous anomaly scores (higher = more anomalous)

        threshold: float, decision threshold used to binarize y_scores

        save_dir:  folder to write PNGs into

    """

    import os

    os.makedirs(save_dir, exist_ok=True)
 
    y_true = np.asarray(y_true)

    y_scores = np.asarray(y_scores)

    y_pred = (y_scores > threshold).astype(int)
 
    # -----------------------------------------------------------------

    # 1. ROC curve + AUC

    # -----------------------------------------------------------------

    fpr, tpr, _ = roc_curve(y_true, y_scores)

    auc = roc_auc_score(y_true, y_scores)
 
    plt.figure(figsize=(6, 5))

    plt.plot(fpr, tpr, color="#1f77b4", lw=2, label=f"ROC (AUC = {auc:.3f})")

    plt.plot([0, 1], [0, 1], "k--", lw=1, label="Random classifier")

    plt.xlabel("False Positive Rate")

    plt.ylabel("True Positive Rate")

    plt.title("ROC Curve")

    plt.legend(loc="lower right")

    plt.grid(alpha=0.3)

    plt.tight_layout()

    plt.savefig(f"{save_dir}/01_roc_curve.png", dpi=150)

    plt.close()

    print(f"[saved] {save_dir}/01_roc_curve.png   AUC = {auc:.4f}")
 
    # -----------------------------------------------------------------

    # 2. Precision-Recall curve + Average Precision

    # -----------------------------------------------------------------

    precisions, recalls, _ = precision_recall_curve(y_true, y_scores)

    ap = average_precision_score(y_true, y_scores)
 
    # Baseline = positive class prevalence (the AP a random classifier gets)

    baseline = y_true.mean()
 
    plt.figure(figsize=(6, 5))

    plt.plot(recalls, precisions, color="#2ca02c", lw=2, label=f"PR (AP = {ap:.3f})")

    plt.axhline(baseline, color="k", ls="--", lw=1,

                label=f"Random (= prevalence {baseline:.2f})")

    plt.xlabel("Recall")

    plt.ylabel("Precision")

    plt.title("Precision–Recall Curve")

    plt.legend(loc="lower left")

    plt.grid(alpha=0.3)

    plt.tight_layout()

    plt.savefig(f"{save_dir}/02_pr_curve.png", dpi=150)

    plt.close()

    print(f"[saved] {save_dir}/02_pr_curve.png    AP  = {ap:.4f}")
 
    # -----------------------------------------------------------------

    # 3. Confusion matrix heatmap

    # -----------------------------------------------------------------

    cm = confusion_matrix(y_true, y_pred)

    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)  # row-normalized
 
    fig, ax = plt.subplots(figsize=(6, 5))

    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
 
    classes = ["Good (0)", "Defective (1)"]

    ax.set_xticks([0, 1]); ax.set_xticklabels(classes)

    ax.set_yticks([0, 1]); ax.set_yticklabels(classes)

    ax.set_xlabel("Predicted")

    ax.set_ylabel("True")

    ax.set_title(f"Confusion Matrix (threshold = {threshold:.4f})")
 
    # Annotate each cell with both the raw count and the row-normalized rate

    for i in range(2):

        for j in range(2):

            color = "white" if cm_norm[i, j] > 0.5 else "black"

            ax.text(j, i, f"{cm[i, j]}\n({cm_norm[i, j]:.1%})",

                    ha="center", va="center", color=color, fontsize=12)
 
    fig.colorbar(im, ax=ax, label="Row-normalized rate")

    plt.tight_layout()

    plt.savefig(f"{save_dir}/03_confusion_matrix.png", dpi=150)

    plt.close()

    print(f"[saved] {save_dir}/03_confusion_matrix.png")
 
    # -----------------------------------------------------------------

    # 4. Metrics bar chart (precision, recall, F1, accuracy)

    # -----------------------------------------------------------------

    metrics = {

        "Precision": precision_score(y_true, y_pred, zero_division=0),

        "Recall":    recall_score(y_true, y_pred, zero_division=0),

        "F1":        f1_score(y_true, y_pred, zero_division=0),

        "Accuracy":  accuracy_score(y_true, y_pred),

    }
 
    plt.figure(figsize=(6, 5))

    bars = plt.bar(metrics.keys(), metrics.values(),

                   color=["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"])

    plt.ylim(0, 1.05)

    plt.ylabel("Score")

    plt.title(f"Classification Metrics (threshold = {threshold:.4f})")

    plt.grid(axis="y", alpha=0.3)

    for bar, val in zip(bars, metrics.values()):

        plt.text(bar.get_x() + bar.get_width() / 2, val + 0.02,

                 f"{val:.3f}", ha="center", va="bottom", fontsize=11)

    plt.tight_layout()

    plt.savefig(f"{save_dir}/04_metrics_bar.png", dpi=150)

    plt.close()

    print(f"[saved] {save_dir}/04_metrics_bar.png")

    for k, v in metrics.items():

        print(f"        {k:<10s} = {v:.4f}")
 
    # -----------------------------------------------------------------

    # 5. Score distribution histogram (good vs defective)

    # -----------------------------------------------------------------

    plt.figure(figsize=(7, 5))

    good_scores = y_scores[y_true == 0]

    bad_scores  = y_scores[y_true == 1]
 
    bins = np.linspace(y_scores.min(), y_scores.max(), 30)

    plt.hist(good_scores, bins=bins, alpha=0.6, label=f"Good (n={len(good_scores)})",

             color="#2ca02c", edgecolor="black")

    plt.hist(bad_scores,  bins=bins, alpha=0.6, label=f"Defective (n={len(bad_scores)})",

             color="#d62728", edgecolor="black")

    plt.axvline(threshold, color="black", ls="--", lw=2,

                label=f"Threshold = {threshold:.4f}")

    plt.xlabel("Anomaly score")

    plt.ylabel("Number of images")

    plt.title("Anomaly Score Distribution by Class")

    plt.legend()

    plt.grid(alpha=0.3)

    plt.tight_layout()

    plt.savefig(f"{save_dir}/05_score_distribution.png", dpi=150)

    plt.close()

    print(f"[saved] {save_dir}/05_score_distribution.png")
 
    return {

        "auc": auc,

        "ap": ap,

        **metrics,

        "confusion_matrix": cm,

    }
 
