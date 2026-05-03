import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, roc_curve, f1_score
)
from sklearn.preprocessing import label_binarize

CLASS_NAMES = ['Normal (N)', 'SVEB (A)', 'VEB (V)', 'Fusion (F)', 'Unknown (Q)']


def get_predictions(model, loader, device):
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for X, y in loader:
            X = X.to(device)
            logits = model(X)
            probs = torch.softmax(logits, dim=1)
            all_preds.extend(probs.argmax(dim=1).cpu().numpy())
            all_labels.extend(y.numpy())
            all_probs.extend(probs.cpu().numpy())
    return np.array(all_preds), np.array(all_labels), np.array(all_probs)


def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=CLASS_NAMES,
                yticklabels=CLASS_NAMES, ax=axes[0], cmap='Blues')
    axes[0].set_title('Confusion Matrix (counts)')
    axes[0].set_ylabel('True')
    axes[0].set_xlabel('Predicted')
    sns.heatmap(cm_norm, annot=True, fmt='.2f', xticklabels=CLASS_NAMES,
                yticklabels=CLASS_NAMES, ax=axes[1], cmap='Blues')
    axes[1].set_title('Confusion Matrix (normalized)')
    axes[1].set_ylabel('True')
    axes[1].set_xlabel('Predicted')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=150)
    plt.show()


def plot_roc_curves(y_true, y_probs, n_classes=5):
    y_bin = label_binarize(y_true, classes=list(range(n_classes)))
    plt.figure(figsize=(8, 6))
    auc_scores = []
    for i, name in enumerate(CLASS_NAMES):
        if y_bin[:, i].sum() == 0:
            continue
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_probs[:, i])
        auc = roc_auc_score(y_bin[:, i], y_probs[:, i])
        auc_scores.append(auc)
        plt.plot(fpr, tpr, label=f'{name} (AUC={auc:.3f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves per Class')
    plt.legend()
    plt.savefig('roc_curves.png', dpi=150)
    plt.show()
    return auc_scores


def full_evaluation(model, test_loader, device):
    preds, labels, probs = get_predictions(model, test_loader, device)
    print("=" * 60)
    print("PER-CLASS CLASSIFICATION REPORT")
    print("=" * 60)
    print(classification_report(labels, preds, target_names=CLASS_NAMES, digits=4))
    macro_f1 = f1_score(labels, preds, average='macro')
    print(f"Macro F1 Score: {macro_f1:.4f}  <- report this, not overall accuracy")
    plot_confusion_matrix(labels, preds)
    auc_scores = plot_roc_curves(labels, probs)
    print(f"Mean AUC: {np.mean(auc_scores):.4f}")
    return preds, labels, probs