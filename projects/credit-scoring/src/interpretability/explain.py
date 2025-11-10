"""
Interpretabilite multi-niveaux : SHAP, LIME, EBM shape functions.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap
from lime.lime_tabular import LimeTabularExplainer

from ..config import RANDOM_STATE, FIGURES_DIR, SHAP_SAMPLE, LIME_FEATURES


def explain_with_shap(model, X_sample, feature_names, model_name,
                      output_dir=FIGURES_DIR):
    """SHAP : importance globale et summary plot.

    Parameters
    ----------
    model : modele fitted avec predict_proba
    X_sample : array (sample pour performance)
    feature_names : list
    model_name : str
    output_dir : Path

    Returns
    -------
    shap_importance : DataFrame
    """
    try:
        explainer = shap.KernelExplainer(model.predict_proba, X_sample[:50])
        shap_vals = explainer.shap_values(X_sample[:SHAP_SAMPLE])

        # Gerer les formats de sortie SHAP
        if isinstance(shap_vals, list):
            shap_vals_for_imp = shap_vals[1] if len(shap_vals) > 1 else shap_vals[0]
        else:
            shap_vals_for_imp = shap_vals

        # Importance
        mean_shap = np.abs(shap_vals_for_imp).mean(axis=0)
        feat_names_short = feature_names[:len(mean_shap)]
        shap_imp = pd.DataFrame({
            "feature": feat_names_short,
            "shap_importance": mean_shap,
        }).sort_values("shap_importance", ascending=False)

        # Summary plot
        fig, ax = plt.subplots(figsize=(10, 7))
        shap.summary_plot(shap_vals_for_imp, X_sample[:SHAP_SAMPLE],
                          feature_names=feat_names_short,
                          show=False, max_display=15)
        plt.title(f"SHAP Summary - {model_name}", fontweight="bold")
        plt.tight_layout()
        fig.savefig(output_dir / "shap_summary.png", dpi=150, bbox_inches="tight")
        plt.close()

        return shap_imp
    except Exception as e:
        print(f"SHAP error (non-bloquant) : {e}")
        return None


def explain_with_lime(model, X_train, X_oot, feature_names,
                      y_oot_probas, output_dir=FIGURES_DIR):
    """LIME : explications individuelles pour clients extremes.

    Returns
    -------
    dict avec explications high_risk et low_risk
    """
    try:
        explainer = LimeTabularExplainer(
            X_train[:300],
            feature_names=feature_names[:X_train.shape[1]],
            class_names=["Non-defaut", "Defaut"],
            mode="classification",
            random_state=RANDOM_STATE,
        )

        idx_high = np.argmax(y_oot_probas)
        idx_low = np.argmin(y_oot_probas)

        exp_high = explainer.explain_instance(
            X_oot[idx_high], model.predict_proba,
            num_features=LIME_FEATURES,
        )
        exp_low = explainer.explain_instance(
            X_oot[idx_low], model.predict_proba,
            num_features=LIME_FEATURES,
        )

        return {
            "high_risk_pd": y_oot_probas[idx_high],
            "high_risk": exp_high.as_list(),
            "low_risk_pd": y_oot_probas[idx_low],
            "low_risk": exp_low.as_list(),
        }
    except Exception as e:
        print(f"LIME error (non-bloquant) : {e}")
        return None


def plot_ebm_shape_functions(ebm, top_k=15, output_dir=FIGURES_DIR):
    """Genere les fonctions de forme EBM pour audit reglementaire.

    Parameters
    ----------
    ebm : ExplainableBoostingClassifier fitted
    top_k : int, nombre de features principales
    output_dir : Path

    Returns
    -------
    list de chemins des fichiers generes
    """
    term_names = ebm.term_names_
    term_scores = ebm.term_scores_
    bins = ebm.bins_
    std_devs = ebm.standard_deviations_

    # Trier effets principaux par importance
    importance = []
    for i, name in enumerate(term_names):
        if "&" not in name:
            importance.append((name, np.mean(np.abs(term_scores[i]))))
    importance.sort(key=lambda x: x[1], reverse=True)

    top_features = importance[:top_k]
    files = []

    # Page 1: Top 6
    f1 = _plot_shape_page(top_features[:6], ebm, term_names, term_scores, bins, std_devs,
                          f"EBM Shape Functions — Page 1/3 (Top 6)", "ebm_shapes_1.png", output_dir)
    files.append(f1)

    # Page 2: 7-12
    if len(top_features) >= 12:
        f2 = _plot_shape_page(top_features[6:12], ebm, term_names, term_scores, bins, std_devs,
                              f"EBM Shape Functions — Page 2/3 (7-12)", "ebm_shapes_2.png", output_dir)
        files.append(f2)

    # Page 3: 13-15 + interactions
    page3 = top_features[12:15]
    inter_names = [(name, np.mean(np.abs(term_scores[i])))
                   for i, name in enumerate(term_names) if "&" in name]
    page3.extend(inter_names)
    f3 = _plot_shape_page(page3, ebm, term_names, term_scores, bins, std_devs,
                          f"EBM Shape Functions — Page 3/3 (Interactions)", "ebm_shapes_3.png", output_dir)
    files.append(f3)

    # Courbes individuelles (top 9)
    for feat_name, _ in top_features[:9]:
        f = _plot_single_shape(feat_name, ebm, term_names, term_scores, bins, std_devs, output_dir)
        files.append(f)

    return files


def _plot_shape_page(feature_group, ebm, term_names, term_scores, bins, std_devs,
                     title, filename, output_dir, n_cols=3):
    """Trace une page de fonctions de forme (grille)."""
    n = len(feature_group)
    n_rows = (n + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4.5 * n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)

    for idx, (feat_name, _) in enumerate(feature_group):
        row, col = idx // n_cols, idx % n_cols
        ax = axes[row, col]

        term_idx = None
        for i, name in enumerate(term_names):
            if name == feat_name:
                term_idx = i
                break

        if term_idx is None:
            ax.set_visible(False)
            continue

        scores = np.array(term_scores[term_idx])
        n_bins = len(scores)

        # Extraire centres des bins
        if term_idx < len(bins) and bins[term_idx] is not None:
            try:
                bin_edges = np.array(bins[term_idx], dtype=float)
            except (ValueError, TypeError):
                centers = np.arange(n_bins)
            else:
                if len(bin_edges) == n_bins + 1:
                    centers = (bin_edges[:-1] + bin_edges[1:]) / 2
                elif len(bin_edges) == n_bins:
                    centers = bin_edges.astype(float)
                else:
                    centers = np.arange(n_bins)
        else:
            centers = np.arange(n_bins)

        ax.plot(centers, scores, "b-", lw=2, marker="o", ms=4)
        ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
        ax.fill_between(ax.get_xlim(), scores.min() * 0.1, scores.max() * 0.1,
                        alpha=0.03, color="green")
        ax.set_xlabel(feat_name, fontsize=10)
        ax.set_ylabel("Contribution (logit)", fontsize=10)
        ax.set_title(feat_name, fontsize=11, fontweight="bold")
        ax.grid(alpha=0.3)

    for idx in range(len(feature_group), n_rows * n_cols):
        row, col = idx // n_cols, idx % n_cols
        axes[row, col].set_visible(False)

    plt.suptitle(title, fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    fig.savefig(output_dir / filename, dpi=150, bbox_inches="tight")
    plt.close()
    return str(output_dir / filename)


def _plot_single_shape(feat_name, ebm, term_names, term_scores, bins, std_devs,
                       output_dir):
    """Trace une fonction de forme individuelle (pleine page)."""
    term_idx = [i for i, n in enumerate(term_names) if n == feat_name][0]
    scores = np.array(term_scores[term_idx])
    n_bins = len(scores)

    fig, ax = plt.subplots(figsize=(10, 5))

    if term_idx < len(bins) and bins[term_idx] is not None:
        try:
            bin_edges = np.array(bins[term_idx], dtype=float)
        except (ValueError, TypeError):
            centers = np.arange(n_bins)
        else:
            if len(bin_edges) == n_bins + 1:
                centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            elif len(bin_edges) == n_bins:
                centers = bin_edges.astype(float)
            else:
                centers = np.arange(n_bins)
    else:
        centers = np.arange(n_bins)

    ax.plot(centers, scores, "b-", lw=3, marker="o", ms=6)
    ax.axhline(y=0, color="red", linestyle="--", alpha=0.5)
    ax.set_xlabel(feat_name, fontsize=13)
    ax.set_ylabel("Contribution au logit (PD)", fontsize=13)
    ax.set_title(f"Fonction de forme EBM : {feat_name}\n(Contribution additive a la PD)",
                 fontsize=14, fontweight="bold")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(output_dir / f"ebm_shape_{feat_name}.png", dpi=150, bbox_inches="tight")
    plt.close()
    return str(output_dir / f"ebm_shape_{feat_name}.png")


def plot_ebm_importance(ebm, output_dir=FIGURES_DIR):
    """Trace l'importance des features EBM."""
    ebm_global = ebm.explain_global()
    data = ebm_global.data()
    df_ebm = pd.DataFrame({
        "feature": data["names"],
        "importance": data["scores"],
    }).sort_values("importance", ascending=False)

    fig, ax = plt.subplots(figsize=(10, 7))
    top = df_ebm.head(20).iloc[::-1]
    colors = ["#e67e22" if "&" in str(f) else "#27ae60" for f in top["feature"]]
    ax.barh(range(len(top)), top["importance"], color=colors)
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(top["feature"])
    ax.set_xlabel("Importance (Mean Absolute Score)")
    ax.set_title("EBM Feature Importance\n(Vert: effet principal, Orange: interaction)",
                 fontweight="bold")
    plt.tight_layout()
    fig.savefig(output_dir / "ebm_importance.png", dpi=150, bbox_inches="tight")
    plt.close()

    return df_ebm
