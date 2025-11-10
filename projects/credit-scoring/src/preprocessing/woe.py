"""
WoE Binning — Weight of Evidence pour scoring credit.
Transforme chaque variable numerique en bins avec leur WoE,
remplacant les valeurs brutes par leur pouvoir predictif.
Standard bancaire pour la regression logistique.
"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


def woe_binning(df, feature, target, bins=10, min_bin_pct=0.02):
    """Binne une variable numerique et calcule son WoE.

    Parameters
    ----------
    df : DataFrame
    feature : str
    target : str
    bins : int
        Nombre de bins (qcut)
    min_bin_pct : float
        Pourcentage minimum par bin (evite les bins vides)

    Returns
    -------
    dict {bin_label: woe_value}
    DataFrame avec statistiques par bin
    """
    df_tmp = df[[feature, target]].dropna().copy()

    if len(df_tmp) < 100:
        return None, None

    try:
        df_tmp["bin"] = pd.qcut(
            df_tmp[feature],
            q=min(bins, df_tmp[feature].nunique()),
            duplicates="drop",
        )
    except Exception:
        return None, None

    grouped = df_tmp.groupby("bin", observed=True)[target].agg(["sum", "count"])
    grouped["good"] = grouped["count"] - grouped["sum"]
    grouped["bad"] = grouped["sum"]

    # Eviter les zeros avec Laplace smoothing
    grouped["good_smooth"] = grouped["good"] + 0.5
    grouped["bad_smooth"] = grouped["bad"] + 0.5

    total_good = grouped["good_smooth"].sum()
    total_bad = grouped["bad_smooth"].sum()

    grouped["pct_good"] = grouped["good_smooth"] / total_good
    grouped["pct_bad"] = grouped["bad_smooth"] / total_bad
    grouped["woe"] = np.log(grouped["pct_good"] / grouped["pct_bad"])
    grouped["iv"] = (grouped["pct_good"] - grouped["pct_bad"]) * grouped["woe"]

    # Filtrer les bins trop petits
    grouped["bin_pct"] = grouped["count"] / grouped["count"].sum()
    grouped = grouped[grouped["bin_pct"] >= min_bin_pct]

    woe_map = dict(zip(grouped.index.astype(str), grouped["woe"]))
    iv_total = grouped["iv"].sum()

    return woe_map, {"iv": iv_total, "bins": grouped, "n_bins": len(grouped)}


class WoETransformer(BaseEstimator, TransformerMixin):
    """Transformer sklearn-compatible pour le WoE binning.

    Remplace chaque variable numerique par sa valeur WoE.
    Les valeurs manquantes sont imputees avec un WoE de 0 (neutre).
    Les valeurs hors bins sont mappees au WoE du bin le plus proche.
    """

    def __init__(self, features=None, bins=10, min_bin_pct=0.02):
        self.features = features
        self.bins = bins
        self.min_bin_pct = min_bin_pct

    def fit(self, X, y):
        """Calcule le mapping WoE pour chaque feature."""
        self.mappings_ = {}
        self.iv_scores_ = {}
        self.bin_edges_ = {}

        # Determiner les noms de colonnes
        if hasattr(X, "columns"):
            self.features_ = list(X.columns)
        else:
            self.features_ = [f"X{i}" for i in range(X.shape[1])]

        df = pd.DataFrame(X, columns=self.features_)
        if hasattr(y, "name"):
            y_series = y
        else:
            y_series = pd.Series(y, name="target")

        df["__target__"] = y_series.values

        features_to_bin = self.features or [c for c in self.features_ if c != "__target__"]

        for feat in features_to_bin:
            if feat == "__target__":
                continue
            if df[feat].dtype == "object":
                continue  # Skip categoriel — deja gere par le pipeline principal

            mapping, info = woe_binning(df, feat, "__target__", bins=self.bins)
            if mapping is not None:
                # Stocker aussi les bords de bins pour le mapping de nouvelles valeurs
                bin_edges = {}
                for bin_label, woe_val in mapping.items():
                    # Extraire les bornes du string Interval
                    if hasattr(bin_label, "left") and hasattr(bin_label, "right"):
                        bin_edges[(bin_label.left, bin_label.right)] = woe_val
                    else:
                        bin_edges[str(bin_label)] = woe_val

                self.mappings_[feat] = mapping
                self.bin_edges_[feat] = bin_edges
                self.iv_scores_[feat] = info["iv"]

        return self

    def transform(self, X):
        """Applique le mapping WoE avec np.digitize (vectorise)."""
        n_features = len(self.mappings_)
        X_out = np.zeros((X.shape[0], n_features))
        if hasattr(X, "columns"):
            df = pd.DataFrame(X, columns=list(X.columns))
        else:
            df = pd.DataFrame(X)

        mapped_features = list(self.mappings_.keys())
        for j, feat in enumerate(mapped_features):
            col_vals = df[feat].fillna(df[feat].median()).values

            if feat in self.bin_edges_:
                # Extraire les bords de bins tries et les valeurs WoE associees
                edges_list = []
                woe_list = []
                for (left, right), woe_val in self.bin_edges_[feat].items():
                    edges_list.append(left)
                    woe_list.append(woe_val)

                # Construire les bords: left de chaque bin, plus la borne sup du dernier
                edges = sorted(edges_list)
                max_right = max(
                    (right for (left, right) in self.bin_edges_[feat].keys()
                     if isinstance(right, (int, float))),
                    default=edges[-1] + 1
                )
                edges.append(max_right)

                # np.digitize assigne chaque valeur a un index de bin
                indices = np.digitize(col_vals, edges, right=True)
                indices = np.clip(indices, 0, len(woe_list) - 1)

                woe_arr = np.array(woe_list)
                X_out[:, j] = woe_arr[indices]
            else:
                # Fallback: mapping simple par cle string
                mapping = self.mappings_[feat]
                for k, val in enumerate(col_vals):
                    key = str(val)
                    X_out[k, j] = mapping.get(key, 0.0)

        return X_out

    def set_output(self, *, transform=None):
        return self

    def get_feature_names_out(self, input_features=None):
        return np.array(list(self.mappings_.keys()))
