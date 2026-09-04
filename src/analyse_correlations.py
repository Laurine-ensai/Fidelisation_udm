"""
Fonctions d'analyse des associations entre variables.

Méthodes disponibles :
- Pearson
- Spearman
- V de Cramér
- Epsilon² (Kruskal-Wallis)
- Eta (correlation ratio)

Les fonctions calculent d'abord les matrices, puis peuvent afficher les
résultats sous forme de heatmaps par blocs.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.stats import chi2_contingency
import pingouin as pg
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# Sélection des variables
# ============================================================================

def variables_quantitatives(df: pd.DataFrame, min_unique: int = 10) -> list[str]:
    """Retourne les variables numériques ayant plus de `min_unique` modalités."""
    return [
        col
        for col in df.columns
        if pd.api.types.is_numeric_dtype(df[col])
        and df[col].nunique(dropna=True) > min_unique
    ]


def variables_qualitatives(
    df: pd.DataFrame,
    max_unique: int = 10,
) -> list[str]:
    """
    Retourne les variables qualitatives.

    Une variable est considérée comme qualitative si elle est de type object,
    category ou bool, ou si elle possède au plus `max_unique` modalités.
    """
    return [
        col
        for col in df.columns
        if (
            pd.api.types.is_object_dtype(df[col])
            or pd.api.types.is_categorical_dtype(df[col])
            or pd.api.types.is_bool_dtype(df[col])
            or df[col].nunique(dropna=True) <= max_unique
        )
    ]

# ============================================================================
# Affichage
# ============================================================================

def afficher_heatmaps(
    matrice: pd.DataFrame,
    batch_size: int = 10,
    titre: str = "Matrice",
    vmin: float | None = None,
    vmax: float | None = None,
    center: float | None = None,
    fmt: str = ".2f",
    cmap: str = "coolwarm",
) -> None:
    """Affiche une matrice sous forme de heatmaps par blocs."""
    if matrice is None or matrice.empty:
        return

    variables = list(matrice.index)

    for i in range(0, len(variables), batch_size):
        for j in range(i, len(variables), batch_size):
            cols_i = variables[i:i + batch_size]
            cols_j = variables[j:j + batch_size]

            sous_matrice = matrice.loc[cols_i, cols_j]

            plt.figure(figsize=(10, 8))
            sns.heatmap(
                sous_matrice,
                annot=True,
                fmt=fmt,
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                center=center,
                linewidths=0.5,
                cbar_kws={"shrink": 0.8},
            )

            plt.title(
                f"{titre} : variables "
                f"{i}-{i + len(cols_i) - 1} vs "
                f"{j}-{j + len(cols_j) - 1}"
            )
            plt.xticks(rotation=45, ha="right")
            plt.yticks(rotation=0)
            plt.tight_layout()
            plt.show()


# ============================================================================
# Pearson / Spearman
# ============================================================================

def coef_pearson(
    df: pd.DataFrame,
    batch_size: int = 10,
    afficher: bool = True,
) -> pd.DataFrame:
    """Calcule la matrice de corrélation de Pearson."""
    cols = variables_quantitatives(df)

    if not cols:
        raise ValueError("Aucune variable quantitative détectée.")

    matrice = df[cols].corr(method="pearson")

    if afficher:
        afficher_heatmaps(
            matrice,
            batch_size=batch_size,
            titre="Pearson",
            vmin=-1,
            vmax=1,
            center=0,
        )

    return matrice


def coef_spearman(
    df: pd.DataFrame,
    batch_size: int = 10,
    afficher: bool = True,
) -> pd.DataFrame:
    """Calcule la matrice de corrélation de Spearman."""
    cols = variables_quantitatives(df)

    if not cols:
        raise ValueError("Aucune variable quantitative détectée.")

    matrice = df[cols].corr(method="spearman")

    if afficher:
        afficher_heatmaps(
            matrice,
            batch_size=batch_size,
            titre="Spearman",
            vmin=-1,
            vmax=1,
            center=0,
        )

    return matrice

# ============================================================================
# V de Cramér
# ============================================================================

def v_cramer(var1: pd.Series, var2: pd.Series) -> float:
    """Calcule le V de Cramér entre deux variables qualitatives."""
    temp = pd.DataFrame({"var1": var1, "var2": var2}).dropna()

    if temp["var1"].nunique() <= 1 or temp["var2"].nunique() <= 1:
        return 0.0

    table = pd.crosstab(temp["var1"], temp["var2"])

    if min(table.shape) <= 1:
        return 0.0

    chi2, _, _, _ = chi2_contingency(table)
    n = table.to_numpy().sum()

    if n == 0:
        return 0.0

    return float(np.sqrt(chi2 / (n * (min(table.shape) - 1))))

def coef_cramer(
    df: pd.DataFrame,
    batch_size: int = 10,
    afficher: bool = True,
) -> pd.DataFrame:
    """Calcule la matrice de V de Cramér."""
    cols = variables_qualitatives(df)

    if not cols:
        raise ValueError("Aucune variable qualitative détectée.")

    matrice = pd.DataFrame(
        np.eye(len(cols)),
        index=cols,
        columns=cols,
    )

    for i, col1 in enumerate(cols):
        for col2 in cols[i + 1:]:
            valeur = v_cramer(df[col1], df[col2])
            matrice.loc[col1, col2] = valeur
            matrice.loc[col2, col1] = valeur

    if afficher:
        afficher_heatmaps(
            matrice,
            batch_size=batch_size,
            titre="V de Cramér",
            vmin=0,
            vmax=1,
            fmt=".2f",
        )

    return matrice

# ============================================================================
# Epsilon²
# ============================================================================

def epsilon_carre_pair(
    categorie: pd.Series,
    quantitatif: pd.Series,
) -> float:
    """Calcule epsilon² entre une variable qualitative et une quantitative."""
    temp = pd.DataFrame({
        "categorie": categorie,
        "quantitatif": quantitatif,
    }).dropna()

    if len(temp) <= 5 or temp["categorie"].nunique() <= 1:
        return 0.0

    try:
        kw = pg.kruskal(
            dv="quantitatif",
            between="categorie",
            data=temp,
        )

        h = kw["H"].iloc[0]
        n = len(temp)

        epsilon_sq = h / ((n**2 - 1) / (n + 1))

        return float(epsilon_sq)

    except Exception:
        return 0.0


def epsilon_carre(
    df: pd.DataFrame,
    batch_size: int = 10,
    afficher: bool = True,
) -> pd.DataFrame:
    """Calcule la matrice epsilon² : variables qualitatives × quantitatives."""
    num_cols = variables_quantitatives(df)
    cat_cols = variables_qualitatives(df)

    if not cat_cols or not num_cols:
        raise ValueError(
            "Il faut au moins une variable qualitative et une variable quantitative."
        )

    matrice = pd.DataFrame(
        index=cat_cols,
        columns=num_cols,
        dtype=float,
    )

    for cat in cat_cols:
        for num in num_cols:
            matrice.loc[cat, num] = epsilon_carre_pair(
                df[cat],
                df[num],
            )

    if afficher:
        for i in range(0, len(cat_cols), batch_size):
            for j in range(0, len(num_cols), batch_size):
                sous_matrice = matrice.loc[
                    cat_cols[i:i + batch_size],
                    num_cols[j:j + batch_size],
                ]

                plt.figure(figsize=(10, 8))
                sns.heatmap(
                    sous_matrice,
                    annot=True,
                    fmt=".3f",
                    cmap="YlOrRd",
                    vmin=0,
                    vmax=0.5,
                    linewidths=0.5,
                    cbar_kws={"shrink": 0.8},
                )

                plt.title(
                    f"Epsilon² : Quali "
                    f"{i}-{i + len(sous_matrice.index) - 1} vs "
                    f"Quanti "
                    f"{j}-{j + len(sous_matrice.columns) - 1}"
                )
                plt.xlabel("Variables numériques")
                plt.ylabel("Variables catégorielles")
                plt.xticks(rotation=45, ha="right")
                plt.yticks(rotation=0)
                plt.tight_layout()
                plt.show()

    return matrice



def correlation_ratio(categories, values):
    """
    Calcule le Correlation Ratio η entre une variable qualitative
    et une variable quantitative.
    """

    temp_df = pd.DataFrame({
        "cat": categories,
        "num": values
    }).dropna()

    if temp_df["cat"].nunique() <= 1:
        return 0

    # Moyenne globale
    y_mean = temp_df["num"].mean()

    # Variance expliquée par les groupes
    numerator = (
        temp_df
        .groupby("cat")["num"]
        .agg(["count", "mean"])
        .apply(lambda x: x["count"] * (x["mean"] - y_mean)**2, axis=1)
        .sum()
    )

    # Variance totale
    denominator = (
        (temp_df["num"] - y_mean)**2
    ).sum()

    if denominator == 0:
        return 0

    eta_squared = numerator / denominator

    return np.sqrt(eta_squared)

# ============================================================================
# Eta
# ============================================================================

def correlation_ratio(
    categories: pd.Series,
    values: pd.Series,
) -> float:
    """Calcule eta entre une variable qualitative et une quantitative."""
    temp = pd.DataFrame({
        "categorie": categories,
        "valeur": values,
    }).dropna()

    if len(temp) <= 5 or temp["categorie"].nunique() <= 1:
        return 0.0

    moyenne_globale = temp["valeur"].mean()

    moyenne_groupes = temp.groupby("categorie")["valeur"].agg(["count", "mean"])

    variance_expliquee = (
        moyenne_groupes["count"]
        * (moyenne_groupes["mean"] - moyenne_globale) ** 2
    ).sum()

    variance_totale = (
        (temp["valeur"] - moyenne_globale) ** 2
    ).sum()

    if variance_totale == 0:
        return 0.0

    return float(np.sqrt(variance_expliquee / variance_totale))


def eta_carre(
    df: pd.DataFrame,
    batch_size: int = 10,
    afficher: bool = True,
) -> pd.DataFrame:
    """Calcule eta : variables qualitatives × quantitatives."""
    num_cols = variables_quantitatives(df)
    cat_cols = variables_qualitatives(df)

    if not cat_cols or not num_cols:
        raise ValueError(
            "Il faut au moins une variable qualitative et une variable quantitative."
        )

    matrice = pd.DataFrame(
        index=cat_cols,
        columns=num_cols,
        dtype=float,
    )

    for cat in cat_cols:
        for num in num_cols:
            matrice.loc[cat, num] = correlation_ratio(
                df[cat],
                df[num],
            )

    if afficher:
        for i in range(0, len(cat_cols), batch_size):
            for j in range(0, len(num_cols), batch_size):
                sous_matrice = matrice.loc[
                    cat_cols[i:i + batch_size],
                    num_cols[j:j + batch_size],
                ]

                plt.figure(figsize=(10, 8))
                sns.heatmap(
                    sous_matrice,
                    annot=True,
                    fmt=".3f",
                    cmap="YlOrRd",
                    vmin=0,
                    vmax=1,
                    linewidths=0.5,
                    cbar_kws={"shrink": 0.8},
                )

                plt.title(
                    f"Eta : Quali "
                    f"{i}-{i + len(sous_matrice.index) - 1} vs "
                    f"Quanti "
                    f"{j}-{j + len(sous_matrice.columns) - 1}"
                )
                plt.xlabel("Variables numériques")
                plt.ylabel("Variables catégorielles")
                plt.xticks(rotation=45, ha="right")
                plt.yticks(rotation=0)
                plt.tight_layout()
                plt.show()

    return matrice


# ============================================================================
# Filtrage
# ============================================================================

def filtrer_coefficients(
    matrice: pd.DataFrame,
    seuil: float = 0.8,
    est_symetrique: bool = True,
) -> pd.DataFrame:
    """
    Retourne les paires de variables dépassant le seuil.

    Pour une matrice symétrique (Pearson, Spearman, Cramér), les doublons
    et l'auto-corrélation sont supprimés.
    """
    if matrice is None or matrice.empty:
        return pd.DataFrame(
            columns=["Variable_1", "Variable_2", "Coefficient"]
        )

    df_pairs = (
        matrice
        .stack()
        .reset_index()
    )
    df_pairs.columns = [
        "Variable_1",
        "Variable_2",
        "Coefficient",
    ]

    if est_symetrique:
        # Position dans la matrice plutôt que comparaison alphabétique
        # des noms de variables.
        lignes = {col: i for i, col in enumerate(matrice.index)}

        df_pairs["_i"] = df_pairs["Variable_1"].map(lignes)
        df_pairs["_j"] = df_pairs["Variable_2"].map(lignes)

        df_pairs = df_pairs[df_pairs["_i"] < df_pairs["_j"]]

        df_pairs = df_pairs.drop(columns=["_i", "_j"])

    else:
        df_pairs = df_pairs[
            df_pairs["Variable_1"] != df_pairs["Variable_2"]
        ]

    df_filtre = df_pairs[
        df_pairs["Coefficient"].abs() >= seuil
    ].copy()

    df_filtre["Force"] = df_filtre["Coefficient"].abs()

    return (
        df_filtre
        .sort_values("Force", ascending=False)
        .drop(columns="Force")
        .reset_index(drop=True)
    )


# ============================================================================
# Analyse complète
# ============================================================================

def analyser_correlations(
    df: pd.DataFrame,
    seuil_spearman: float = 0.8,
    seuil_cramer: float = 0.8,
    seuil_eta: float = 0.5,
    batch_size: int = 10,
    afficher: bool = True,
) -> dict[str, pd.DataFrame]:
    """
    Calcule les trois matrices d'association et leurs paires fortes.

    Retourne :
        {
            "spearman": matrice,
            "cramer": matrice,
            "eta": matrice,
            "spearman_fortes": paires,
            "cramer_fortes": paires,
            "eta_forts": paires,
        }
    """
    spearman = coef_spearman(
        df,
        batch_size=batch_size,
        afficher=afficher,
    )

    cramer = coef_cramer(
        df,
        batch_size=batch_size,
        afficher=afficher,
    )

    eta = eta_carre(
        df,
        batch_size=batch_size,
        afficher=afficher,
    )

    return {
        "spearman": spearman,
        "cramer": cramer,
        "eta": eta,
        "spearman_fortes": filtrer_coefficients(
            spearman,
            seuil=seuil_spearman,
            est_symetrique=True,
        ),
        "cramer_fortes": filtrer_coefficients(
            cramer,
            seuil=seuil_cramer,
            est_symetrique=True,
        ),
        "eta_forts": filtrer_coefficients(
            eta,
            seuil=seuil_eta,
            est_symetrique=False,
        ),
    }
