"""
agregation_adherents.py
===================================================
Fonctions d'agrégation et de transformation pour passer d'une granularité
niveau contact (1 ligne par individu) à une granularité organisation / adhérent
(1 ligne par groupe / 'GROUPE - ID').

Principe d'architecture :
Chaque fonction extrait une dimension du contact et la résume au niveau organisation
soit sous la forme d'un petit DataFrame intermédiaire (clé: group_col), soit en
l'injectant directement dans la table de destination (df_marque).

Dimensions agrégées :
- Données catégorielles : mode statistique majoritaire, gestion des conflits et statuts membres
- Profil d'équipe       : flags hiérarchiques (C-level, VIP...), couverture des départements (logique 3 états)
- Volumétrie & effectifs: nombre total de contacts, contacts actifs vs anciens
- Récence & ancienneté  : ancienneté de la relation, écarts à la date cible (min, max, jours)
- Engagement & canaux   : taux de réception/joignabilité, consentements de communication (opt-in)
- Métriques continues   : moyennes, sommes et minima d'activité (clics, ouvertures, anticipation)

Installation :
    pip install pandas numpy

Usage :
    import pandas as pd
    from agregation_adherents import aggregate_columns_by_mode, count_members_by_status

    df_contacts = pd.read_csv("contacts.csv")
    df_org = count_members_by_status(df_contacts, group_col="GROUPE - ID", status_col="Status", org_status_col="Org Status")
"""

from __future__ import annotations

import pandas as pd
from datetime import datetime
from src.agregation_membres import check_columns, ajout_date_cible


GROUP_COL = "GROUPE - ID"
# ---------------------------------------------------------------------------
# 1. Agrégation générique par mode (valeur la plus fréquente)
# ---------------------------------------------------------------------------

def first_mode(s: pd.Series):
    """Retourne le mode d'une série (le premier en cas d'égalité), ou pd.NA
    si la série ne contient aucune valeur non nulle."""
    m = s.mode(dropna=True)
    return m.iloc[0] if not m.empty else pd.NA


def detect_conflicts(df: pd.DataFrame, group_col: str, cols: list[str]) -> tuple[pd.DataFrame, dict]:
    """Pour chaque colonne de `cols`, compte le nombre de groupes ayant plus
    d'une valeur non nulle distincte (= un "conflit" à arbitrer lors d'une
    agrégation par mode).

    Retourne (nunique_par_groupe, résumé_des_conflits).
    """
    cols = [c for c in cols if c in df.columns]
    nunique = df[[group_col] + cols].groupby(group_col).agg(lambda s: s.dropna().nunique())
    conflict_summary = {c: int((nunique[c] > 1).sum()) for c in cols}
    return nunique, conflict_summary


def aggregate_columns_by_mode(df: pd.DataFrame, group_col: str, cols: list[str], verbose: bool = True) -> pd.DataFrame:
    """Agrège `cols` au niveau `group_col` en prenant le mode de chaque
    colonne. Affiche un résumé des conflits détectés si verbose=True.

    Retourne un DataFrame avec une ligne par `group_col`.
    """
    missing_cols = [col for col in cols if col not in df.columns]
    if missing_cols:
        raise KeyError(f"Colonnes manquantes : {missing_cols}")
    
    keep_cols = [c for c in cols if c in df.columns]
    if group_col not in keep_cols:
        keep_cols = [group_col] + keep_cols

    df_info = df[keep_cols].copy()
    _, conflict_summary = detect_conflicts(df, group_col, [c for c in keep_cols if c != group_col])

    if verbose:
        print("Colonnes agrégées :", keep_cols)
        print(f"Relations uniques : {df[group_col].nunique():,}")
        flagged = {c: n for c, n in conflict_summary.items() if n}
        if flagged:
            print("Conflits détectés (nb de relations avec >1 valeur) :")
            for col, cnt in flagged.items():
                print(f" - {col}: {cnt}")
        else:
            print("Aucun conflit détecté.")

    return df_info.groupby(group_col, dropna=False).agg(first_mode).reset_index()


def aggregate_membership_status(df, group_col, status_col):
    """
    Agrège le statut d'adhésion par groupe.
    Si un groupe contient à la fois 'Current Member' et 'Former Member',
    on garde 'Current Member'.
    """
    def resolve_membership(group):
        unique_vals = group.dropna().unique()
        if len(unique_vals) > 1 and 'Current Member' in unique_vals:
            return 'Current Member'
        return group.mode()[0] if len(group.mode()) > 0 else group.iloc[0]
    
    result = df.groupby(group_col, dropna=False)[status_col].apply(resolve_membership).reset_index()
    result.columns = [group_col, status_col]
    return result


def resolve_value_conflicts_interactive(df_info: pd.DataFrame, df_aggregated: pd.DataFrame, group_col: str, col: str) -> list:
    """Pour une colonne en conflit (plusieurs valeurs distinctes dans un même
    groupe), propose interactivement (via `input()`) de remplacer toutes les
    valeurs du groupe par le mode déjà calculé dans `df_aggregated`.

    Outil d'exploration ponctuelle (pas destiné à un pipeline automatisé).
    Modifie `df_info` en place et retourne la liste des group_col modifiés.
    """
    modified = []
    nunique = df_info.groupby(group_col)[col].agg(lambda s: s.dropna().nunique())
    conflict_ids = nunique.index[nunique > 1]

    for relation_id in conflict_ids:
        values = df_info.loc[df_info[group_col] == relation_id, col].dropna().unique()
        mode_value = df_aggregated.loc[df_aggregated[group_col] == relation_id, col].iloc[0]

        print(f"ID {relation_id}: valeurs actuelles = {list(values)} | mode proposé = {mode_value}")
        if input("  Remplacer par le mode ? (o/n): ").strip().lower() == "o":
            df_info.loc[df_info[group_col] == relation_id, col] = mode_value
            modified.append(relation_id)
            print("  ✓ Remplacé")
        else:
            print("  ✗ Conservé")

    return modified


# ---------------------------------------------------------------------------
# 2. Indicateurs booléens "au moins un contact du groupe vérifie X"
# ---------------------------------------------------------------------------

def _any_true(s: pd.Series):
    """True si au moins une valeur True/1.0, NA si rien n'est renseigné,
    False sinon. Gère indifféremment les colonnes booléennes et 0/1."""
    if s.dropna().empty:
        return pd.NA
    return bool(s.eq(True).any() or s.eq(1.0).any())


def add_any_true_flags(df: pd.DataFrame, df_marque: pd.DataFrame, group_col: str, flags: dict[str, str]) -> pd.DataFrame:
    """Pour chaque entrée {colonne_source: nom_nouvelle_colonne} de `flags`,
    calcule "le groupe contient-il au moins un contact où colonne_source est
    vraie ?" et fusionne le résultat dans df_marque.

    Exemple :
        add_any_true_flags(df, df_marque, group_col, {
            'bool_Is F LEVEL': 'has_F_level',
            'VIP': 'has_VIP',
        })
    """
    src_cols = list(flags.keys())
    summary = (
        df[[group_col] + src_cols]
        .groupby(group_col, dropna=False)
        .agg(_any_true)
        .reset_index()
        .rename(columns=flags)
    )
    return df_marque.merge(summary, on=group_col, how="left")


# ---------------------------------------------------------------------------
# 3. Colonnes "department_*" (règle d'agrégation à 3 états)
# ---------------------------------------------------------------------------

def agg_tri_state(s: pd.Series):
    """Règle d'agrégation à 3 états pour une colonne booléenne par groupe :
    - True si au moins une valeur True dans le groupe
    - pd.NA si le groupe n'a aucune valeur renseignée
    - False si toutes les valeurs renseignées sont False
    - False si mélange de False et de NaN sans aucun True (cas ambigu,
      prudence : on ne tranche pas)
    """
    if s.eq(True).any():
        return True
    if s.isna().all():
        return pd.NA
    if s.eq(False).all():
        return False
    if s.eq(False).any() and s.isna().any() and not s.eq(True).any():
        return False
    return pd.NA


def aggregate_department_flags(df: pd.DataFrame, group_col: str, prefix: str = "department_") -> tuple[pd.DataFrame, dict]:
    """Agrège toutes les colonnes commençant par `prefix` au niveau groupe
    avec `agg_tri_state`, et détecte les groupes ambigus (mélange de False
    et de NaN sans aucun True) pour chaque colonne.

    Retourne (df_dept_agrégé, ambiguous) où ambiguous est un
    dict {colonne: [group_ids ambigus]}.
    """
    dept_cols = [c for c in df.columns if c.startswith(prefix)]
    df_dept = (
        df[[group_col] + dept_cols]
        .groupby(group_col, dropna=False)
        .agg(agg_tri_state)
        .reset_index()
    )

    ambiguous = {}
    for col in dept_cols:
        mask = df.groupby(group_col)[col].agg(
            lambda s: bool(s.eq(False).any() and s.isna().any() and not s.eq(True).any())
        )
        ids = mask[mask].index.tolist()
        if ids:
            ambiguous[col] = ids

    return df_dept, ambiguous


# ---------------------------------------------------------------------------
# 4. Membres actifs (statut, ancienneté, speakers)
# ---------------------------------------------------------------------------
def count_members(
    df: pd.DataFrame,
    group_col: str,
    id_col: str = "System ID",
    new_name: str = "nb_contacts",
) -> pd.DataFrame:
    """Nombre de contacts uniques (`id_col`) par groupe."""
    return (
        df[id_col]
        .groupby(df[group_col])
        .nunique()
        .reset_index(name=new_name)
    )


def count_members_by_status(
    df: pd.DataFrame,
    group_col: str,
    status_col: str,
    org_status_col: str,
    id_col: str = "System ID",
    new_name: str = "nb_current_contacts",
) -> pd.DataFrame:
    """Compte les contacts uniques par organisation.
    - Si l'organisation est "Current Member" :
      compte uniquement les contacts dont le statut contact est "Current Member".
    - Si l'organisation est "Former Member" :
      compte tous les contacts uniques sans filtre sur le statut contact.
    """
    for col in [group_col, status_col, org_status_col, id_col]:
        if col not in df.columns:
            raise KeyError(f"Colonne '{col}' introuvable dans le dataframe")

    df_result = df.copy()

    # On applique la règle selon le statut de l'organisation
    df_result = df_result[
        (
            (df_result[org_status_col] == "Former Member")
        )
        |
        (
            (df_result[org_status_col] == "Current Member")
            & (df_result[status_col] == "Current Member")
        )
    ]

    return (
        df_result
        .groupby(group_col)[id_col]
        .nunique()
        .reset_index(name=new_name)
    )

def contact_date_range(
    df: pd.DataFrame,
    group_col: str,
    date_col: str,
) -> pd.DataFrame:
    """Pour chaque groupe, retourne la date de création du contact
    le plus ancien et le plus récent, sans filtre sur le statut du contact.
    """
    check_columns(df, [group_col, date_col])

    if "date_cible" not in df.columns:
            df = ajout_date_cible(df)
    
    sub = df[[group_col, date_col, "date_cible"]].copy()

    sub[date_col] = pd.to_datetime(sub[date_col], errors="coerce")

    # On ignore les dates postérieures à la date cible
    sub[date_col] = sub[date_col].where(sub[date_col] <= sub["date_cible"])

    result = (
        sub.groupby(group_col)
        .agg(
            min=(date_col, "min"),
            max=(date_col, "max"),
            date_cible=("date_cible", "first"),
        )
        .reset_index()
    )

    # Récupérer l'année
    result["oldest_contact_year"] = result["min"].dt.year
    result["youngest_contact_year"] = result["max"].dt.year

    # Calculer le nombre de jours depuis ces dates
    result["days_since_oldest_contact"] = (
        result["date_cible"] - result["min"]
    ).dt.days
    result["days_since_youngest_contact"] = (
        result["date_cible"] - result["max"]
    ).dt.days

    result = result.rename(
        columns={
            "min": "creation_date_oldest_contact",
            "max": "creation_date_youngest_contact",
        }
    )

    return result


def active_contact_date_range(
    df: pd.DataFrame,
    group_col: str,
    status_col: str,
    org_status_col: str,
    date_col: str,
) -> pd.DataFrame:
    """Pour chaque organisation :
    - Organisation active :
      date du contact actif le plus ancien et le plus récent.
    - Organisation ancienne :
      date du contact le plus ancien et le plus récent sans filtre de statut.
    """

    for col in [group_col, status_col, org_status_col, date_col]:
        if col not in df.columns:
            raise KeyError(f"Colonne '{col}' introuvable dans le dataframe")

    sub = df.copy()

    # Règle métier selon le statut de l'organisation
    sub = sub[
        (sub[org_status_col] == "Former Member")
        |
        (
            (sub[org_status_col] == "Current Member")
            & (sub[status_col] == "Current Member")
        )
    ]

    sub[date_col] = pd.to_datetime(sub[date_col], errors="coerce")

    result = (
        sub.groupby(group_col)[date_col]
        .agg(["min", "max"])
        .reset_index()
    )

    # Récupérer l'année
    result["oldest_current_member_year"] = result["min"].dt.year
    result["youngest_current_member_year"] = result["max"].dt.year

    # Calculer le nombre de jours depuis ces dates
    reference_date = pd.Series(datetime(2026, 1, 12), index=df.index)
    result["days_since_oldest_current_member"] = (
        reference_date - result["min"]
    ).dt.days
    result["days_since_youngest_current_member"] = (
        reference_date - result["max"]
    ).dt.days

    result = result.rename(
        columns={
            "min": "creation_date_oldest_current_member",
            "max": "creation_date_youngest_current_member",
        }
    )

    return result


# ---------------------------------------------------------------------------
# 5. Taux par modalité d'une variable catégorielle (ex : Recipient Status)
# ---------------------------------------------------------------------------
# Pour le total, plutôt prendre la valeur
def categorical_rate_recipient_status(
    df: pd.DataFrame,
    group_col: str,
    col: str,
    value: str = "Active, relation receives e-mail",
    prefix: str | None = None,
) -> pd.DataFrame:
    """Pour chaque groupe, calcule le nombre et le taux (%) de contacts
    correspondant à une modalité donnée de `col`.
    """

    if col not in df.columns:
        raise KeyError(f"Colonne '{col}' introuvable dans le dataframe")

    # Nombre total de contacts par groupe (dénominateur)
    totals = df.groupby(group_col).size()

    # Contacts correspondant uniquement à la modalité recherchée
    counts = (
        df[df[col] == value]
        .groupby(group_col)
        .size()
    )

    result = (
        counts
        .reindex(totals.index, fill_value=0)
        .to_frame("nb")
    )

    result["pct"] = (
        result["nb"]
        .div(totals)
        .mul(100)
        .round(2)
    )

    tag = f"{prefix}_" if prefix else ""

    result = result.rename(
        columns={
            "nb": f"{tag}nb_active_relation_receives_email",
            "pct": f"{tag}pct_active_relation_receives_email",
        }
    )

    return result.reset_index()

def categorical_rates_by_group(
    df: pd.DataFrame,
    group_col: str,
    col: str,
    prefix: str | None = None,
    include_counts: bool = True,
) -> pd.DataFrame:
    """Pour chaque modalité de `col`, calcule le nombre et le taux (%) de
    contacts par groupe. `prefix` est ajouté devant chaque nom de colonne
    pour éviter les collisions de noms lors de la fusion dans df_marque.
    """
    if col not in df.columns:
        raise KeyError(f"Colonne '{col}' introuvable dans le dataframe")

    totals = df.groupby(group_col).size()
    counts = df.groupby([group_col, col]).size().unstack(fill_value=0)
    rates_pct = (counts.div(totals, axis=0).fillna(0) * 100).round(2)

    tag = f"{prefix}_" if prefix else ""
    counts = counts.add_prefix(f"{tag}nb_").reset_index()
    rates_pct = rates_pct.add_prefix(f"{tag}pct_").reset_index()

    if include_counts:
        return counts.merge(rates_pct, on=group_col)
    return rates_pct


# ---------------------------------------------------------------------------
# 6. Acceptation des communications (colonnes "Communication - *")
# ---------------------------------------------------------------------------

ACCEPT_STRINGS = {"oui", "yes", "true", "1", "1.0", "y", "o"}


def build_acceptance_mask(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Masque booléen "a accepté" pour des colonnes pouvant contenir des
    valeurs numériques (1 / 1.0), booléennes (True) ou textuelles
    ('oui', 'yes', ...)."""
    mask_num = df[cols].eq(1.0) | df[cols].eq(True)
    mask_str = df[cols].astype(str).apply(lambda s: s.str.lower().isin(ACCEPT_STRINGS))
    return mask_num | mask_str


def acceptance_counts_and_rates_by_group(
    df: pd.DataFrame,
    group_col: str,
    cols: list[str],
    prefix: str = "comm_accept",
) -> pd.DataFrame:
    """Pour chaque colonne de `cols` (ex : 'Communication - Email'), calcule
    le nombre et le taux (%) de contacts du groupe ayant accepté."""
    if not cols:
        raise KeyError("Aucune colonne fournie pour le calcul d'acceptation")

    accept_mask = build_acceptance_mask(df, cols)
    base = pd.concat([df[[group_col]], accept_mask], axis=1)

    counts = base.groupby(group_col)[cols].sum().add_prefix(f"{prefix}_nb_")
    rates_pct = (base.groupby(group_col)[cols].mean() * 100).round(2).add_prefix(f"{prefix}_pct_")

    return counts.reset_index().merge(rates_pct.reset_index(), on=group_col)


# ---------------------------------------------------------------------------
# 7. Indicateurs numériques génériques (moyenne / minimum par groupe)
# ---------------------------------------------------------------------------

def numeric_agg_by_group(
    df: pd.DataFrame,
    group_col: str,
    col: str,
    new_name: str,
    how: str = "mean",
) -> pd.DataFrame:
    """Agrège une colonne numérique par groupe (moyenne ou minimum), avec
    conversion robuste en numérique (errors='coerce').

    how: 'mean' (valeur moyenne par contact) ou 'min' (utilisé pour les
    indicateurs de récence : le minimum de "jours depuis X" = l'évènement
    le plus récent).
    """
    if col not in df.columns:
        raise KeyError(f"Colonne '{col}' introuvable dans le dataframe")
    if how not in {"mean", "min", "sum"}:
        raise ValueError("how doit valoir 'mean', 'min' ou 'sum'")

    series = pd.to_numeric(df[col], errors="coerce")
    grouped = series.groupby(df[group_col])
    result = (grouped.mean() if how == "mean" else grouped.min() if how == "min" else grouped.sum()).reset_index(name=new_name)
    result[new_name] = result[new_name].round(2)
    return result



def add_numeric_aggregations(
    df: pd.DataFrame,
    df_marque: pd.DataFrame,
    group_col: str,
    specs: list[tuple],
    how: str = "mean",
) -> pd.DataFrame:
    """Calcule et fusionne dans df_marque une série d'agrégats numériques.

    `specs` est une liste de tuples :
        (colonne_source, nom_colonne_résultat)

    Exemple :
        add_numeric_aggregations(df, df_marque, group_col, [
            ('clicks_count_by_relation', 'avg_clicks_per_contact'),
            ('taux_ouverture', 'avg_taux_ouverture'),
        ], how='mean')
    """
    for spec in specs:
        col, new_name = spec[0], spec[1]
        result = numeric_agg_by_group(df, group_col, col, new_name, how=how)
        df_marque = df_marque.merge(result, on=group_col, how="left")
    return df_marque


# ---------------------------------------------------------------------------
# 8. Récence générique (ex : dernière connexion)
# ---------------------------------------------------------------------------

def recency_by_group(
    df: pd.DataFrame,
    group_col: str,
    date_col: str,
    new_name: str = "days_since_last_online",
    reference_date: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Nombre de jours depuis l'évènement le plus récent de `date_col`
    (déjà au format datetime) par groupe.
    """
    reference_date = reference_date or pd.Timestamp.now()
    days_since = (reference_date - df[date_col]).dt.days
    return days_since.groupby(df[group_col]).min().reset_index(name=new_name)


# ---------------------------------------------------------------------------
# 9. Désabonnement
# ---------------------------------------------------------------------------

def unsubscribe_rate_by_group(df: pd.DataFrame, group_col: str, unsub_col: str) -> pd.DataFrame:
    """Nombre et taux (%) de contacts désabonnés par groupe."""
    if unsub_col not in df.columns:
        raise KeyError(f"Colonne '{unsub_col}' introuvable dans le dataframe")

    is_unsub = df[unsub_col].eq(1.0) | df[unsub_col].eq(True)
    counts = is_unsub.groupby(df[group_col]).sum().reset_index(name="nb_unsubscribed_by_relation")
    totals = df.groupby(group_col).size().reset_index(name="total_contacts")

    result = counts.merge(totals, on=group_col, how="right").fillna({"nb_unsubscribed_by_relation": 0})
    result["unsub_rate_pct"] = (result["nb_unsubscribed_by_relation"] / result["total_contacts"] * 100).round(2)
    return result


####### autre #######
def average_anticipation_days_by_group(
    df: pd.DataFrame,
    group_col: str,
    source_col: str = "average_anticipation_days_by_relation",
    new_name: str = "average_anticipation_days_by_group",
) -> pd.DataFrame:
    """Agrège la variable `average_anticipation_days_by_relation` au niveau groupe."""
    return numeric_agg_by_group(
        df,
        group_col,
        source_col,
        new_name,
        how="mean",
    )

