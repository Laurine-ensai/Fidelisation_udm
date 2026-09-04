"""
agregation_membres.py
======================
Ensemble de fonctions de feature engineering et d'agrégation au niveau membre / relation.
Permet d'enrichir les données relationnelles (présence aux réunions, réactivité
aux invitations, délais d'interaction, tags) avant modélisation ou reporting.

Catégories de fonctions :
- Dates & récence      : calcul des jours écoulés par rapport à une date cible ou de démission
- Comptages & volumes  : décompte de lignes, de valeurs ciblées et de champs non vides
- Ratios & taux        : calcul de pourcentages d'engagement et de présence
- Statuts & indicateurs: création de flags binaires sur conditions métier
- Délais d'interaction : calculs des durées (jours, heures, minutes) entre événements

Installation :
    pip install pandas numpy

Usage :
    import pandas as pd
    from agregation_membres import days_since_last_event, add_rate

    df = pd.read_csv("donnees_membres.csv")
    df = days_since_last_event(df, id_col="Attendee Relation ID", date_col="Meeting Start date", new_col="recence_reunion")
"""



import pandas as pd
import numpy as np
from datetime import datetime

DATE_REFERENCE = pd.Timestamp("2026-01-12")

def check_columns(
    df: pd.DataFrame,
    columns:list[str]
) -> None :
    """
    Vérifie que les colonnes nécessaires sont présentes dans le DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame à vérifier.
    columns : list[str]
        Liste des colonnes obligatoires.

    Raises
    ------
    KeyError
        Si une ou plusieurs colonnes sont absentes.
    """
    missing_cols = [col for col in columns if col not in df.columns]

    if missing_cols:
        raise KeyError(f"Colonnes manquantes : {missing_cols}")
    

###### fonction avec date ######
def ajout_date_cible(
    df: pd.DataFrame, 
    annee_demission: str = "GROUPE - Année démission *", 
    nom_colonne: str = "date_cible",
) -> pd.DataFrame:
    """
    Ajoute une colonne avec la date de référence au dataframe :
    - Si l'année de démission est comprise entre 2019 et 2025 : 1er janvier de cette année
    - Sinon : DATE_REFERENCE
    """
    check_columns(df,[annee_demission])

    df = df.copy()
    
    # Valeur par défaut pour toutes les lignes
    df[nom_colonne] = DATE_REFERENCE

    # Conversion robuste des années
    years = pd.to_numeric(df[annee_demission], errors="coerce")

    # Condition : démission entre 2019 et 2025 inclus
    mask = years.between(2019, 2025)  # between() est inclusif des deux bornes
    
    # Remplacement pour les années valides
    df.loc[mask, nom_colonne] = pd.to_datetime(
        years.loc[mask].astype(int).astype(str) + "-01-01"
    )
    
    return df


def days_since_last_event(
    df: pd.DataFrame,
    id_col: str,
    date_col: str,
    new_col: str,
    day_first = True,
    valid_col: str | None = None
) -> pd.DataFrame:
    """Ajoute une colonne avec le nombre de jours écoulés depuis la dernière
    participation (basée sur Meeting Start date) pour chaque Attendee Relation ID.

    Pour chaque Attendee Relation ID, on calcule la différence en jours entre la date actuelle et la Meeting Start la plus récente. Si aucune participation précédente n'existe, la valeur est
    NaN.

    Parameters
    ----------
    id_col : str
        Colonne identifiant les individus (Relation ID, Attendee Relation ID...).

    date_col : str
        Colonne contenant la date de l'évènement
        (réunion, inscription, clic, ouverture...).

    new_col : str
        Nom de la colonne créée.
    """
    check_columns(df,[id_col,date_col])

    if valid_col is not None:
        check_columns(df, [valid_col])

    # Ajout de la date cible si elle n'existe pas déjà
    if "date_cible" not in df.columns:
        df = ajout_date_cible(df)

    # Conversion des dates de réunion
    event_dates = pd.to_datetime(
        df[date_col],
        dayfirst=day_first,      # pour gérer les formats jour/mois/année
        errors="coerce",    # les valeurs non convertibles deviennent NaT (équivalent datetime de NaN)
    )

    # Dernière participation par relation
    valid_event_dates = event_dates.where(event_dates <= df["date_cible"])
    # Si une colonne de validité est fournie, on filtre également dessus
    if valid_col is not None:
        valid_event_dates = valid_event_dates.where(df[valid_col])

    last_event_date = valid_event_dates.groupby(df[id_col]).transform("max")
    
    # Nombre de jours entre la date cible et la dernière participation
    df[new_col] = (df["date_cible"] - last_event_date).dt.days

    return df


###### fonction qui compte ######
def add_number_of_lines(
    df: pd.DataFrame,
    id_col: str,
    new_col: str,
) -> pd.DataFrame:
    check_columns(df,[id_col])
    
    df[new_col] = df.groupby(df[id_col])[id_col].transform('count')
    return df


def add_count_of_a_value(
    df: pd.DataFrame,
    id_col: str,
    valeur_col: str,
    valeur,
    new_col: str,
) -> pd.DataFrame:
    check_columns(df,[id_col, valeur_col])
    
    if isinstance(valeur, (list, tuple, set)):
        mask = df[valeur_col].isin(valeur)
    else:
        mask = df[valeur_col] == valeur

    counts = (
        df.loc[mask]
        .groupby(id_col)
        .size()
    )

    df[new_col] = (
        df[id_col]
        .map(counts)
        .fillna(0)
        .astype(int)
    )

    return df


def add_count_non_empty(
    df: pd.DataFrame,
    id_col: str,
    colonne: str,
    new_col: str,
    valid_col: str | None = None,
) -> pd.DataFrame:
    check_columns(df,[id_col, colonne])
    
    mask = df[colonne].notna() & (df[colonne] != '')

    if valid_col is not None:
        check_columns(df, [valid_col])
        mask &= df[valid_col]

    counts = (
        df.loc[mask]
        .groupby(id_col)
        .size()
    )

    df[new_col] = (
        df[id_col]
        .map(counts)
        .fillna(0)
        .astype(int)
    )

    return df


###### fonction qui crée un taux ######
def add_rate_condition(
    df: pd.DataFrame,
    id_col: str,
    numerateur_mask,
    denominateur_mask,
    new_col: str,   
) -> pd.DataFrame:
    check_columns(df,[id_col])
     
    num = numerateur_mask.groupby(df[id_col]).transform("sum")
    den = denominateur_mask.groupby(df[id_col]).transform("sum")

    df[new_col] = (
        num
        .div(den.replace(0, np.nan))
        .mul(100)
        .round(2)
    )
    
    return df

def add_rate(
    df: pd.DataFrame,
    id_col: str,
    numerateur_col: str,
    denominateur_col: str,
    new_col: str,   
) -> pd.DataFrame:
    check_columns(df,[id_col, numerateur_col, denominateur_col])
    
    ratio = df[numerateur_col].div(df[denominateur_col])
    ratio = ratio.where(df[denominateur_col] != 0)

    df[new_col] = ratio.mul(100).round(2)

    return df


###### fonction qui crée un booléen ######
def add_condition_indicator(
    df: pd.DataFrame,
    id_col: str,
    condition: pd.Series,
    new_col: str,
) -> pd.DataFrame:
    check_columns(df,[id_col])
    
    df[new_col] = (
        condition
        .groupby(df[id_col])
        .transform("max")
        .fillna(0)
        .astype(int)
    )

    return df


# à revoir pour utiliser la fonction juste au dessus
def add_published_registered_flag_by_relation(
    df: pd.DataFrame,
    attendee_relation_id_col: str = 'Attendee Relation ID',
    meeting_status_col: str = 'Meeting Status',
    is_registered_col: str = 'Attendee Is registered?',
    published_value: str = 'Published',
    registered_value: str = 'Registered',
    new_col: str = 'published_and_registered_by_relation',
) -> pd.DataFrame:
    """Renvoie un DataFrame avec Attendee Relation ID et une variable binaire indiquant si,
    pour la relation, il existe au moins une ligne avec Meeting Status = Published et
    Attendee Is registered? = Registered.

    Args:
        df: DataFrame contenant les colonnes d'entrée.
        attendee_relation_id_col: Nom de la colonne pour l'identifiant de relation.
        meeting_status_col: Nom de la colonne avec le statut de la réunion.
        is_registered_col: Nom de la colonne indiquant si l'attendee est enregistré.
        published_value: Valeur indiquant que la réunion est publiée.
        registered_value: Valeur indiquant que l'attendee est enregistré.
        new_col: Nom de la nouvelle colonne créée.

    Returns:
        DataFrame avec une colonne Attendee Relation ID et la nouvelle variable binaire.
    """
    check_columns(df,[attendee_relation_id_col,meeting_status_col,is_registered_col])
    
    mask = (
        (df[meeting_status_col] == published_value) &
        (df[is_registered_col] == registered_value)
    )
    relation_flag = (
        df.loc[mask, attendee_relation_id_col]
        .drop_duplicates()
        .to_frame()
        .assign(**{new_col: 1})
    )

    all_relations = df[[attendee_relation_id_col]].drop_duplicates()
    result = all_relations.merge(relation_flag, on=attendee_relation_id_col, how='left')
    result[new_col] = result[new_col].fillna(0).astype(int)

    return result


###### Autres ######
def add_invitation_reactivity_by_relation(
    df: pd.DataFrame,
    attendee_relation_id_col: str = 'Attendee Relation ID',
    was_present_col: str = 'Attendee Was present?',
    is_invited_col: str = 'Attendee Is invited?',
    present_value: int = 1,
    invited_value: int = 1,
    new_col: str = 'invitation_reactivity_by_relation',
) -> pd.DataFrame:
    """Ajoute une colonne avec la réactivité aux invitations par Attendee Relation ID.

    La réactivité est définie comme :
      nombre de lignes avec Attendee Was present? == present_value
      divisé par nombre de lignes avec Attendee Is invited? == invited_value.

    Si le dénominateur est 0, on place NaN.
    """
    check_columns(df,[attendee_relation_id_col,was_present_col,is_invited_col])
    
    present_mask = (df[was_present_col] == present_value) & (df[is_invited_col] == invited_value)
    invited_mask = df[is_invited_col] == invited_value

    present_count = present_mask.groupby(df[attendee_relation_id_col]).transform('sum')
    invited_count = invited_mask.groupby(df[attendee_relation_id_col]).transform('sum')

    rate = present_count / invited_count.replace(0, np.nan)

    df[new_col] = rate.mul(100).round(2).astype(float)
    return df


def add_unique_meeting_tag_count_by_relation(
    df: pd.DataFrame,
    attendee_relation_id_col: str = 'Attendee Relation ID',
    meeting_tags_col: str = 'Meeting Tags',
    new_col: str = 'unique_meeting_tag_count_by_relation',
) -> pd.DataFrame:
    """Ajoute une colonne avec le nombre de thèmes différents (Meeting Tags)
    par Attendee Relation ID.

    Args:
        df: DataFrame contenant les colonnes d'entrée.
        attendee_relation_id_col: Nom de la colonne pour l'identifiant de relation.
        meeting_tags_col: Nom de la colonne avec les thèmes des meetings.
        new_col: Nom de la nouvelle colonne créée.

    Returns:
        DataFrame modifié avec la nouvelle colonne.
    """
    check_columns(df,[attendee_relation_id_col,meeting_tags_col])
    
    tags_df = df[[attendee_relation_id_col, meeting_tags_col]].copy()
    tags_df[meeting_tags_col] = tags_df[meeting_tags_col].fillna('').astype(str)       # Traiter les NaN comme des chaînes vides pour éviter les erreurs lors de l'explosion
    tags_df[meeting_tags_col] = tags_df[meeting_tags_col].str.strip()   # Supprimer les espaces avant/après pour éviter les doublons dus à des espaces
    tags_df = tags_df[tags_df[meeting_tags_col] != '']

    unique_tag_counts = (
        tags_df.drop_duplicates()
        .groupby(attendee_relation_id_col)[meeting_tags_col]
        .size()
    )

    df[new_col] = df[attendee_relation_id_col].map(unique_tag_counts).fillna(0).astype(int)     # Les relations sans tags auront une valeur de 0

    return df


def add_delay_within_threshold(
    df: pd.DataFrame,
    end_date_col: str,
    start_date_col: str,
    threshold_days: int = 30,
    new_col: str = "delay_within_threshold",
) -> pd.DataFrame:
    """Ajoute une variable booléenne indiquant si le délai entre deux dates
    est compris entre 0 et `threshold_days` jours inclus.
    """
    check_columns(df, [end_date_col, start_date_col])

    end_date = pd.to_datetime(
        df[end_date_col],
        dayfirst=True,
        errors="coerce",
    )
    start_date = pd.to_datetime(
        df[start_date_col],
        dayfirst=True,
        errors="coerce",
    )

    delay_days = (end_date - start_date).dt.days

    df[new_col] = delay_days.between(0, threshold_days)

    return df


def add_average_delay_by_relation(
    df: pd.DataFrame,
    attendee_relation_id_col: str,
    end_date_col: str,
    start_date_col: str,
    new_col: str,
    unit: str = "days",
    valid_col: str | None = None,
) -> pd.DataFrame:
    check_columns(df,[attendee_relation_id_col,end_date_col, start_date_col])
        
    end_date_dt_col = f"{end_date_col}_datetime"
    start_date_dt_col = f"{start_date_col}_datetime"

    df[end_date_dt_col] = pd.to_datetime(
        df[end_date_col],
        dayfirst=True,
        errors='coerce'
    )
    df[start_date_dt_col] = pd.to_datetime(
        df[start_date_col],
        dayfirst=True,
        errors='coerce'
    )

    delay = df[end_date_dt_col] - df[start_date_dt_col]    # On ignore les délais négatifs

    if unit == "days":
        delay = delay.dt.days
    elif unit == "hours":
        delay = delay.dt.total_seconds() / 3600
    elif unit == "minutes":
        delay = delay.dt.total_seconds() / 60
    else:
        raise ValueError("unit must be 'days', 'hours' or 'minutes'")


    delay  = delay.where(delay >= 0)
    # On peut également filtrer les lignes selon une colonne de validité si elle est fournie
    if valid_col is not None:
        check_columns(df,[valid_col])
        delay = delay.where(df[valid_col])

    average_delay = (
        delay
        .groupby(df[attendee_relation_id_col])
        .transform('mean')
    )

    df[new_col] = average_delay.astype(float).round(1)

    return df
