# filtrer sur l'année
import pandas as pd
from datetime import datetime


def add_meeting_year(df: pd.DataFrame, meeting_start_col: str = 'Meeting Start date', year_col: str = 'Meeting Year') -> pd.DataFrame:
    """Extrait l'année de la colonne Meeting Start date et l'ajoute au DataFrame."""
    if meeting_start_col not in df.columns:
        raise KeyError(f"Colonne manquante: {meeting_start_col}")
    
    # Conversion sécurisée en datetime
    dt_series = pd.to_datetime(df[meeting_start_col], dayfirst=True, errors="coerce")
    # Extraction de l'année (les NaT deviendront NaN, qu'on peut laisser ou filtrer)
    df[year_col] = dt_series.dt.year
    return df


def convert_to_date_day_first(
        df: pd.DataFrame,
        date_col
):
    df[date_col] = pd.to_datetime(
        df[date_col],
        dayfirst=True,  # pour gérer les formats jour/mois/année
        errors="coerce" # les valeurs non convertibles deviennent NaT (équivalent datetime de NaN)
    )
    return df


def convert_to_date_year_first(
        df: pd.DataFrame,
        date_col
):
    df[date_col] = pd.to_datetime(
        df[date_col],
        yearfirst=True,  # pour gérer les formats jour/mois/année
        errors="coerce" # les valeurs non convertibles deviennent NaT (équivalent datetime de NaN)
    )
    return df


def convert_to_bool(
        df : pd.DataFrame,
        col
):
    df[col] = df[col].astype("boolean")
    return df


def convert_to_cat(
        df : pd.DataFrame,
        col
):
    df[col] = df[col].astype("category")


def convert_to_int(
        df : pd.DataFrame,
        col
):
    df[col] = df[col].astype("Int64")