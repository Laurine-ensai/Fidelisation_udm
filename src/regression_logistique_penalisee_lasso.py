"""
Modélisation prédictive du risque de désengagement et sélection de variables 
par régression logistique pénalisée LASSO (L1) au niveau adhérent.

Périmètre & Cible :
- Données d'entrée : table analytique consolidée (1 ligne / adhérent)
- Cible binaire (inversée) :
    0 = Maintien / Rétention (adhésion >= 3 ans)
    1 = Départ / Attrition précoce (< 3 ans avec démission)

Composants du pipeline :
1. Prétraitement étanche : imputation (médiane/constante) + centrage-réduction via ColumnTransformer
2. Stabilité de sélection : validation par bootstrap stratifié (500 réplications)
3. Évaluation sans biais : validation croisée OOF (Out-Of-Fold) 5-fold stratifiée
4. Diagnostic opérationnel : seuil optimal de Youden, courbe Precision-Recall et calibration
5. Exports & Dataviz : coefficients standardisés, exports Excel dans outputs/ et graphiques de décision

Installation :
    pip install scikit-learn pandas numpy matplotlib seaborn openpyxl

Usage :
    python regression_logistique_penalisee_lasso.py
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    precision_score,
    recall_score
)
from sklearn.model_selection import StratifiedKFold
from sklearn.calibration import calibration_curve
from sklearn.base import clone


# ──────────────────────────────────────────────────────────────────────
# 1. Configuration & Constantes
# ──────────────────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parent.parent if "__file__" in locals() else Path.cwd()
FILEPATH = ROOT_DIR / "data" / "base_adherents.xlsx"
OUTPUT_DIR = ROOT_DIR / "outputs" / "modelisation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "target"

COLONNES_A_EXCLURE = [
    # Variable Cible
    "target",
    
    # Identifiants / dates de construction
    "GROUPE - Nom",
    "GROUPE - ID",
    "date_cible",
]

# Variables sélectionnées
num_cols = [
    # Communication
    "nb_moyen_emails_recus_par_contact_par_an",
    "avg_taux_ouverture",
    "avg_taux_click",
    "avg_open_delay_hours_by_contact",
    "recipient_status_pct_Inactive, relation will not receive mailings (Reason: Too many bounces)",
    "comm_accept_pct_Communication - Communautés",
    "comm_accept_pct_Communication - Partner communications",
    "comm_accept_pct_Communication - Veille juridique",

    # Evénements
    "nb_moyen_inscriptions_par_contact_par_an",
    "avg_presence_rate_by_relation",
    "avg_invitation_reactivity_by_relation",
    "average_anticipation_days_by_group",
    "avg_spontaneous_participation_rate_by_relation",
]

cat_bool_cols = [
    "has_F_level",
    "has_M_level",
    "has_active_speaker",
]

RANDOM_STATE = 42
TOP_N_PLOT = 15

# ──────────────────────────────────────────────────────────────────────
# 2. Utilitaires
# ──────────────────────────────────────────────────────────────────────
LABELS_VARIABLES = {
    # Communication
    "nb_moyen_emails_recus_par_contact_par_an": "Nb moyen d'e-mails reçus / contact / an",
    "avg_taux_ouverture": "Taux d'ouverture moyen",
    "avg_taux_click": "Taux de clic moyen",
    "avg_open_delay_hours_by_contact": "Délai moyen d'ouverture (heures)",
    "recipient_status_pct_Inactive, relation will not receive mailings (Reason: Too many bounces)": "Statut inactif : Trop de rebonds (bounces)",
    "comm_accept_pct_Communication - Communautés": "Taux d'accord : Communautés",
    "comm_accept_pct_Communication - Partner communications": "Taux d'accord : Partenaires",
    "comm_accept_pct_Communication - Veille juridique": "Taux d'accord : Veille juridique",

    # Événements
    "nb_moyen_inscriptions_par_contact_par_an": "Nb moyen d'inscriptions / contact / an",
    "avg_presence_rate_by_relation": "Taux moyen de présence",
    "avg_invitation_reactivity_by_relation": "Réactivité moyenne aux invitations",
    "average_anticipation_days_by_group": "Délai moyen d'anticipation (jours)",
    "avg_spontaneous_participation_rate_by_relation": "Taux de participation spontanée",

    # Caractéristiques structurelles & booléennes
    "Organisation - Marketing Budget 2019": "Budget Marketing 2019",
    "has_F_level": "Possède un F-Level",
    "has_M_level": "Possède un M-Level",
    "has_active_speaker": "Possède un Intervenant (Speaker)"
}


# ──────────────────────────────────────────────────────────────────────
# 3. Chargement & Préparation
# ──────────────────────────────────────────────────────────────────────
def charger_donnees(filepath, target):
    """Charge le fichier et vérifie que la colonne cible existe bien."""
    df = pd.read_excel(filepath)
    if target not in df.columns:
        raise ValueError(f"La colonne cible '{target}' est absente du fichier {filepath}.")
    return df


def preparer_donnees(df, target, colonnes_a_exclure):
    """
    Ne garde que les lignes où la cible est renseignée, puis sépare X et y.
    Inverse la cible : 1 = Départ, 0 = Reste.
    """
    df_historique = df[df[target].notna()].copy()
    print(f"[AVANT] {len(df)} lignes au total, {len(df_historique)} avec target renseignée "
          f"({len(df) - len(df_historique)} exclues, pas assez de recul).")

    colonnes_manquantes = [c for c in colonnes_a_exclure if c not in df_historique.columns]
    if colonnes_manquantes:
        print(f"⚠️  Colonnes à exclure absentes du DataFrame (ignorées) : {colonnes_manquantes}")

    X = df_historique.drop(columns=colonnes_a_exclure, errors="ignore")
    
    # Inversion de la cible : 1 = Départ, 0 = Reste
    y = 1 - df_historique[target].astype(int)

    print("\n--- Définition de la cible ---")
    print("0 = Reste")
    print("1 = Départ")
    print("\nRépartition de la cible :")
    print(y.value_counts(normalize=True).round(3).to_dict())
    print(f"\n[APRÈS] X : {X.shape[0]} lignes x {X.shape[1]} colonnes")

    return X, y


def preparation_types(X, cat_bool_cols, num_cols):
    """
    Vérifie les types de colonnes et prépare les variables catégorielles booléennes.
    """
    X = X.copy()

    existing_cat_bool_cols = [col for col in cat_bool_cols if col in X.columns]
    existing_num_cols = [col for col in num_cols if col in X.columns]
    var_autres = [col for col in X.columns if col not in existing_num_cols and col not in existing_cat_bool_cols]
    
    if var_autres:
        print("\n⚠️ Variables ni dans cat_bool_cols ni dans num_cols :", var_autres)

    mapping = {
        "Former Member": 0,
        "Current Member": 1
    }

    for col in existing_cat_bool_cols:
        X[col] = X[col].replace(mapping)
        X[col] = pd.to_numeric(X[col], errors="coerce")

    return X


# ──────────────────────────────────────────────────────────────────────
# 4. Pipeline & Modélisation
# ──────────────────────────────────────────────────────────────────────
def construire_pipeline(X, num_cols, cat_bool_cols, random_state=42):
    """
    Construit le pipeline complet étanche :
    - Médiane + Standardisation sur le numérique
    - Imputation constante (0) sur le catégoriel booléen
    - LogisticRegressionCV LASSO (L1)
    """
    existing_cat_bool_cols = [col for col in cat_bool_cols if col in X.columns]
    existing_num_cols = [col for col in num_cols if col in X.columns]
    
    preprocessor = ColumnTransformer(transformers=[
        ('num', Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ]), existing_num_cols),
        ('cat_bool', Pipeline([
            ('imputer', SimpleImputer(strategy='constant', fill_value=0))
        ]), existing_cat_bool_cols)
    ])

    lasso_model = LogisticRegressionCV(
        Cs=np.logspace(-3, 2, 20),
        penalty='l1',
        solver='liblinear',
        cv=5,
        scoring='roc_auc',
        max_iter=50000,
        random_state=random_state,
        class_weight='balanced',
        n_jobs=-1
    )

    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('lasso_model', lasso_model)
    ])
    return pipeline

# ──────────────────────────────────────────────────────────────────────
# 5. Stabilité & Sélection (Bootstrap)
# ──────────────────────────────────────────────────────────────────────
def bootstrap_stabilite_lasso(
    X,
    y,
    pipeline,
    n_bootstrap=500,
    random_state=42,
):
    """
    Évalue la stabilité de la sélection LASSO par bootstrap stratifié.

    À chaque réplication :
    - tirage avec remise des RESTES
    - tirage avec remise des DÉPARTS
    - réentraînement complet du pipeline LASSO
    - récupération des coefficients

    Retour :
    - résultats détaillés de chaque bootstrap
    - résumé de stabilité par variable
    """

    rng = np.random.default_rng(random_state)

    variables = list(X.columns)

    resultats = []

    # Positions des deux classes
    idx_reste = np.where(y.to_numpy() == 0)[0]
    idx_depart = np.where(y.to_numpy() == 1)[0]

    print(
        f"\nBootstrap LASSO : {n_bootstrap} réplications"
        f"\nAdhésion longue : {len(idx_reste)}"
        f"\nAdhésion courte : {len(idx_depart)}"
    )

    for b in range(n_bootstrap):

        # ---------------------------------------------------------
        # 1. Bootstrap stratifié
        # ---------------------------------------------------------
        boot_reste = rng.choice(
            idx_reste,
            size=len(idx_reste),
            replace=True
        )

        boot_depart = rng.choice(
            idx_depart,
            size=len(idx_depart),
            replace=True
        )

        boot_idx = np.concatenate([
            boot_reste,
            boot_depart
        ])

        # Mélange
        rng.shuffle(boot_idx)

        X_boot = X.iloc[boot_idx].copy()
        y_boot = y.iloc[boot_idx].copy()

        # ---------------------------------------------------------
        # 2. Nouveau LASSO
        # ---------------------------------------------------------
        modele_boot = clone(pipeline)

        modele_boot.fit(
            X_boot,
            y_boot
        )

        # ---------------------------------------------------------
        # 3. Extraction des coefficients
        # ---------------------------------------------------------
        preprocessor = modele_boot.named_steps["preprocessor"]

        modele_lasso = modele_boot.named_steps["lasso_model"]

        feature_names = [
            nom.split("__")[-1]
            for nom in preprocessor.get_feature_names_out()
        ]

        coefficients = modele_lasso.coef_[0]

        # ---------------------------------------------------------
        # 4. Stockage
        # ---------------------------------------------------------
        for variable, coef in zip(
            feature_names,
            coefficients
        ):

            resultats.append({
                "bootstrap": b + 1,
                "variable": variable,
                "coefficient": coef,
                "selectionnee": int(coef != 0),
                "signe": (
                    1 if coef > 0
                    else -1 if coef < 0
                    else 0
                )
            })

        if (b + 1) % 50 == 0:
            print(
                f"{b + 1}/{n_bootstrap} bootstrap terminés"
            )

    # =============================================================
    # TABLEAU DÉTAILLÉ
    # =============================================================

    df_bootstrap = pd.DataFrame(
        resultats
    )

    # =============================================================
    # RÉSUMÉ
    # =============================================================

    resume = (
        df_bootstrap
        .groupby("variable")
        .agg(
            frequence_selection=(
                "selectionnee",
                "mean"
            ),
            coefficient_moyen=(
                "coefficient",
                "mean"
            ),
            coefficient_median=(
                "coefficient",
                "median"
            ),
            coefficient_std=(
                "coefficient",
                "std"
            ),
            pct_coef_positif=(
                "coefficient",
                lambda x: (x > 0).mean()
            ),
            pct_coef_negatif=(
                "coefficient",
                lambda x: (x < 0).mean()
            ),
        )
        .reset_index()
    )

    # pourcentages
    resume[
        "frequence_selection"
    ] *= 100

    resume[
        "pct_coef_positif"
    ] *= 100

    resume[
        "pct_coef_negatif"
    ] *= 100

    resume = (
        resume
        .sort_values(
            "frequence_selection",
            ascending=False
        )
        .reset_index(drop=True)
    )

    return df_bootstrap, resume

def graphique_stabilite_lasso(
    resume,
    seuil=50
):
    df_plot = (
        resume[
            resume["frequence_selection"] >= seuil
        ]
        .sort_values(
            "frequence_selection",
            ascending=True
        )
    )

    # Application des libellés de présentation
    df_plot["variable_label"] = df_plot["variable"].map(lambda x: LABELS_VARIABLES.get(x, x))

    plt.figure(figsize=(10, 6))

    plt.barh(
        df_plot["variable_label"],
        df_plot["frequence_selection"],
        color="#4c1d95"
    )

    plt.axvline(
        80,
        linestyle="--",
        label="80 %",
        color = "#e6563c"
    )

    plt.xlabel(
        "Fréquence de sélection dans les bootstrap (%)"
    )

    plt.ylabel(
        "Variable"
    )

    plt.title(
        "Stabilité de la sélection des variables par LASSO"
    )

    plt.legend()

    plt.tight_layout()

    plt.show()

# ──────────────────────────────────────────────────────────────────────
# 6. Évaluation & Diagnostics
# ──────────────────────────────────────────────────────────────────────
def validation_croisee_detaillee(df, pipeline, X, y, seuil=0.5, n_splits=5):
    """
    Validation croisée avec affichage des résultats par fold.
    predict_proba()[:, 1] = probabilité de DÉPART.
    """
    cv = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=RANDOM_STATE
    )

    resultats_folds = []
    scores_auc = []
    y_proba_depart_oof = np.zeros(len(y))

    for fold, (train_idx, test_idx) in enumerate(cv.split(X, y), start=1):
        print("\n" + "="*60)
        print(f"FOLD {fold}")
        print("="*60)

        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        modele_fold = clone(pipeline)
        print("Entraînement...")
        modele_fold.fit(X_train, y_train)

        # Probabilités de départ (classe 1)
        y_proba_depart = modele_fold.predict_proba(X_test)[:, 1]
        y_proba_depart_oof[test_idx] = y_proba_depart

        y_pred = (y_proba_depart >= seuil).astype(int)

        auc = roc_auc_score(y_test, y_proba_depart)
        scores_auc.append(auc)

        print(f"Seuil : {seuil:.3f}")
        print(f"AUC : {auc:.3f}")
        print("\nMatrice de confusion :")
        print(confusion_matrix(y_test, y_pred))
        print("\nRapport :")
        print(classification_report(y_test, y_pred, target_names=["Reste", "Départ"], zero_division=0))

        resultats = pd.DataFrame({
            "Fold": fold,
            "Organisation": df.loc[X_test.index, "GROUPE - Nom"],
            "duree_derniere_adhesion": df.loc[X_test.index, "duree_derniere_adhesion"].values,
            "Classe réelle": y_test.values,
            "Proba_depart": y_proba_depart,
            "Proba_rester": 1 - y_proba_depart,
            "Classe prédite": y_pred,
            "Erreur": y_pred != y_test.values
        })
        resultats_folds.append(resultats)

    resultats_complets = pd.concat(resultats_folds, ignore_index=True)

    cm_globale = confusion_matrix(
        resultats_complets["Classe réelle"],
        resultats_complets["Classe prédite"]
    )
    print("\nMatrice de confusion globale :")
    print(cm_globale)
    print("\nRapport de classification global :")
    print(classification_report(
        resultats_complets["Classe réelle"],
        resultats_complets["Classe prédite"],
        target_names=["Reste", "Départ"],
        zero_division=0
    ))

    # Scoring opérationnel basé sur la probabilité de départ
    resultats_complets['score_risque_depart'] = np.round(resultats_complets['Proba_depart'] * 100, 2)
    conditions = [
        (resultats_complets['score_risque_depart'] <= 30),
        (resultats_complets['score_risque_depart'] > 30) & (resultats_complets['score_risque_depart'] <= 70),
        (resultats_complets['score_risque_depart'] > 70)
    ]
    choix = ['Risque Faible', 'Risque Modéré', 'Risque Critique']
    resultats_complets['zone_alerte'] = np.select(conditions, choix, default='Inconnu')

    print("\n" + "="*60)
    print("RÉSULTAT GLOBAL")
    print("="*60)
    print(f"AUC moyen des folds : {np.mean(scores_auc):.3f} +/- {np.std(scores_auc):.3f}")

    auc_oof = roc_auc_score(y, y_proba_depart_oof)
    print(f"AUC OOF global : {auc_oof:.3f}")

    return resultats_complets, scores_auc, y_proba_depart_oof


def seuil_Youden(y_true, y_proba_depart):
    """Trouve le seuil optimal de Youden directement sur la probabilité de départ."""
    fpr, tpr, thresholds = roc_curve(y_true, y_proba_depart)
    youden_index = tpr - fpr
    meilleur_indice = np.argmax(youden_index)
    seuil = thresholds[meilleur_indice]
    
    print(f"Seuil optimal (Youden) : {seuil:.3f} "
          f"(Recall départ / Sensibilité={tpr[meilleur_indice]:.3f}, "
          f"Spécificité={1 - fpr[meilleur_indice]:.3f})")

    auc = roc_auc_score(y_true, y_proba_depart)
    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, label=f"ROC (AUC = {auc:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--", label="Hasard")
    plt.scatter(fpr[meilleur_indice], tpr[meilleur_indice], s=80, label=f"Youden = {seuil:.3f}")
    plt.xlabel("Taux de faux positifs (FPR)")
    plt.ylabel("Taux de vrais positifs (TPR)")
    plt.title("Courbe ROC et seuil optimal de Youden")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()

    return seuil


def seuil_et_recall_depart(y_true, y_proba_depart):
    """Calcule le rappel de la classe 1 (Départ) selon différents seuils."""
    seuils = np.arange(0.01, 1.00, 0.01)
    resultats = []
    for seuil in seuils:
        y_pred = (y_proba_depart >= seuil).astype(int)
        recall_1 = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
        resultats.append({
            "seuil": seuil,
            "recall_classe_1_depart": recall_1
        })
    return pd.DataFrame(resultats)


def courbe_precision_recall(y_true, y_proba_depart, seuils=None):
    """Trace la courbe Precision-Recall pour la classe 1 (Départ)."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba_depart)

    plt.figure(figsize=(7, 6))
    plt.plot(recall, precision, label="Courbe Precision-Recall")

    if seuils is not None:
        if isinstance(seuils, (int, float)):
            seuils = [seuils]
        for seuil in seuils:
            y_pred = (y_proba_depart >= seuil).astype(int)
            p_seuil = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
            r_seuil = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
            plt.scatter(r_seuil, p_seuil, s=100, label=f"Seuil = {seuil:.3f} (P={p_seuil:.2f}, R={r_seuil:.2f})")

    plt.xlabel("Recall – Départs")
    plt.ylabel("Precision – Départs")
    plt.title("Courbe Precision-Recall – classe 1 (Départ)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

    return precision, recall, thresholds


def calibration_plot(y_true, y_proba_depart):
    """Vérifie la calibration des probabilités de départ."""
    prob_true, prob_pred = calibration_curve(y_true, y_proba_depart, n_bins=5, strategy="quantile")

    plt.figure(figsize=(7, 6))
    plt.plot(prob_pred, prob_true, marker="o", label="Modèle")
    plt.plot([0, 1], [0, 1], linestyle="--", label="Calibration parfaite")
    plt.xlabel("Probabilité prédite de départ")
    plt.ylabel("Proportion réelle de départs")
    plt.title("Calibration des probabilités de départ")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


# ──────────────────────────────────────────────────────────────────────
# 7. Interprétation du modèle final
# ──────────────────────────────────────────────────────────────────────
def extraire_variables_selectionnees(pipeline_final):
    """Récupère les coefficients du modèle et analyse leur direction d'impact."""
    preprocessor = pipeline_final.named_steps['preprocessor']
    logistic_model = pipeline_final.named_steps['lasso_model']
    
    feature_names = preprocessor.get_feature_names_out()
    coefficients = logistic_model.coef_[0]

    df_influence = pd.DataFrame({
        "Variable": feature_names,
        "Coefficient": coefficients
    })
    df_influence["Variable"] = df_influence["Variable"].str.split("__").str[-1]
    df_influence['Importance_Absolue'] = df_influence['Coefficient'].abs()
    df_influence = df_influence.sort_values(by='Importance_Absolue', ascending=False).drop(columns=['Importance_Absolue'])
    
    # 1 = Départ : un coefficient positif favorise le départ (attrition)
    df_influence['Impact'] = np.where(
        df_influence['Coefficient'] > 0,
        "Pousse au départ (Attrition)",
        "Retient le membre (Rétention)"
    )

    variables_conservees = df_influence[df_influence['Coefficient'] != 0]
    variables_rejetees = df_influence[df_influence['Coefficient'] == 0]['Variable'].tolist()

    print(f"\n{len(variables_conservees)} variables conservées par LASSO / "
          f"{len(variables_rejetees)} rejetées (coefficient = 0).")

    return df_influence, variables_conservees, variables_rejetees


def afficher_influence_variables(variables_conservees, top_n=15):
    """
    Graphique à barres :
    - Rouge : coefficient > 0 (Pousse au départ)
    - Bleu/Violet : coefficient < 0 (Favorise la rétention)
    """
    df_plot = variables_conservees.head(top_n).copy()
    if df_plot.empty:
        print("Aucune variable avec un coefficient non nul à afficher.")
        return

    # Application des libellés de présentation
    df_plot['Variable_label'] = df_plot['Variable'].map(lambda x: LABELS_VARIABLES.get(x, x))

    plt.figure(figsize=(10, 6))
    couleurs = ['#ff5a36' if c > 0 else '#533fe4' for c in df_plot['Coefficient']]
    sns.barplot(x='Coefficient', y='Variable_label', data=df_plot, palette=couleurs, hue='Variable', legend=False)
    plt.title(f"Variables sélectionnées par le LASSO (Cible = Départ) – Top {top_n}", fontsize=14, fontweight='bold')    
    plt.xlabel("Coefficient du modèle LASSO (> 0 : Risque de départ accru)")    
    plt.ylabel("Variable")
    plt.axvline(x=0, color='black', linestyle='--', linewidth=1)
    plt.tight_layout()
    plt.show()


# ──────────────────────────────────────────────────────────────────────
# 8. Bloc d'exécution
# ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    df = charger_donnees(FILEPATH, TARGET)
    X, y = preparer_donnees(df, TARGET, COLONNES_A_EXCLURE)
    variables_modele = num_cols + cat_bool_cols
    X = X[variables_modele].copy()    
    X = preparation_types(X, cat_bool_cols, num_cols)
    
    pipeline = construire_pipeline(X, num_cols, cat_bool_cols, random_state=RANDOM_STATE)

    # Rééchantillonage bootstrap pour évaluer la stabilité de la sélection LASSO
    df_bootstrap_lasso, resume_bootstrap_lasso = (
        bootstrap_stabilite_lasso(
            X=X,
            y=y,
            pipeline=pipeline,
            n_bootstrap=500,
            random_state=RANDOM_STATE
        )
    )
    graphique_stabilite_lasso(
        resume_bootstrap_lasso
    )

    # Validation croisée : calcul des performances et scores OOF
    resultats_complets, scores_auc, y_proba_depart_oof = validation_croisee_detaillee(
        df, pipeline, X, y, seuil=0.368, n_splits=5
    )
    resultats_complets.to_excel(OUTPUT_DIR / "df_scored_lasso.xlsx", index=False)
    print("\n✓ Fichier 'df_scored_lasso.xlsx' exporté avec succès.")

    # 2. Évaluation des seuils et calibration
    seuil_optimal = seuil_Youden(y, y_proba_depart_oof)
    df_seuils_recall = seuil_et_recall_depart(y, y_proba_depart_oof)
    courbe_precision_recall(y, y_proba_depart_oof, seuils=[0.5, seuil_optimal])
    calibration_plot(y, y_proba_depart_oof)

    # 3. Ré-entraînement sur 100% des données pour interprétation
    pipeline_final = clone(pipeline)
    pipeline_final.fit(X, y)

    # 4. Analyse de l'importance des variables
    df_influence, variables_conservees, variables_rejetees = extraire_variables_selectionnees(pipeline_final)
    afficher_influence_variables(variables_conservees, TOP_N_PLOT)

    print("\nVariables sélectionnées par LASSO :")
    print(variables_conservees)