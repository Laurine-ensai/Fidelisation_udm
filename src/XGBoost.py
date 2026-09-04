"""
Modélisation prédictive du risque de désengagement et analyse d'impact
par gradient boosting (XGBoost) et explicabilité SHAP au niveau adhérent.

Périmètre & Cible :
- Données d'entrée : table analytique consolidée (1 ligne / adhérent)
- Cible binaire (inversée) :
    0 = Maintien / Rétention (adhésion >= 3 ans)
    1 = Départ / Attrition précoce (< 3 ans avec démission)

Composants du pipeline :
1. Prétraitement étanche : imputation (médiane/constante) via ColumnTransformer
2. Nested CV (5-fold externe x 5-fold interne) : optimisation fine par GridSearchCV
3. Évaluation sans biais : métriques OOF (Out-Of-Fold), seuil de Youden, courbe PR et calibration
4. Stabilité SHAP OOB : validation de l'importance des variables par bootstrap stratifié
5. Explicabilité : SHAP globale (beeswarm/bar), locale (waterfall) et interactions croisées
6. Inférence opérationnelle : scoring des profils sans cible et cohortes récentes

Installation :
    pip install xgboost shap scikit-learn pandas numpy matplotlib seaborn openpyxl

Usage :
    python modelisation_xgboost.py
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns

from sklearn.base import clone
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    classification_report,
    confusion_matrix,
)
from sklearn.calibration import calibration_curve

from xgboost import XGBClassifier
import shap


# ──────────────────────────────────────────────────────────────────────
# 1. Configuration & Constantes
# ──────────────────────────────────────────────────────────────────────
# # Créer une colormap personnalisée : [négatif, positif]
# # (SHAP applique le premier pour les baisses, le second pour les hausses)
# shap.plots.colors.red_blue = mcolors.LinearSegmentedColormap.from_list(
#     "custom_red_blue", ["#533fe4", "#ff5a36"]
# )

ROOT_DIR = Path(__file__).resolve().parent.parent if "__file__" in locals() else Path.cwd()
FILEPATH = ROOT_DIR / "data" / "base_adherents.xlsx"
OUTPUT_DIR = ROOT_DIR / "outputs" / "modelisation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
SEUIL_DECISION = 0.221

TARGET = "target"

COLONNES_A_EXCLURE = [
    # Variable Cible
    "target",
    
    # Identifiants / dates de construction
    "GROUPE - Nom",
    "GROUPE - ID",
    "date_cible",  
]

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

N_SPLITS = 5
N_SPLITS_INTERNE = 5
TEST_SIZE = 0.2
RANDOM_STATE = 42
TOP_N_PLOT = 20


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

def formater_nom_variable(nom):
    """Renvoie le libellé propre s'il existe, sinon conserve le nom technique."""
    return LABELS_VARIABLES.get(nom, nom)
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
    Ne garde que les lignes où la cible est renseignée.

    Nouvelle convention :
        0 = Reste
        1 = Départ
    """

    df_historique = df[df[target].notna()].copy()

    print(
        f"[AVANT] {len(df)} lignes au total, "
        f"{len(df_historique)} avec target renseignée "
        f"({len(df) - len(df_historique)} exclues, pas assez de recul)."
    )

    colonnes_manquantes = [
        c for c in colonnes_a_exclure
        if c not in df_historique.columns
    ]

    if colonnes_manquantes:
        print(
            "⚠️ Colonnes à exclure absentes du DataFrame "
            f"(ignorées) : {colonnes_manquantes}"
        )

    X = df_historique.drop(
        columns=colonnes_a_exclure,
        errors="ignore"
    )

    # ============================================================
    # INVERSION DE LA CIBLE
    #
    # Avant :
    #   0 = départ
    #   1 = rester
    #
    # Maintenant :
    #   0 = rester
    #   1 = départ
    # ============================================================

    y = 1 - df_historique[target].astype(int)

    print("\n--- Définition de la cible ---")
    print("0 = Reste")
    print("1 = Départ")

    print("\nRépartition de la cible :")
    print(y.value_counts())

    print("\nRépartition en pourcentage :")
    print(
        y.value_counts(normalize=True)
        .round(3)
        .to_dict()
    )

    print(
        f"\n[APRÈS] X : {X.shape[0]} lignes x "
        f"{X.shape[1]} colonnes"
    )

    return X, y

def preparation_types(X, cat_bool_cols, num_cols):
    """
    Vérifie les types de colonnes et prépare les variables catégorielles booléennes :
    - transforme les valeurs texte binaires en 0/1
    - convertit toutes les colonnes en float64
    """
    X = X.copy()

    existing_cat_bool_cols = [
        col for col in cat_bool_cols
        if col in X.columns
    ]
    existing_num_cols = [
        col for col in num_cols
        if col in X.columns
    ]
    var_autres = [
        col for col in X.columns
        if col not in existing_num_cols
        and col not in existing_cat_bool_cols
    ]
    if var_autres:
        print(
            "\n⚠️ Variables ni dans cat_bool_cols ni dans num_cols :"
        )
        print(var_autres)


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
def construire_pipeline(
    X,
    y,
    num_cols,
    cat_bool_cols,
    random_state=42
):
    """
    Construit le pipeline XGBoost.

    Convention :
        0 = RESTE
        1 = DÉPART

    scale_pos_weight :
        poids de la classe 1 (départ)
        = nombre de restants / nombre de départs
    """

    existing_cat_bool_cols = [
        col for col in cat_bool_cols
        if col in X.columns
    ]

    existing_num_cols = [
        col for col in num_cols
        if col in X.columns
    ]

    preprocessor = ColumnTransformer(
        transformers=[

            (
                'num',
                Pipeline([
                    (
                        'imputer',
                        SimpleImputer(strategy='median')
                    ),
                ]),
                existing_num_cols
            ),

            (
                'cat_bool',
                Pipeline([
                    (
                        'imputer',
                        SimpleImputer(
                            strategy='constant',
                            fill_value=0
                        )
                    )
                ]),
                existing_cat_bool_cols
            )
        ]
    )

    # ---------------------------------------------------------------
    # NOUVELLE CONVENTION :
    #
    # 0 = RESTE
    # 1 = DÉPART
    #
    # scale_pos_weight agit sur la classe 1 = DÉPART
    # ---------------------------------------------------------------

    nb_reste = (y == 0).sum()
    nb_depart = (y == 1).sum()

    scale_pos_weight = nb_reste / nb_depart

    print("\n--- Pondération XGBoost ---")
    print(f"Nombre de restants : {nb_reste}")
    print(f"Nombre de départs  : {nb_depart}")
    print(
        f"scale_pos_weight   : "
        f"{scale_pos_weight:.3f}"
    )

    XGB_model = XGBClassifier(
        n_estimators=300,
        max_depth=3,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,

        scale_pos_weight=scale_pos_weight,

        objective="binary:logistic",
        eval_metric="auc",

        random_state=random_state,
        n_jobs=-1,
    )

    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('XGB_model', XGB_model)
    ])

    return pipeline


# ──────────────────────────────────────────────────────────────────────
# 5. Stabilité & Importance SHAP (Bootstrap OOB)
# ──────────────────────────────────────────────────────────────────────
def bootstrap_stabilite_shap(X, y, pipeline_final, n_bootstrap=300, top_k=5, random_state=42):
    """Évalue la stabilité des importances SHAP par bootstrap stratifié Out-Of-Bag."""
    rng = np.random.default_rng(random_state)
    idx_reste = np.where(y.to_numpy() == 0)[0]
    idx_depart = np.where(y.to_numpy() == 1)[0]
    indices_tous = np.arange(len(X))
    resultats = []

    print(f"\nBootstrap SHAP : {n_bootstrap} réplications")

    for b in range(n_bootstrap):
        boot_reste = rng.choice(idx_reste, size=len(idx_reste), replace=True)
        boot_depart = rng.choice(idx_depart, size=len(idx_depart), replace=True)
        boot_idx = np.concatenate([boot_reste, boot_depart])
        rng.shuffle(boot_idx)

        oob_idx = np.setdiff1d(indices_tous, np.unique(boot_idx))
        if len(oob_idx) == 0:
            continue

        X_boot, y_boot = X.iloc[boot_idx].copy(), y.iloc[boot_idx].copy()
        X_oob = X.iloc[oob_idx].copy()

        modele_boot = clone(pipeline_final)
        modele_boot.fit(X_boot, y_boot)

        preprocessor = modele_boot.named_steps["preprocessor"]
        xgb_model = modele_boot.named_steps["XGB_model"]

        X_oob_trans = preprocessor.transform(X_oob)
        feature_names = [
            formater_nom_variable(nom.split("__")[-1])
            for nom in preprocessor.get_feature_names_out()
        ]
        X_oob_df = pd.DataFrame(X_oob_trans, columns=feature_names, index=X_oob.index)

        explainer = shap.TreeExplainer(xgb_model)
        shap_vals = explainer(X_oob_df)
        importance = np.abs(shap_vals.values).mean(axis=0)

        df_imp = pd.DataFrame({"variable": feature_names, "mean_abs_shap": importance})
        df_imp = df_imp.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
        df_imp["rang"] = np.arange(len(df_imp)) + 1
        df_imp["top_k"] = df_imp["rang"] <= top_k
        df_imp["bootstrap"] = b + 1
        resultats.append(df_imp)

        if (b + 1) % 50 == 0:
            print(f"{b + 1}/{n_bootstrap} itérations bootstrap terminées")

    df_bootstrap_shap = pd.concat(resultats, ignore_index=True)
    resume_shap = (
        df_bootstrap_shap.groupby("variable")
        .agg(
            shap_moyen=("mean_abs_shap", "mean"),
            shap_median=("mean_abs_shap", "median"),
            shap_std=("mean_abs_shap", "std"),
            rang_median=("rang", "median"),
            frequence_top_k=("top_k", "mean")
        )
        .reset_index()
    )
    resume_shap["frequence_top_k"] *= 100
    resume_shap = resume_shap.sort_values(
        ["frequence_top_k", "rang_median"], ascending=[False, True]
    ).reset_index(drop=True)

    return df_bootstrap_shap, resume_shap


# ──────────────────────────────────────────────────────────────────────
# 6. Évaluation & Diagnostics
# ──────────────────────────────────────────────────────────────────────
def validation_croisee_detaillee(
    df,
    X,
    y,
    num_cols,
    cat_bool_cols,
    random_state,
    seuil=0.5,
    n_splits=5,
    n_splits_interne=5,
):
    """
    Validation croisée avec affichage des résultats par fold.

    Convention :
        0 = RESTE
        1 = DÉPART

    Donc :
        predict_proba()[:, 1] = probabilité de DÉPART
    """

    cv = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=RANDOM_STATE
    )

    resultats_folds = []
    scores_auc = []

    # Pour conserver les hyperparamètres choisis
    # dans chaque fold externe
    meilleurs_params_folds = []

    # Probabilités OOF de DÉPART
    y_proba_depart_oof = np.zeros(len(y))

    for fold, (train_idx, test_idx) in enumerate(
        cv.split(X, y),
        start=1
    ):

        print("\n" + "=" * 60)
        print(f"FOLD {fold}")
        print("=" * 60)

        # -----------------------------------------------------------
        # Séparation du fold
        # -----------------------------------------------------------

        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]

        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx]

        print(
            f"Taille entraînement : {len(X_train)}"
        )
        print(
            f"Taille validation   : {len(X_test)}"
        )

        print(
            "\nRépartition entraînement :"
        )
        print(
            y_train.value_counts().sort_index()
        )

        print(
            "\nRépartition validation :"
        )
        print(
            y_test.value_counts().sort_index()
        )

        # ============================================================
        # 3. CONSTRUCTION DU PIPELINE POUR LE FOLD
        # ============================================================

        pipeline = construire_pipeline(
            X_train,
            y_train,
            num_cols,
            cat_bool_cols,
            random_state=RANDOM_STATE
        )

        # ============================================================
        # 4. GRILLE D'HYPERPARAMÈTRES
        # ============================================================
        param_grid = {
            'XGB_model__n_estimators': [60, 100, 150],
            'XGB_model__max_depth': [1, 2, 3],
            'XGB_model__learning_rate': [0.03, 0.05],
            'XGB_model__min_child_weight': [3, 5],
            'XGB_model__subsample': [0.7, 0.8],
            'XGB_model__colsample_bytree': [0.7, 0.8],
            'XGB_model__reg_lambda': [1.0, 3.0]
        }
        # ============================================================
        # 5. CV INTERNE
        # ============================================================
        cv_interne = StratifiedKFold(
            n_splits=n_splits_interne,
            shuffle=True,
            random_state=random_state
        )

        grid_search = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grid,
            scoring="roc_auc",
            cv=cv_interne,
            n_jobs=-1,
            verbose=0,
            refit=True
        )

        print(
            "\nRecherche des hyperparamètres "
            "sur le train uniquement..."
        )

        grid_search.fit(
            X_train,
            y_train
        )

        # ============================================================
        # 6. MEILLEURS HYPERPARAMÈTRES
        # ============================================================

        meilleurs_params = grid_search.best_params_

        meilleurs_params_folds.append({
            "Fold": fold,
            **meilleurs_params,
            "ROC_AUC_CV_interne": grid_search.best_score_
        })

        for param, valeur in meilleurs_params.items():
            print(
                f"  {param} = {valeur}"
            )

        print(
            f"✓ ROC-AUC CV interne : "
            f"{grid_search.best_score_:.3f}"
        )

        print(
            "\n✓ Meilleurs hyperparamètres :"
        )

        for param, valeur in meilleurs_params.items():
            print(
                f"  {param} = {valeur}"
            )

        print(
            f"✓ ROC-AUC CV interne : "
            f"{grid_search.best_score_:.3f}"
        )

        # ============================================================
        # 7. MEILLEUR MODÈLE DU FOLD
        # ============================================================

        modele_fold = grid_search.best_estimator_

        # ============================================================
        # 8. PRÉDICTION SUR LE FOLD EXTERNE
        # ============================================================

        # IMPORTANT :
        # X_test n'a jamais participé au GridSearchCV

        y_proba_depart = modele_fold.predict_proba(
            X_test
        )[:, 1]

        # Stockage OOF
        y_proba_depart_oof[test_idx] = y_proba_depart

        # -----------------------------------------------------------
        # PRÉDICTION AVEC LE SEUIL
        # -----------------------------------------------------------

        y_pred = (
            y_proba_depart >= seuil
        ).astype(int)

        # -----------------------------------------------------------
        # AUC
        # -----------------------------------------------------------

        auc = roc_auc_score(
            y_test,
            y_proba_depart
        )

        scores_auc.append(auc)

        print(f"Seuil : {seuil:.3f}")
        print(f"AUC : {auc:.3f}")

        # -----------------------------------------------------------
        # MATRICE DE CONFUSION
        # -----------------------------------------------------------

        print("\nMatrice de confusion :")

        print(
            confusion_matrix(
                y_test,
                y_pred
            )
        )

        # -----------------------------------------------------------
        # RAPPORT
        # -----------------------------------------------------------

        print("\nRapport :")

        print(
            classification_report(
                y_test,
                y_pred,
                target_names=[
                    "Reste",
                    "Départ"
                ],
                zero_division=0
            )
        )

        # -----------------------------------------------------------
        # TABLEAU DES RÉSULTATS
        # -----------------------------------------------------------

        resultats = pd.DataFrame({

            "Fold": fold,

            "Organisation":
                df.loc[
                    X_test.index,
                    "GROUPE - Nom"
                ],

            "Durée adhésion" :
                df.loc[
                    X_test.index,
                    "duree_derniere_adhesion"
                ],

            "Année adhésion" :
                            df.loc[
                                X_test.index,
                                "GROUPE - Année adhésion *"
                            ],

            "Année démission" :
                            df.loc[
                                X_test.index,
                                "GROUPE - Année démission *"
                            ],

            "Classe réelle":
                y_test.values,

            "Proba_depart":
                y_proba_depart,

            "Proba_rester":
                1 - y_proba_depart,

            "Classe prédite":
                y_pred,

            "Erreur":
                y_pred != y_test.values
        })

        resultats_folds.append(resultats)

    # ────────────────────────────────────────────────────────────────
    # FUSION DES FOLDS
    # ────────────────────────────────────────────────────────────────

    resultats_complets = pd.concat(
        resultats_folds,
        ignore_index=True
    )

    # ────────────────────────────────────────────────────────────────
    # MATRICE DE CONFUSION GLOBALE
    # ────────────────────────────────────────────────────────────────

    cm_globale = confusion_matrix(
        resultats_complets["Classe réelle"],
        resultats_complets["Classe prédite"]
    )

    print("\nMatrice de confusion globale :")
    print(cm_globale)

    print("\nRapport de classification global :")

    print(
        classification_report(
            resultats_complets["Classe réelle"],
            resultats_complets["Classe prédite"],
            target_names=[
                "Reste",
                "Départ"
            ],
            zero_division=0
        )
    )

    # ────────────────────────────────────────────────────────────────
    # SCORE DE RISQUE DE DÉPART
    # ────────────────────────────────────────────────────────────────

    resultats_complets[
        "score_risque_depart"
    ] = np.round(
        resultats_complets[
            "Proba_depart"
        ] * 100,
        2
    )

    conditions = [
        (
            resultats_complets[
                "score_risque_depart"
            ] <= 30
        ),

        (
            resultats_complets[
                "score_risque_depart"
            ] > 30
        )
        &
        (
            resultats_complets[
                "score_risque_depart"
            ] <= 70
        ),

        (
            resultats_complets[
                "score_risque_depart"
            ] > 70
        )
    ]

    choix = [
        "Risque Faible",
        "Risque Modéré",
        "Risque Critique"
    ]

    resultats_complets[
        "zone_alerte"
    ] = np.select(
        conditions,
        choix,
        default="Inconnu"
    )

    # ────────────────────────────────────────────────────────────────
    # RÉSULTAT GLOBAL
    # ────────────────────────────────────────────────────────────────

    print("\n" + "=" * 60)
    print("RÉSULTAT GLOBAL")
    print("=" * 60)

    print(
        f"AUC moyen : "
        f"{np.mean(scores_auc):.3f} "
        f"+/- "
        f"{np.std(scores_auc):.3f}"
    )

    auc_oof = roc_auc_score(
        y,
        y_proba_depart_oof
    )

    print(
        f"AUC OOF global : "
        f"{auc_oof:.3f}"
    )

    return (
        resultats_complets,
        scores_auc,
        y_proba_depart_oof,
        meilleurs_params_folds
    )


def seuil_Youden(y_true, y_proba_depart):
    """Trouve le seuil optimal via l'indice de Youden sur les prédictions OOF."""
    fpr, tpr, thresholds = roc_curve(y_true, y_proba_depart)
    youden_index = tpr - fpr
    meilleur_indice = np.argmax(youden_index)
    seuil = thresholds[meilleur_indice]

    print(f"Seuil optimal (Youden) : {seuil:.3f} "
          f"(Sensibilité={tpr[meilleur_indice]:.3f}, Spécificité={1 - fpr[meilleur_indice]:.3f})")

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
    plt.tight_layout()
    plt.show()

    return seuil

def seuil_et_recall_depart(y_true, y_proba_depart):
    """Calcule le rappel de la classe 1 (Départ) selon une plage continue de seuils."""
    seuils = np.arange(0.01, 1.00, 0.01)
    resultats = []
    for seuil in seuils:
        y_pred = (y_proba_depart >= seuil).astype(int)
        recall = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
        resultats.append({"seuil": seuil, "recall_classe_1_depart": recall})
    return pd.DataFrame(resultats)

def courbe_precision_recall(y_true, y_proba_depart, seuils=None):
    """Trace la courbe Precision-Recall pour la classe critique (Départ)."""
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
    """Contrôle la fidélité des probabilités prédites."""
    prob_true, prob_pred = calibration_curve(y_true, y_proba_depart, n_bins=5, strategy="quantile")

    plt.figure(figsize=(7, 6))
    plt.plot(prob_pred, prob_true, marker="o", label="XGBoost")
    plt.plot([0, 1], [0, 1], linestyle="--", label="Calibration parfaite")
    plt.xlabel("Probabilité prédite de départ")
    plt.ylabel("Proportion réelle de départs")
    plt.title("Calibration des probabilités de départ")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

# ──────────────────────────────────────────────────────────────────────
# 7. Interprétation du modèle final (SHAP) & Inférence
# ──────────────────────────────────────────────────────────────────────

def analyser_shap(
    pipeline_final,
    X,
    nom_organisation="ACCOR",
    top_n=15
):
    """
    Génère les graphiques SHAP :

    1. SHAP globale
    2. SHAP individuelle
    3. SHAP interactions

    Attention :
        le modèle prédit maintenant la probabilité de DÉPART.
    """

    print("\n" + "=" * 60)
    print("CALCUL ET ANALYSE SHAP")
    print("=" * 60)

    # ────────────────────────────────────────────────────────────────
    # EXTRACTION DES DONNÉES PRÉTRAITÉES
    # ────────────────────────────────────────────────────────────────

    preprocessor = (
        pipeline_final
        .named_steps["preprocessor"]
    )

    xgb_model = (
        pipeline_final
        .named_steps["XGB_model"]
    )

    X_transformed_array = (
        preprocessor.transform(X)
    )

    # Noms des variables
    if hasattr(
        preprocessor,
        "get_feature_names_out"
    ):

        raw_feature_names = [
            f.split("__")[-1]
            for f in
            preprocessor.get_feature_names_out()
        ]

    else:

        existing_num_cols = [
            c for c in num_cols
            if c in X.columns
        ]

        existing_cat_cols = [
            c for c in cat_bool_cols
            if c in X.columns
        ]

        raw_feature_names = (
            existing_num_cols
            + existing_cat_cols
        )

    # 👉 APPLICATION DU MAPPING ICI (Tous les plots SHAP hériteront de ces noms)
    feature_names = [LABELS_VARIABLES.get(f, f) for f in raw_feature_names]

    X_transformed = pd.DataFrame(
        X_transformed_array,
        columns=feature_names,
        index=X.index
    )

    # ────────────────────────────────────────────────────────────────
    # EXPLAINER
    # ────────────────────────────────────────────────────────────────

    explainer = shap.TreeExplainer(
        xgb_model
    )

    shap_values = explainer(
        X_transformed
    )

    # ────────────────────────────────────────────────────────────────
    # A. SHAP GLOBALE
    # ────────────────────────────────────────────────────────────────

    print(
        "\n--- 1. Graphiques SHAP Globaux ---"
    )

    plt.figure(
        figsize=(10, 8)
    )

    plt.title(
        "Importance Globale et Impact des Variables "
        "(SHAP Beeswarm)",
        fontsize=12
    )

    shap.plots.beeswarm(
        shap_values,
        max_display=top_n,
        show=False
    )

    plt.tight_layout()
    plt.show()

    # Bar plot
    plt.figure(
        figsize=(10, 6)
    )

    plt.title(
        "Importance Moyenne Absolue (|SHAP Value|)",
        fontsize=12
    )

    shap.plots.bar(
        shap_values,
        max_display=top_n,
        show=False,
    )

    # 2. Appliquer la couleur personnalisée à toutes les barres
    ax = plt.gca()
    for patch in ax.patches:
        patch.set_facecolor("#ff5a36")

    plt.tight_layout()
    plt.show()

    # ────────────────────────────────────────────────────────────────
    # B. SHAP INDIVIDUELLE
    # ────────────────────────────────────────────────────────────────

    print(
        "\n--- 2. Graphique SHAP Individuel ---"
    )

    if nom_organisation in df["GROUPE - Nom"].values:

        idx_exemple = df[
            df["GROUPE - Nom"]
            == nom_organisation
        ].index[0]

        plt.figure(
            figsize=(10, 6)
        )

        plt.title(
            f"Explication SHAP pour : "
            f"{nom_organisation}",
            fontsize=12
        )

        shap.plots.waterfall(
            shap_values[idx_exemple],
            max_display=12,
            show=False,
        )

        plt.tight_layout()
        plt.show()

    else:

        print(
            f"⚠️ L'organisation "
            f"'{nom_organisation}' "
            f"est introuvable dans le jeu de données."
        )

    # ────────────────────────────────────────────────────────────────
    # C. SHAP INTERACTIONS
    # ────────────────────────────────────────────────────────────────

    print(
        "\n--- 3. Graphiques d'Interactions SHAP ---"
    )

    print(
        "Calcul des valeurs d'interaction SHAP en cours..."
    )

    interaction_values = (
        explainer.shap_interaction_values(
            X_transformed
        )
    )

    print("✓ Terminé")

    # Moyenne absolue
    mean_interaction = (
        np.abs(interaction_values)
        .mean(axis=0)
    )

    # Suppression de la diagonale
    mean_interaction_hors_diag = (
        mean_interaction.copy()
    )

    np.fill_diagonal(
        mean_interaction_hors_diag,
        0
    )

    # Score d'interaction par variable
    score_interaction_par_variable = (
        mean_interaction_hors_diag
        .sum(axis=0)
    )

    # Top variables interactives
    top_k = min(
        10,
        len(feature_names)
    )

    top_k_indices = (
        np.argsort(
            score_interaction_par_variable
        )[::-1][:top_k]
    )

    matrix_top_k = mean_interaction[
        np.ix_(
            top_k_indices,
            top_k_indices
        )
    ]

    top_k_names = [
        feature_names[i]
        for i in top_k_indices
    ]

    # Masque de la diagonale
    matrix_top_k_affichage = (
        matrix_top_k.copy()
    )

    np.fill_diagonal(
        matrix_top_k_affichage,
        np.nan
    )

    plt.figure(
        figsize=(9, 7)
    )
    custom_cmap = mcolors.LinearSegmentedColormap.from_list(
        "custom_purple", ["#ffffff", "#533fe4"]
    )

    sns.heatmap(
        matrix_top_k_affichage,
        xticklabels=top_k_names,
        yticklabels=top_k_names,
        annot=True,
        fmt=".3f",
        cmap=custom_cmap,
        mask=np.isnan(
            matrix_top_k_affichage
        )
    )

    plt.title(
        "Matrice des Interactions SHAP Croisées "
        "(Top Variables les plus interactives)",
        fontsize=12
    )

    plt.xticks(
        rotation=45,
        ha="right"
    )

    plt.tight_layout()
    plt.show()

    return (
        explainer,
        shap_values,
        interaction_values
    )

def generer_predictions_operationnelles(df, pipeline_final, variables_modele, cat_bool_cols, num_cols, resultats_complets, seuil=0.221):
    """Produit le tableau prédictif unifié (organisations sans cible + cohortes récentes OOF)."""
    df_a_predire = df[df[TARGET].isna()].copy()
    print(f"\nNombre d'organisations sans cible historique à scorer : {len(df_a_predire)}")

    # Identification des cohortes récentes avec score OOF
    masque_recentes = (
        pd.to_numeric(df["GROUPE - Année adhésion *"], errors="coerce").ge(2023)
        & df["GROUPE - Année démission *"].isna()
        & df[TARGET].notna()
    )
    entreprises_recentes = df.loc[masque_recentes, "GROUPE - Nom"].tolist()

    df_recentes_oof = resultats_complets[resultats_complets["Organisation"].isin(entreprises_recentes)].copy()
    df_recentes_oof = df_recentes_oof.rename(columns={"Classe prédite": "Classe_predite", "Classe réelle": "Classe_reelle"})
    df_recentes_oof["Type_score"] = "Prédiction Out-of-Fold"

    # Inférence sur les données sans cible
    X_a_predire = df_a_predire[variables_modele].copy()
    X_a_predire = preparation_types(X_a_predire, cat_bool_cols, num_cols)

    proba_depart = pipeline_final.predict_proba(X_a_predire)[:, 1]
    df_a_predire["Proba_depart"] = proba_depart
    df_a_predire["Proba_rester"] = 1 - proba_depart
    df_a_predire["score_risque_depart"] = np.round(proba_depart * 100, 2)
    df_a_predire["Classe_predite"] = (proba_depart >= seuil).astype(int)
    df_a_predire["Classe_reelle"] = np.nan
    df_a_predire["Type_score"] = "Prédiction modèle final"

    cond_alerte = [
        df_a_predire["score_risque_depart"] <= 30,
        (df_a_predire["score_risque_depart"] > 30) & (df_a_predire["score_risque_depart"] <= 70),
        df_a_predire["score_risque_depart"] > 70
    ]
    df_a_predire["zone_alerte"] = np.select(cond_alerte, ["Risque Faible", "Risque Modéré", "Risque Critique"], default="Inconnu")
    df_a_predire = df_a_predire.rename(columns={"GROUPE - Nom": "Organisation"})

    colonnes_finales = [
        "Organisation", "Proba_depart", "Proba_rester",
        "score_risque_depart", "Classe_predite", "Classe_reelle",
        "zone_alerte", "Type_score"
    ]
    df_predictions = pd.concat([df_a_predire[colonnes_finales], df_recentes_oof[colonnes_finales]], ignore_index=True)
    df_predictions = df_predictions.sort_values("Proba_depart", ascending=False).reset_index(drop=True)
    df_predictions[["Proba_depart", "Proba_rester", "score_risque_depart"]] = df_predictions[["Proba_depart", "Proba_rester", "score_risque_depart"]].round(2)

    return df_predictions



# ──────────────────────────────────────────────────────────────────────
# ORCHESTRATION
# ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    df = charger_donnees(FILEPATH, TARGET)
    X, y = preparer_donnees(df, TARGET, COLONNES_A_EXCLURE)
    variables_modele = num_cols + cat_bool_cols
    X = X[variables_modele].copy()
    X = preparation_types(X, cat_bool_cols, num_cols)

    # 2. Validation croisée imbriquée (Nested CV) et scoring OOF
    resultats_complets, scores_auc, y_proba_oof, meilleurs_params_folds = validation_croisee_detaillee(df, X, y, num_cols=num_cols, cat_bool_cols=cat_bool_cols, n_splits=N_SPLITS, n_splits_interne=N_SPLITS_INTERNE, random_state=RANDOM_STATE, seuil=SEUIL_DECISION)
    resultats_complets.to_excel(OUTPUT_DIR / "df_scored_xgboost.xlsx", index=False) # export des scores out-of-fold
    print("\n✓ Fichier 'df_scored_xgboost.xlsx' exporté avec succès.")

    # 3. Diagnostic des seuils opérationnels et calibration
    seuil_optimal = seuil_Youden(y,y_proba_oof) # critère 1 : Youden
    df_seuils_recall_0 = seuil_et_recall_depart(y, y_proba_oof)  # totalité des possibilité : essayer de maximiser le recall de la classe 0 (spécificité)
    df_seuils_recall_0.to_excel(OUTPUT_DIR / "XGBoost_recall_classe_0_par_seuil.xlsx", index=False)
    courbe_precision_recall(y, y_proba_oof, seuils=[0.67, seuil_optimal])
    calibration_plot(y, y_proba_oof)
    
    # 4. Modèle final avec consolidation des hyperparamètres modaux
    df_params = pd.DataFrame(meilleurs_params_folds).drop(columns=["Fold", "ROC_AUC_CV_interne"], errors="ignore")
    params_finaux = df_params.mode().iloc[0].to_dict()
    for p in ["XGB_model__n_estimators", "XGB_model__max_depth", "XGB_model__min_child_weight"]:
        if p in params_finaux and pd.notna(params_finaux[p]):
            params_finaux[p] = int(params_finaux[p])

    pipeline_final = construire_pipeline(X, y, num_cols, cat_bool_cols, random_state=RANDOM_STATE)
    pipeline_final.set_params(**params_finaux)
    pipeline_final.fit(X, y)

    # 5. Stabilité SHAP par rééchantillonnage bootstrap OOB
    df_bootstrap_shap, resume_bootstrap_shap = bootstrap_stabilite_shap(
        X=X,
        y=y,
        pipeline_final=pipeline_final,
        n_bootstrap=500,
        top_k=5,
        random_state=RANDOM_STATE
    )
    resume_bootstrap_shap.to_excel(OUTPUT_DIR / "bootstrap_stabilite_shap.xlsx", index=False)
    print("\n✓ Rapport de stabilité SHAP exporté dans 'outputs/modelisation/bootstrap_stabilite_shap.xlsx'.")

    # 6. Explicabilité SHAP
    explainer, shap_values, interaction_values = analyser_shap(
        pipeline_final=pipeline_final,
        X=X,
        nom_organisation="AESIO MUTUELLE",
        top_n=TOP_N_PLOT
    )

    # 7. Prédictions opérationnelles (sans cible et récentes OOF)
    df_predictions_final = generer_predictions_operationnelles(
        df=df,
        pipeline_final=pipeline_final,
        variables_modele=variables_modele,
        cat_bool_cols=cat_bool_cols,
        num_cols=num_cols,
        resultats_complets=resultats_complets,
        seuil=SEUIL_DECISION
    )
    df_predictions_final.to_excel(OUTPUT_DIR / "predictions_entreprises_a_suivre.xlsx", index=False)
    print("\n✓ Livrable de scoring exporté dans 'outputs/modelisation/predictions_entreprises_a_suivre.xlsx'.")