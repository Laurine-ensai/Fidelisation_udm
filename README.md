# Prédiction de l'Attrition et Facteurs de Fidélisation — Union des Marques

Ce projet applique des méthodes de science des données et d'apprentissage automatique supervisé pour analyser l'engagement des entreprises adhérentes de l'**Union des marques** (UdM) et modéliser le risque de départ précoce.

---

## Contexte & Objectifs

L'Union des marques rassemble un écosystème d'entreprises aux profils variés (Grande consommation, Services, Industrie & Mobilité, Banque/Assurance, Santé...). Même si une part importante des adhérents affiche une fidélité de longue durée, l'association fait face à un renouvellement constant de ses membres et à des départs précoces.

Le projet répond à deux objectifs :
1. **Comprendre (Outil explicatif)** : Identifier les facteurs comportementaux, structurels et relationnels qui distinguent les adhésions pérennes des départs avant 3 ans.
2. **Scorer & Anticiper (Outil d'aide à la décision)** : Repérer les signaux faibles d'attrition afin d'activer des plans de rétention ciblés en amont des échéances annuelles de renouvellement.

---

## Structure du Projet

```text
.
├── data/                                      # Données sources, intermédiaires et référentiels
│   ├── adherent/                              # Données historiques d'adhésion et démission
│   ├── archive_bdd/                           # Exports CRM Procurios (base contacts)
│   ├── dossier_back_up_mailing/               # Logs d'envois, réceptions, ouvertures et clics
│   ├── dossier_back_up_procurios_meetings/    # Logs d'inscriptions et participations aux réunions
│   ├── Dictionnaire_variables_sources.xlsx    # Référentiel des variables sources CRM
│   ├── dictionnaire des nouvelles variables.xlsx
│   └── nv_dictionnaire_variables.xlsx
│
├── notebook/                                  # Pipeline pas-à-pas (Data Prep & EDA)
│   ├── 01_base_membres.ipynb                  # Feature engineering à la maille contact individuel
│   ├── 02_nettoyage_base_membres.ipynb        # Nettoyage, typage strict et harmonisation métier
│   ├── 03_base_adherents.ipynb                # Agrégation à la maille organisation adhérente
│   ├── 04_analyse_univariee_adherents.ipynb   # Distributions statistiques et visualisations descriptives
│   └── 05_analyse_correlations.ipynb          # Étude de colinéarité (Spearman, Cramér, Epsilon², Eta)
│
├── src/                                       # Scripts modulaires et modélisation
│   ├── utils.py                               # Utilitaires de conversion de types et dates
│   ├── brand_charts.py                        # Fonctions de visualisation chartées (style UdM)
│   ├── agregation_membres.py                  # Agrégation et enrichissement des logs contacts
│   ├── agregation_adherents.py                # Consolidation à l'échelle entreprise
│   ├── analyse_univariee.py                   # Fonctions d'analyse univariée automatique
│   ├── analyse_correlations.py                # Matrices d'association multi-types et filtrage
│   ├── regression_logistique_penalisee_lasso.py # Pipeline LASSO + validation croisée + bootstrap
│   └── XGBoost.py                             # Pipeline Gradient Boosting + explicabilité SHAP
│
├── outputs/                                   # Livrables générés automatiquement
│   ├── graphiques/                            # Figures et graphiques chartés
│   ├── modelisation/                          # Fichiers de scoring, matrices OOF et rapports
│   └── rapports/                              # Profils HTML interactifs (skrub TableReport)
│
├── requirements.txt                           # Dépendances Python du projet
└── README.md                                  # Documentation du dépôt
```

---

## Pipeline de Données & Méthodologie

Le pipeline transforme plus de 7,2 millions de logs bruts en une table analytique consolidée de 175 entreprises adhérentes observées :

```
[Mailing & Clics]  +  [Événements & Assiduité]  +  [Profils Contacts CRM]
                                 │
                                 ▼ (src/agregation_membres.py)
                   Base Membres (1 ligne / contact individuel)
                                 │
                                 ▼ (src/agregation_adherents.py)
                   Base Adhérents (1 ligne / entreprise)
                                 │
                                 ▼ (src/analyse_correlations.py)
                   Sélection de 16 variables non-colinéaires
```

---

## Installation & Exécution

### Prérequis
* Python `3.10` ou version supérieure
* Gestionnaire d'environnement (`venv` ou `conda`)

### 1. Installation

```bash
# Cloner le dépôt
git clone <url_du_depot>
cd fidelisation_udm

# Créer et activer l'environnement virtuel
python -m venv .venv
source .venv/bin/activate       # Linux / macOS
# .venv\Scripts\activate        # Windows

# Installer les packages requis
pip install -r requirements.txt
```

### 2. Exécution du Projet

#### étape 1 : Exploration via les Notebooks
Exécuter les notebooks séquentiellement depuis le répertoire `notebook/` :
1. `01_base_membres.ipynb` : Construction de la base contacts enrichie.
2. `02_nettoyage_base_membres.ipynb` : Typage strict, encodage et exclusions.
3. `03_base_adherents.ipynb` : Agrégation au niveau organisation.
4. `04_analyse_univariee_adherents.ipynb` : Exploration descriptive univariée.
5. `05_analyse_correlations.ipynb` : Analyse des associations et sélection de variables.

#### étape 2 : Lancement des Modèles (CLI)
Pour exécuter la validation croisée, les diagnostics et générer les exports :

```bash
# Modèle 1 : Régression Logistique LASSO
python src/regression_logistique_penalisee_lasso.py

# Modèle 2 : XGBoost & Explicabilité SHAP
python src/XGBoost.py
```

Les livrables sont exportés dans `outputs/modelisation/` :
* `df_scored_lasso.xlsx` & `df_scored_xgboost.xlsx` : Tables évaluées en Out-of-Fold avec probabilités de départ et segments de risque.
* `bootstrap_stabilite_shap.xlsx` : Stabilité des variables explicatives sous rééchantillonnage.
* `predictions_entreprises_a_suivre.xlsx` : Table de scoring opérationnel des entreprises récentes à prioriser.
