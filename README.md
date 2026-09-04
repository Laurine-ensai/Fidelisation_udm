# Détermination des facteurs de rétention

## Description

L'objectif de ce projet est de mieux comprendre le profil et les comportements des adhérents les plus fidèles pour identifier les leviers de rétention et anticiper les départs.

## Structure du projet

```text
projet/
├── data/
├── src/
│   ├── utils.py
│   ├── preprocessing.py
│   └── model.py
├── notebooks/
├── requirements.txt
├── README.md
└── main.py
```

## Installation
Cloner le projet :
```bash
git clone ...
cd projet
```

Créer un environnement virtuel :
```bash
python -m venv .venv
```

Activer l'environnement :
Windows :
```bash
.venv\Scripts\activate
```
Mac/Linux :
```bash
source .venv/bin/activate
```

Installer les dépendances :
```bash
pip install -r requirements.txt
```

---
## Utilisation
Dans notebook :

1. Contructions des bases de données :
Lancer 01_base_membres pour obtenir la base membres.
Puis 02_nettoyage_base_membres pour la nettoyer.
Ensuite 03_base_adherents pour obtenir la base adhérents.
Les fichiers : 04_analyse_univariee_adherents et 05_analyse_correlations_adherents sont utilisables une fois que les bases adhérents et membres sont créées.

Dans src :
lancer ... ou ... pour obtenir le modèle XGBoost ou la régression logistique
