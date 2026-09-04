"""
analyse_univariee.py
====================
Outils d'exploration et de visualisation univariée automatique pour Pandas.
Détecte automatiquement la nature de la variable (numérique continue vs 
qualitative / discrète) pour afficher les statistiques adaptées et générer 
les graphiques associés.

Fonctionnalités :
- Détection automatique : aiguillage selon le type et le nombre de valeurs uniques (seuil : > 10)
- Variables qualitatives / discrètes :
    * Tableaux textuels des effectifs et des proportions (incluant NaN)
    * Barres horizontales de proportions
    * Barre 100% empilée (composition globale)
    * Camembert / Donut (gestion automatique de la légende selon la cardinalité)
- Variables numériques continues :
    * Résumé statistique complet (moyenne, écart-type, quartiles...)
    * Histogramme avec courbe d'estimation de densité (KDE)
    * Boîte à moustaches (Boxplot) pour identifier les valeurs aberrantes (outliers)

Installation :
    pip install pandas matplotlib seaborn

Usage :
    import pandas as pd
    from analyse_univariee import analyse_univariee

    df = pd.read_csv("mes_donnees.csv")
    analyse_univariee(df, "colonne_a_analyser")
"""


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def analyse_univariee(df, variable):
    """
    Analyse univariée d'une variable.
    """
    print(f"\n===== Analyse univariée de {variable} =====\n")
    
    if pd.api.types.is_numeric_dtype(df[variable]) and df[variable].nunique() > 10:
        # Variable numérique continue
        analyser_numerique(df, variable)
    else:
        # Variable qualitative ou numérique discrète
        analyser_qualitative(df, variable)
        

def analyser_qualitative(data, variable):
    # 1. Préparation des données
    effectifs = data[variable].value_counts(dropna=False)
    proportions = data[variable].value_counts(dropna=False, normalize=True).round(4)
    nb_modalites = len(proportions)

    print("Effectifs :")
    print(effectifs)
    print("\nProportions :")
    print(proportions)
    print("\n ")
    
    # Taille dynamique : on augmente la hauteur totale
    hauteur_par_graph = max(4, nb_modalites * 0.4)
    sns.set_theme(style="whitegrid")
    palette = sns.color_palette("viridis", n_colors=nb_modalites)

    # Création d'une figure avec 3 sous-graphiques
    fig, axes = plt.subplots(3, 1, figsize=(12, hauteur_par_graph * 3))
    fig.suptitle(f'Analyses de la variable : {variable}', fontsize=18, fontweight='bold', y=1.02)

    # --- 1. Barres Horizontales ---
    sns.barplot(x=proportions.values, y=proportions.index.astype(str), ax=axes[0], palette=palette)
    axes[0].set_title('I. Répartition détaillée (Proportions)', pad=20, fontsize=14)
    axes[0].set_xlabel('Proportion')

    # --- 2. Barre empilée ---
    # Pour un "empilé", on crée un DataFrame d'une seule ligne
    df_stacked = proportions.to_frame().T
    df_stacked.plot(kind='barh', stacked=True, ax=axes[1], color=palette, edgecolor='white')
    axes[1].set_title('II. Composition globale (Cumul)', pad=20, fontsize=14)
    axes[1].set_yticks([])
    axes[1].legend(loc='center left', bbox_to_anchor=(1, 0.5), title=variable, ncol=1 if nb_modalites < 12 else 2)

    # --- 3. Secteurs ---
    # On ajoute un peu de "vide" autour du camembert pour éviter qu'il touche le titre du dessus
    axes[2].pie(proportions, labels=(proportions.index if nb_modalites < 6 else None), 
                autopct='%1.1f%%', startangle=140, colors=palette, pctdistance=0.85)
    axes[2].set_title('III. Distribution relative', pad=20, fontsize=14)
    if nb_modalites >= 6:
        axes[2].legend(proportions.index, loc='center left', bbox_to_anchor=(1, 0.5))

    # RÉGLAGE CRUCIAL DE L'ESPACEMENT
    # hspace définit l'espace vertical entre les subplots
    plt.subplots_adjust(hspace=0.6, top=0.92, bottom=0.05, right=0.8)
    
    plt.show()


def analyser_numerique(df, variable):
    # 1. Résumé statistique
    print(f"--- Résumé statistique : {variable} ---")
    print(df[variable].describe().round(2))
    print("\n")

    # Configuration du style
    sns.set_theme(style="whitegrid")
    
    # Création d'une figure avec 2 sous-graphiques (1 ligne, 2 colonnes)
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), gridspec_kw={'width_ratios': [3, 1]})
    fig.suptitle(f'Analyse de la distribution : {variable}', fontsize=16)

    # --- 1. Histogramme avec courbe de densité (KDE) ---
    sns.histplot(df[variable], kde=True, ax=axes[0], color='skyblue', edgecolor='black')
    axes[0].set_title('Histogramme & Densité')
    axes[0].set_ylabel('Fréquence')

    # --- 2. Boxplot (Boîte à moustaches) ---
    sns.boxplot(y=df[variable], ax=axes[1], color='lightgreen')
    axes[1].set_title('Boxplot (Détection des outliers)')
    axes[1].set_ylabel(variable)

    # Ajustement de l'espacement
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()