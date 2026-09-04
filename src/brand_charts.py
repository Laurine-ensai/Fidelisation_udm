"""
brand_charts.py
================
Fonctions matplotlib prêtes à l'emploi pour refaire les graphiques dans le style de la présentation "Union des Marques" (couleurs, dégradés, typographie).

Palette :
- Violet foncé (titres)        : #1e1b4b
- Violet (titres secondaires)  : #4c1d95
- Violet medium (accent/liens) : #533fe4
- Orange/corail (accent)       : #ff5a36 / #e6563c
- Gris texte body              : #334155
- Gris clair fond de carte     : #f8fafc

Chaque fonction exporte un PNG en fond transparent, prêt à être glissé
dans Canva.

Installation :
    pip install matplotlib numpy pandas

Usage : voir le bloc "EXEMPLES" tout en bas du fichier — remplacez les
données d'exemple par vos vraies données.
"""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np

OUTPUT_DIR = Path("outputs")
PLOTS_DIR    = OUTPUT_DIR / "graphiques"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)    # crée le dossier s'il n'existe pas

# ----------------------------------------------------------------------
# 1. STYLE DE L'UNION DES MARQUES
# ----------------------------------------------------------------------
BRAND = {
    "navy": "#1e1b4b",       # titres principaux
    "purple_dark": "#4c1d95",  # sous-titres / accents violets
    "purple": "#533fe4",     # violet médian (médiane, liens, accents)
    "orange": "#ff5a36",     # accent chaud (moyenne, alertes)
    "orange_dark": "#e6563c",
    "gray_text": "#334155",  # texte de corps
    "gray_light": "#94a3b8", # texte secondaire / axes
    "card_bg": "#f8fafc",    # fond de carte clair
    "grid": "#e2e8f0",
}

# Dégradé signature orange -> violet utilisé sur les barres "top/flop"
BRAND_GRADIENT = LinearSegmentedColormap.from_list(
    "brand_gradient", [BRAND["orange"], BRAND["purple"]]
)


def setup_brand_style():
    """Configure les rcParams matplotlib pour coller au style de la deck."""
    # Police : remplacez par une police géométrique proche de celle de Canva
    # (ex. "Poppins", "Montserrat", "Manrope") si vous l'avez installée.
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Poppins", "Montserrat", "DejaVu Sans"],
        "axes.edgecolor": BRAND["grid"],
        "axes.labelcolor": BRAND["gray_text"],
        "text.color": BRAND["navy"],
        "xtick.color": BRAND["gray_text"],
        "ytick.color": BRAND["gray_text"],
        "figure.facecolor": "none",
        "axes.facecolor": "none",
        "savefig.facecolor": "none",
    })


def _save(fig, filename, dpi=300):
    fig.savefig(filename, dpi=dpi, transparent=True, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    print(f"✔ Graphique enregistré : {filename}")


# ----------------------------------------------------------------------
# 2. BARRES HORIZONTALES DÉGRADÉES
# ----------------------------------------------------------------------

def gradient_hbar(categories, values, filename, value_suffix="",
                   highlight_label=None, highlight_color=None,
                   figsize=(11, 5)):
    """
    Barres horizontales avec dégradé orange -> violet, valeurs affichées
    au bout de chaque barre. Idéal pour les classements (ex: contacts
    par entreprise, taux de bounce par entreprise...).

    categories : liste de labels (ex noms d'entreprises), la plus grande
                 valeur en premier
    values     : liste de valeurs numériques (même ordre)
    highlight_label : optionnel, un label à faire ressortir en solide
                       (ex: "Moyenne Globale") au lieu du dégradé
    """
    setup_brand_style()
    fig, ax = plt.subplots(figsize=figsize)

    n = len(categories)
    y_pos = np.arange(n)[::-1]  # premier en haut
    vmax = max(values)

    if highlight_label:
        if isinstance(highlight_label, str):
            highlight_labels = [highlight_label]
        else:
            highlight_labels = highlight_label
    else:
        highlight_labels = []

    for y, cat, val in zip(y_pos, categories, values):
        if cat in highlight_labels:
            ax.barh(y, val, height=0.62, color=highlight_color or BRAND["purple"],
                     zorder=3)
        else:
            # barre en dégradé : on dessine une image clip-ée à la forme de la barre
            grad = np.linspace(0, 1, 256).reshape(1, -1)
            ax.imshow(grad, extent=[0, val, y - 0.31, y + 0.31], aspect="auto",
                       cmap=BRAND_GRADIENT, zorder=2)
        ax.text(val + vmax * 0.015, y, f"{val:,}{value_suffix}".replace(",", " "),
                va="center", ha="left", fontsize=12, fontweight="bold",
                color=BRAND["navy"])

    ax.set_xlim(0, vmax * 1.18)
    ax.set_ylim(-0.5, n - 0.5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(categories, fontsize=12, color=BRAND["navy"], fontweight="bold")
    ax.set_xticks([])
    for spine in ["top", "right", "bottom", "left"]:
        ax.spines[spine].set_visible(False)
    ax.tick_params(length=0)
    _save(fig, filename)


# ----------------------------------------------------------------------
# 3. CAMEMBERT / DONUT 
# ----------------------------------------------------------------------

def branded_donut(labels, values, filename, colors=None, donut=True,
                   figsize=(6, 6), percent_position="outside"):
    """
    Camembert (ou donut) avec la palette de marque. 
    
    Parameters
    ----------
    labels : list
        Labels des différentes catégories.
    values : list
        Valeurs associées aux catégories.
    filename : str
        Chemin du fichier de sortie.
    colors : list, optional
        Couleurs personnalisées. Si None, utilise la palette de marque.
    donut : bool, default=True
        True pour un donut, False pour un camembert.
    figsize : tuple, default=(8, 8)
        Taille de la figure.
    percent_position : {"inside", "outside"}, default="outside"
        Position des pourcentages.
    """
    setup_brand_style()
    if colors is None:
        palette = [BRAND["purple"], BRAND["purple_dark"], BRAND["orange"], BRAND["gray_light"]]
        colors = [palette[i % len(palette)] for i in range(len(values))]

    if percent_position == "outside":
        pctdistance = 1.12
        labeldistance = 1.28
        percent_color = BRAND["navy"]
    elif percent_position == "inside":
        pctdistance = 0.8 if donut else 0.6
        labeldistance = 1.1
        percent_color = "white"
    else:
        raise ValueError(
            "percent_position doit être 'inside' ou 'outside'."
        )

    fig, ax = plt.subplots(figsize=figsize)
    wedges, texts, autotexts = ax.pie(
        values, labels=labels, colors=colors, autopct="%1.1f%%",
        startangle=90, pctdistance=pctdistance, labeldistance=labeldistance,
        wedgeprops=dict(width=0.42 if donut else 1, edgecolor="white", linewidth=2),
        textprops=dict(color=BRAND["navy"], fontsize=12, fontweight="bold"),
    )
    for a in autotexts:
        a.set_color(percent_color)
        a.set_fontsize(10)
        a.set_fontweight("bold")
    ax.set_aspect("equal")
    _save(fig, filename)


# ----------------------------------------------------------------------
# 4. BARRES EMPILÉES 100% 
# ----------------------------------------------------------------------

def stacked_percent_bar(categories, series_dict, filename, figsize=(9, 5.5)):
    """
    Barres empilées à 100%, une barre par catégorie.

    categories : ex ["C", "M", "F"]
    series_dict : dict ordonné {nom_série: [valeurs % par catégorie]}
                  ex {"Présent": [93,66,30], "Absent": [1,27,64], "Manquant":[6,7,6]}
    """
    setup_brand_style()
    colors = [BRAND["purple_dark"], BRAND["purple"], BRAND["orange"]]  # présent / absent / manquant
    fig, ax = plt.subplots(figsize=figsize)

    bottoms = np.zeros(len(categories))
    for i, (name, vals) in enumerate(series_dict.items()):
        vals = np.array(vals, dtype=float)
        bars = ax.bar(categories, vals, bottom=bottoms, label=name,
                       color=colors[i % len(colors)], width=0.55)
        for b, v in zip(bars, vals):
            if v > 4:  # évite les labels illisibles sur de trop petites tranches
                ax.text(b.get_x() + b.get_width() / 2, b.get_y() + v / 2,
                         f"{v:.0f}%", ha="center", va="center",
                         color="white", fontsize=11, fontweight="bold")
        bottoms += vals

    ax.set_ylim(0, 100)
    ax.set_yticks([])
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="x", length=0, labelsize=13, colors=BRAND["navy"])
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=3,
              frameon=False, fontsize=11)
    _save(fig, filename)


# ----------------------------------------------------------------------
# 5. HISTOGRAMME + DENSITÉ
# ----------------------------------------------------------------------

def branded_hist(data, filename, xlabel="", bins=20, mean=None, median=None,
                  color=None, figsize=(7.5, 5)):
    """
    Histogramme avec courbe de densité (KDE), dans le style de la deck
    (bleu/violet, pas de fioritures type seaborn par défaut).

    data   : array-like des valeurs brutes
    mean, median : si fournis, tracent des lignes verticales annotées
    """
    setup_brand_style()
    from scipy.stats import gaussian_kde  # pip install scipy si besoin

    color = color or BRAND["purple"]
    data = np.asarray(data)
    fig, ax = plt.subplots(figsize=figsize)

    ax.hist(data, bins=bins, color=color, alpha=0.35, edgecolor=color, linewidth=1.2)

    # KDE mise à l'échelle sur le même axe que l'histogramme
    kde = gaussian_kde(data)
    xs = np.linspace(data.min(), data.max(), 300)
    ys = kde(xs)
    bin_width = (data.max() - data.min()) / bins
    ax.plot(xs, ys * len(data) * bin_width, color=color, linewidth=2.5)

    if mean is not None:
        ax.axvline(mean, color=BRAND["orange"], linestyle="--", linewidth=2)
    if median is not None:
        ax.axvline(median, color=BRAND["navy"], linestyle="--", linewidth=2)

    ax.set_xlabel(xlabel, fontsize=11, color=BRAND["gray_text"])
    ax.set_yticks([])
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="x", colors=BRAND["gray_text"])
    _save(fig, filename)


def branded_hist_pair(data_left, data_right, filename,
                       xlabel_left="", xlabel_right="",
                       figsize=(14, 5)):
    """Deux histogrammes côte à côte (style p.26 et p.34)."""
    setup_brand_style()
    from scipy.stats import gaussian_kde

    fig, axes = plt.subplots(1, 2, figsize=figsize)
    for ax, data, xlabel, color in zip(
        axes, [data_left, data_right], [xlabel_left, xlabel_right],
        [BRAND["purple"], BRAND["orange"]]
    ):
        data = np.asarray(data)
        ax.hist(data, bins=20, color=color, alpha=0.35, edgecolor=color, linewidth=1.2)
        kde = gaussian_kde(data)
        xs = np.linspace(data.min(), data.max(), 300)
        bin_width = (data.max() - data.min()) / 20
        ax.plot(xs, kde(xs) * len(data) * bin_width, color=color, linewidth=2.5)
        ax.set_xlabel(xlabel, fontsize=11, color=BRAND["gray_text"])
        ax.set_yticks([])
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.tick_params(axis="x", colors=BRAND["gray_text"])
    _save(fig, filename)


# ----------------------------------------------------------------------
# 6. TABLEAU STYLISÉ
# ----------------------------------------------------------------------

def branded_table(headers, rows, filename, col_widths=None, figsize=(11, None)):
    """
    Tableau propre avec en-tête violet, pour remplacer les captures
    d'écran de DataFrame (p.15 notamment).

    headers : liste de noms de colonnes
    rows    : liste de listes (une par ligne)
    """
    setup_brand_style()
    n_rows = len(rows) + 1
    height = figsize[1] or (0.7 * n_rows)
    fig, ax = plt.subplots(figsize=(figsize[0], height))
    ax.axis("off")

    table = ax.table(cellText=rows, colLabels=headers, loc="center",
                      cellLoc="center", colWidths=col_widths)
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 2.2)

    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor(BRAND["grid"])
        if row == 0:
            cell.set_facecolor(BRAND["purple"])
            cell.set_text_props(color="white", fontweight="bold")
        else:
            cell.set_facecolor("white" if row % 2 else BRAND["card_bg"])
            cell.set_text_props(color=BRAND["gray_text"])
    _save(fig, filename, dpi=250)


# ----------------------------------------------------------------------
# EXEMPLES — à adapter avec vos vraies données
# ----------------------------------------------------------------------

if __name__ == "__main__":
    gradient_hbar(
        categories=["RUGBY WORLD CUP FRANCE 2023", "PHYTEUROPE", "Moyenne", "Médiane","GROUPE COMPAGNIE DES ALPES","UNICEF FRANCE"],
        values=[293, 198, 66, 57, 19, 4],
        value_suffix=" emails",
        highlight_label=["Moyenne", "Médiane"],
        highlight_color=BRAND["purple"],
        filename=PLOTS_DIR / "p21_top_flop_mails.png",
    )
    
    branded_donut(
        labels=["C-Level", "F-Level", "M-Level", "Non-renseigné"],
        values=[21.2, 1.5, 10.7, 66.6],
        colors=[BRAND["navy"],"#4a7ff7", "#f59e0b",BRAND["orange"]],
        filename=PLOTS_DIR / "p16_pie_contacts.png",
    )

    stacked_percent_bar(
        categories=["C", "M", "F"],
        series_dict={
            "Présent": [93.8, 68.4, 30.5],
            "Absent": [1.1, 26.6, 64.4],
            "Manquant": [5.1, 5.1, 5.1],
        },
        filename=PLOTS_DIR / "p16_stacked_entreprises.png",
    )

    gradient_hbar(
        categories=["Marketing", "Juridique", "Performance digitale", "Médias",
                    "Direction générale", "Insights", "Impact", "Affaires publiques","RH"],
        values=[96, 66.5, 65.3, 57.8, 54.9, 51.4, 41.6, 37, 31.2],
        value_suffix=" %",
        filename=PLOTS_DIR / "p18_departements.png",
    )

    rng = np.random.default_rng(0)
    branded_hist_pair(
        data_left=rng.gamma(3, 70, 400),   # ~ avg_days_since_last_open
        data_right=rng.exponential(60, 400),  # ~ days_since_most_recent_open
        xlabel_left="avg_days_since_last_open",
        xlabel_right="days_since_most_recent_open",
        filename=PLOTS_DIR / "p26_recence.png",
    )

    branded_hist(
        data=rng.beta(2, 3, 400) * 100,
        xlabel="avg_presence_rate_by_relation",
        mean=30,
        filename=PLOTS_DIR / "p32_presence.png",
    )

    branded_table(
        headers=["Entreprise", "Jours écoulés (moy.)", "Jours écoulés (récent)"],
        rows=[
            ["KERING", "1558", "1457"],
            ["BPIFRANCE", "1349", "319"],
            ["CREDIT AGRICOLE", "1286", "66"],
            ["MATTEL FRANCE", "1165", "82"],
            ["TERRES DE COMMUNICATION", "1137", "684"],
        ],
        filename=PLOTS_DIR / "p35_table.png",
    )

    print("\nTous les graphiques d'exemple ont été générés dans " + str(PLOTS_DIR) + "/")
