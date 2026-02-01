# Prédiction de la consommation électrique journalière domestique

## Description

Ce dépôt contient le projet de prédiction de la consommation électrique journalière (kWh) d'un logement à Sceaux, France. Les données proviennent du jeu UCI « Individual Household Electric Power Consumption » et de l'API météo Open-Meteo. L'objectif est de construire un modèle de régression exploitant la météo et le calendrier (jours fériés, week-end) pour expliquer et prédire la consommation journalière.

Le pipeline comprend la collecte et la fusion des données (consommation, météo, jours fériés), l'ingénierie des variables (DJU, lag-1, encodage cyclique du mois), le filtrage des jours atypiques, la sélection des variables par importance (Random Forest), et la comparaison de trois modèles : Random Forest, Gradient Boosting et régression linéaire. Le modèle retenu (Random Forest) est utilisé pour une prédiction sur sept jours.

## Résultats

- **Modèle retenu :** Random Forest (variables sélectionnées par importance).
- **Métriques sur le jeu de test (découpage 75 % / 25 %, ordre chronologique) :**
  - R² : 0,66
  - RMSE : 4,1 kWh/jour
  - MAE : 3,18 kWh/jour
- **Comparaison :** Gradient Boosting (R² = 0,661), régression linéaire (R² = 0,534). Le Random Forest offre les meilleures performances.
- **Prédiction :** prévision sur une semaine ; marge d'erreur journalière d'environ 4,1 kWh (équivalent à environ 0,62 €/jour pour un tarif de 0,15 €/kWh).

## Contenu du dépôt

| Élément | Description |
|--------|-------------|
| **Rapport** | Document rédigé au format IEEE (conference) : méthodologie, résultats, limites et recommandations. Fichier : `rapport_consommation_electrique.pdf` (ou `.tex` à compiler). |
| **Vidéo de présentation** | Enregistrement de la soutenance ou présentation du projet. Fichier : `presentation_projet.mp4` (adapter le nom si nécessaire). |
| **Code** | Scripts Python (collecte, prétraitement, entraînement, évaluation, graphiques) et éventuels notebooks. |

## Rapport IEEE

Le rapport respecte la norme IEEE pour les articles de conférence (classe `IEEEtran`, deux colonnes, sections numérotées en chiffres romains, références bibitem). Il est fourni au format LaTeX (`.tex`) ; compiler avec pdflatex pour obtenir le PDF.

## Vidéo de présentation

Le fichier vidéo de présentation du projet est disponible à la racine du dépôt ou dans le dossier indiqué. Il présente la problématique, la méthodologie et les résultats.

## Auteurs

[Indiquer les noms et affiliations.]

## Licence

[Indiquer la licence du projet, par exemple MIT, CC-BY-NC, ou usage académique.]
