#  NLP - Analyse de Sentiments

##  Objectif

Ce projet a pour but de développer un système de **classification des sentiments** à partir de **posts issus des réseaux sociaux**, en utilisant des **modèles NLP** basés sur BERT.

L'objectif est de déterminer si un post est :

- **Positif** : exprime une opinion favorable, de la joie ou de la satisfaction.
- **Négatif** : reflète une critique, une insatisfaction ou une émotion désagréable.

---

##  Dataset

Nous utilisons le dataset **[Sentiment140](https://www.kaggle.com/datasets/kazanova/sentiment140)** :
-  **1,6 million de tweets annotés** selon leur polarité (positif ou négatif)

###  Instructions pour l'installation :

1. Télécharger le dataset depuis Kaggle.
2. Le renommer en `finetuning.csv`.
3. Exécuter le script de préparation des données :
   ```bash
   python datapreparation.py

Ce script effectue les étapes suivantes :

    Nettoyage des tweets (URLs, mentions, hashtags…)

    Division des données en jeu d'entraînement, de validation et de test

 Modèles entraînés
1. distilbert-base-uncased-finetuned-sst-2-english

    Modèle pré-entraîné disponible via HuggingFace, fine-tuné sur SST-2

    Exécution directe sur les données de test :

    python finetuned.py

     Accuracy obtenue : 72%

2.  bert-base-uncased + MLP personnalisé

    Ajout de 3 couches fully-connected après BERT (architecture BertClassifier)

    Pour entraîner le modèle :

python deeptuning.py

Pour évaluer sur les données de test :

    python resultdeeptuning.py

     Accuracy obtenue : 78%

3. distilbert-base-uncased + couche linéaire

    Fine-tuning de DistilBERT avec une couche Linear à 2 sorties

    Meilleure performance globale

    Entraînement :

python finetuning.py

Le modèle est sauvegardé dans le dossier treatment/.

Évaluation :

    python resultfinetuning.py

     Accuracy obtenue : 82%

 Scraping & Analyse Reddit

Pour scraper des publications Reddit et les analyser avec le modèle DistilBERT :

docker compose up -d --build

Cela lance un environnement Docker qui s'occupe de :

    Scraper les posts via l'API Reddit

    Filtrer les contenus en anglais

    Prédire le sentiment de chaque post


    Structure du projet (optionnel)

├── datapreparation.py
├── model_deep_fine/
│ └── best_bert_classifier.pt
├── deeptuning.py
├── resultdeeptuning.py
├── finetuning.py
├── resultfinetuning.py
├── finetuned.py
├── treatment/
│ ├── best_model.pt
│ ├── config.json
│ ├── model.safetensors
│ ├── Dockerfile
│ └── script.py
├── finetuning.csv
├── dockercompose.yml
└── README.md


 Technologies utilisées

    Python 3.10

    PyTorch

    HuggingFace Transformers

    scikit-learn

    pandas, numpy, tqdm

    Docker


## 💡 En cas de problème

Si vous rencontrez des erreurs liées à des fichiers manquants, voici les étapes à suivre :

1. **Créer un dossier `model_deep_fine`** contenant le fichier `best_bert_classifier.pt` :  
   🔗 [Lien Google Drive](https://drive.google.com/drive/u/0/folders/17bf1eKwtQ_8FahoiZuCpoY9tj4WS-8xO)

2. **Dans le dossier `treatment/`, créer un sous-dossier `best_model/`** contenant les fichiers suivants :  
   - `config.json`  
   - `model.safetensors`  
   🔗 [Lien Google Drive](https://drive.google.com/drive/u/0/folders/1O-D2xAY7Azgqc13bISA-0vwDjYi6Oigo)

3. **Télécharger le fichier `finetuning.csv`** (dataset nettoyé) :  
   🔗 [Lien Google Drive](https://drive.google.com/drive/u/0/folders/1RE-AoqLj37QvteUobdxvcBbecaNkn4Tg)