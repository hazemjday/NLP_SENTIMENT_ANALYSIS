# NLP - Analyse de Sentiments

## Objectif

Développer un système de **classification des sentiments** à partir de **posts issus des réseaux sociaux**, en utilisant des **modèles NLP** basés sur BERT.

L'objectif est de déterminer si un post est :
- **Positif** : opinion favorable, joie ou satisfaction.
- **Négatif** : critique, insatisfaction ou émotion désagréable.

---

## Dataset

Nous utilisons le dataset **[Sentiment140](https://www.kaggle.com/datasets/kazanova/sentiment140)** :
- **1,6 million de tweets annotés** selon leur polarité (positif ou négatif)

### Installation des données

1. Télécharger le dataset depuis Kaggle.
2. Le renommer en `finetuning.csv`.
3. Exécuter le script de préparation des données :

   ```bash
   python datapreparation.py
   ```

Ce script effectue :
- Nettoyage des tweets (URLs, mentions, hashtags…)
- Division des données en jeu d'entraînement, de validation et de test

---

## Modèles entraînés

### 1. `distilbert-base-uncased-finetuned-sst-2-english`
- Modèle pré-entraîné HuggingFace, fine-tuné sur SST-2
- Exécution sur les données de test :

  ```bash
  python finetuned.py
  ```
- **Accuracy obtenue : 72%**

### 2. `bert-base-uncased` + MLP personnalisé
- Ajout de 3 couches fully-connected après BERT (`BertClassifier`)
- Pour entraîner le modèle :

  ```bash
  python deeptuning.py
  ```
- Pour évaluer sur les données de test :

  ```bash
  python resultdeeptuning.py
  ```
- **Accuracy obtenue : 78%**

### 3. `distilbert-base-uncased` + couche linéaire
- Fine-tuning de DistilBERT avec une couche Linear à 2 sorties
- Meilleure performance globale
- Entraînement :

  ```bash
  python finetuning.py
  ```
- Le modèle est sauvegardé dans le dossier `treatment/`.
- Évaluation :

  ```bash
  python resultfinetuning.py
  ```
- **Accuracy obtenue : 82%**

---

## Scraping & Analyse Reddit

Pour scraper des publications Reddit et les analyser avec le modèle DistilBERT :

```bash
docker compose up -d --build
```

Cela lance un environnement Docker qui :
- Scrape les posts via l'API Reddit
- Filtre les contenus en anglais
- Prédit le sentiment de chaque post

---

## Structure du projet

```text
├── datapreparation.py
├── model_deep_fine/
│   └── best_bert_classifier.pt
├── deeptuning.py
├── resultdeeptuning.py
├── finetuning.py
├── resultfinetuning.py
├── finetuned.py
├── treatment/
│   ├── best_model.pt
│   ├── config.json
│   ├── model.safetensors
│   ├── Dockerfile
│   └── script.py
├── finetuning.csv
├── docker-compose.yml
└── README.md
```

---

## Technologies utilisées

- Python 3.10
- PyTorch
- HuggingFace Transformers
- scikit-learn
- pandas, numpy, tqdm
- Docker

---

## 💡 Dépannage : Fichiers manquants

Si vous rencontrez des erreurs liées à des fichiers manquants, suivez ces étapes :

1. **Créer un dossier `model_deep_fine`** contenant le fichier `best_bert_classifier.pt` :  
   [Lien Google Drive](https://drive.google.com/drive/u/0/folders/17bf1eKwtQ_8FahoiZuCpoY9tj4WS-8xO)

2. **Dans le dossier `treatment/`, créer un sous-dossier `best_model/`** contenant :
   - `config.json`
   - `model.safetensors`
   [Lien Google Drive](https://drive.google.com/drive/u/0/folders/1O-D2xAY7Azgqc13bISA-0vwDjYi6Oigo)

3. **Télécharger le fichier `finetuning.csv`** (dataset nettoyé) :  
   [Lien Google Drive](https://drive.google.com/drive/u/0/folders/1RE-AoqLj37QvteUobdxvcBbecaNkn4Tg)