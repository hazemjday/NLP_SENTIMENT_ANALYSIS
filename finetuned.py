#utulisation du modéle fintuné par hugging face
import pandas as pd
from transformers import pipeline
import torch
from sklearn.metrics import confusion_matrix, classification_report

# Vérifier si le GPU est disponible
device = 0 if torch.cuda.is_available() else -1
print(device)
# Chargement du DataFrame
df = pd.read_csv(r"./test.csv", encoding="ISO-8859-1")
columns = ["target", "id", "date", "flag", "user", "text"]

# Pipeline avec GPU si disponible
classifier = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english",
    device=device  
)

texts = df["text"].astype(str).tolist()

# Traitement par batch
batch_size = 128
results = []
for i in range(0, len(texts), batch_size):
    batch = texts[i:i+batch_size]
    preds = classifier(batch)
    results.extend(preds)
    print(f"Traitement : {len(results)}/{len(texts)}")

# Ajouter les résultats dans le DataFrame
df["predicted_label"] = [res["label"] for res in results]



df["target_label"] = df["target"].map({0: "NEGATIVE", 4: "POSITIVE"})
y_true = df["target_label"]
y_pred = df["predicted_label"]
# Rapport d’évaluation
print(classification_report(y_true, y_pred, labels=["NEGATIVE", "POSITIVE"]))

# Matrice de confusion
cm = confusion_matrix(y_true, y_pred, labels=["NEGATIVE", "POSITIVE"])
print("Matrice de confusion (ligne = vérité, colonne = prédiction) :\n")
print(pd.DataFrame(cm,
                   index=["NEGATIVE (vrai)", "POSITIVE (vrai)"],
                   columns=["NEGATIVE (prédit)", "POSITIVE (prédit)"]))


