import pandas as pd
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# 1. Tokenizer standard
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

# 2. Charger le modèle fine-tuné
model = DistilBertForSequenceClassification.from_pretrained("./treatement/best_model")

# 3. Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# 4. Charger et préparer test.csv
df_test = pd.read_csv("./test.csv", encoding="ISO-8859-1")
df_test['text'] = df_test['text'].astype(str)
df_test['label'] = df_test['target'].apply(lambda x: 1 if x == 4 else 0)

# 5. Tokenisation
MAX_LEN = 256
inputs = tokenizer(
    df_test['text'].tolist(),
    padding="max_length",
    truncation=True,
    max_length=MAX_LEN,
    return_tensors="pt"
)

# 6. DataLoader
test_dataset = TensorDataset(
    inputs["input_ids"],
    inputs["attention_mask"],
    torch.tensor(df_test['label'].tolist())
)
test_dataloader = DataLoader(test_dataset, batch_size=32)

# 7. Prédiction
all_preds = []
all_labels = []

with torch.no_grad():
    for batch in tqdm(test_dataloader, desc="Prediction"):
        input_ids, attention_mask, labels = [t.to(device) for t in batch]
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        preds = torch.argmax(logits, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# 8. Ajouter les prédictions au DataFrame
df_test["predicted_label"] = all_preds
df_test["predicted_sentiment"] = df_test["predicted_label"].apply(lambda x: "positive" if x == 1 else "negative")

# 9. Sauvegarder le fichier
df_test.to_csv("test_with_predictions.csv", index=False, encoding="utf-8")
print("Prédictions enregistrées dans test_with_predictions.csv")

# 10. Rapport de classification
print("\n Classification Report :")
print(classification_report(all_labels, all_preds, target_names=["negative", "positive"]))

# 11. Matrice de confusion
cm = confusion_matrix(all_labels, all_preds)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Pred: Neg", "Pred: Pos"], yticklabels=["True: Neg", "True: Pos"])
plt.title("Matrice de Confusion")
plt.xlabel("Prédiction")
plt.ylabel("Vérité terrain")
plt.show()