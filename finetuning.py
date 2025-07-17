#utulisation de distilbirt pour l'entrainement
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import DistilBertTokenizer
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import DistilBertForSequenceClassification
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm
import numpy as np
import torch.nn as nn



df_train = pd.read_csv("./train.csv", encoding="ISO-8859-1")
df_val = pd.read_csv("./validation.csv", encoding="ISO-8859-1")
df_test = pd.read_csv("./test.csv", encoding="ISO-8859-1")
df_train['label'] = df_train['target'].apply(lambda x: 1 if x == 4 else 0)
df_val['label'] = df_val['target'].apply(lambda x: 1 if x == 4 else 0)
df_test['label'] = df_test['target'].apply(lambda x: 1 if x == 4 else 0)
df_train['text'] = df_train['text'].astype(str)
df_val['text'] = df_val['text'].astype(str)
df_test['text'] = df_test['text'].astype(str)

print (df_train.head(2))
print (df_test.head(2))

tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased", do_lower_case=True)
MAX_LEN = 256

def encode_texts(texts, labels):
    inputs = tokenizer(
        texts.tolist(),
        #remplir jusqua max_length
        padding="max_length",
        #si depasse max_length
        truncation=True,
        max_length=MAX_LEN,
        # resultat sont convertit en tensors 
        return_tensors="pt"
    )
    #vectorisation des donne pour etre efficace etre efficace avec le modele
    return {
        "input_ids": inputs["input_ids"],
        "attention_mask": inputs["attention_mask"],
        "labels": torch.tensor(labels.tolist())  # Labels en tensor
    }

# Encodage des données
train_encodings = encode_texts(df_train['text'] , df_train['label'])
val_encodings = encode_texts(df_val['text'], df_val['label'])
test_encodings = encode_texts(df_test['text'], df_test['label'])


batch_size = 32

#datasets a partirs des tensors
train_dataset = TensorDataset(
    train_encodings["input_ids"],
    train_encodings["attention_mask"],
    train_encodings["labels"]
)
val_dataset = TensorDataset(
    val_encodings["input_ids"],
    val_encodings["attention_mask"],
    val_encodings["labels"]
)

train_dataloader = DataLoader(
    train_dataset,
    sampler=RandomSampler(train_dataset),
    batch_size=batch_size
)
val_dataloader = DataLoader(
    val_dataset,
    sampler=SequentialSampler(val_dataset),
    batch_size=batch_size
)

model = DistilBertForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",  # Version de base (sans fine-tuning SST-2)
    num_labels=2,               # Nouvelle tête pour 2 classes (0/1)
    output_attentions=False,
    output_hidden_states=False
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
model.to(device)

#gestion des weights optimiser pour bert
optimizer = AdamW(model.parameters(), lr=2e-5, eps=1e-8)
epochs = 3
total_steps = len(train_dataloader) * epochs
#decroissance du learning rate au cours du temps
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=0,
    num_training_steps=total_steps
)

best_val_loss = float("inf")


def evaluate(model, dataloader):
    #modification du modéle
    model.eval()
    val_loss = 0
    val_preds = []
    val_labels = []
    #desactive le calcul du gradiant
    with torch.no_grad():
        for batch in dataloader:
            batch = tuple(t.to(device) for t in batch)
            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "labels": batch[2]
            }
            #calcule des prediction
            outputs = model(**inputs)
            #calcule des loss
            loss = outputs.loss
            val_loss += loss.item()
            logits = outputs.logits.detach().cpu().numpy()
            #logits est le score atteint pour 0 et pour 1
            label_ids = inputs["labels"].detach().cpu().numpy()
            val_preds.extend(np.argmax(logits, axis=1))
            val_labels.extend(label_ids)
    #moyenne de perte pour chaque item
    avg_val_loss = val_loss / len(dataloader)
    #calcule combien de prediction correcte
    accuracy = np.mean(np.array(val_preds) == np.array(val_labels))
    return avg_val_loss, accuracy




#entrainement du modele
#les epochs calcule les meme poids
for epoch in range(epochs):
    #actib=ve le modele de entrainement
    model.train()
    #perte totale dune epoche
    total_loss = 0
    for batch in tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{epochs}"):
        #transferer les torseur sur gpu
        batch = tuple(t.to(device) for t in batch)
       #structuration des donnees pour le modele
        inputs = {
            "input_ids": batch[0],
            "attention_mask": batch[1],
            "labels": batch[2]
        }
        #eviter de accumuler les gradiants 
        optimizer.zero_grad()
        #calculer les loss du output
        outputs = model(**inputs)
        loss = outputs.loss
        #loss totale du batch
        total_loss += loss.item()
        #calcul des gradiants 
        loss.backward()
        #L2 doit etre inferieur 1.0 pouur eviter exploding des gradiants
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
       # met a jour le gradiant 
        optimizer.step()
     # mettre a jour le learning rate
        scheduler.step()
    avg_train_loss = total_loss / len(train_dataloader)
    print(f"Train loss: {avg_train_loss:.4f}")
    val_loss, val_accuracy = evaluate(model, val_dataloader) 
    print(f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}, Val Loss = {val_loss:.4f}")
    # Sauvegarde du meilleur modèle
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        model.save_pretrained("./treatement/best_model")
