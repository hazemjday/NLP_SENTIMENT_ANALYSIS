from pymongo import MongoClient
import pandas as pd
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix
import re
import praw
from langdetect import detect


def clean_text(text):
    text = re.sub(r"http\S+", "", text)      # Supprime les URLs
    text = re.sub(r"@\w+", "", text)         # Supprime les @mentions
    text = re.sub(r"#\w+", "", text)         # Supprime les #hashtags
    text = re.sub(r"\s+", " ", text)         # Supprime les espaces multiples
    return text.strip()

reddit = praw.Reddit(
    client_id="jEWljAMSq6zjNHphH6vWQQ",
    client_secret="tKdWnN1q8YFKQCSMAzgHhskqkfaPvg",
    user_agent="windows:hazem/hazem/project"
)    


subreddit = reddit.subreddit("relationships")
top_posts = subreddit.new(limit=1000)


filtered_posts = []
for post in top_posts:
    if post.selftext and post.selftext.strip():
        try:
            if detect(post.selftext) != "en":
                continue  
        except:
            continue 
  
    if post.selftext and post.selftext.strip():
        post_text = post.selftext.strip().lower()
        word_count = len(post.selftext.strip().split())    
        if word_count < 150:
            filtered_posts.append({
                "text": post.selftext.strip()
           
                })

df = pd.DataFrame(filtered_posts)
df["text"] = df["text"].astype(str).apply(clean_text)



# 2. Tokenizer standard
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

# 3. Charger le modèle fine-tuné
model = DistilBertForSequenceClassification.from_pretrained("./best_model")

# 4. Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# 5. Tokenisation
MAX_LEN = 256
inputs = tokenizer(
    df['text'].tolist(),
    padding="max_length",
    truncation=True,
    max_length=MAX_LEN,
    return_tensors="pt"
)

# 6. DataLoader
test_dataset = TensorDataset(
    inputs["input_ids"],
    inputs["attention_mask"]
)
test_dataloader = DataLoader(test_dataset, batch_size=32)

# 7. Prédiction
all_preds = []

with torch.no_grad():
    for batch in tqdm(test_dataloader, desc="Prediction"):
        input_ids, attention_mask = [t.to(device) for t in batch]
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        preds = torch.argmax(logits, dim=1)
        all_preds.extend(preds.cpu().numpy())



total = len(all_preds)
positifs = sum(1 for p in all_preds if p == 1)
negatifs = sum(1 for p in all_preds if p == 0)

print(f"Nombre total de posts : {total}")
print(f"Nombre de commentaires positifs (label=1) : {positifs}")
print(f"Nombre de commentaires négatifs (label=0) : {negatifs}")







