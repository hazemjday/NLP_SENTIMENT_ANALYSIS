#preparer la base de donnee qui sera utuliser pour le finetuning du projet

import pandas as pd
import re
from sklearn.model_selection import train_test_split
from pandas import concat

def clean_text(text):
    text = re.sub(r"http\S+", "", text)      # Supprime les URLs
    text = re.sub(r"@\w+", "", text)         # Supprime les @mentions
    text = re.sub(r"#\w+", "", text)         # Supprime les #hashtags
    text = re.sub(r"\s+", " ", text)         # Supprime les espaces multiples
    return text.strip()


columns = ["target", "id", "date", "flag", "user", "text"]
# Chargement du DataFrame
df = pd.read_csv("C:/Users/Lenovo/Downloads/archive/finetuning.csv", encoding="ISO-8859-1", header=None, names=columns)
df = df.drop(columns=["id", "date", "flag", "user"])
df["text"] = df["text"].astype(str).apply(clean_text)

df_negatifs = df[df["target"] == 0]
df_positifs = df[df["target"] == 4]

df_negatifs = df_negatifs.sample(n=50000, random_state=42)
df_positifs = df_positifs.sample(n=50000, random_state=42)

# 1. Découper les négatifs
neg_train_val, neg_test = train_test_split(df_negatifs, test_size=0.2, random_state=42)
neg_train, neg_val = train_test_split(neg_train_val, test_size=0.1, random_state=42)  # 0.125 de 80% = 10%

# 2. Découper les positifs
pos_train_val, pos_test = train_test_split(df_positifs, test_size=0.2, random_state=42)
pos_train, pos_val = train_test_split(pos_train_val, test_size=0.1, random_state=42)

# 3. Combiner
train_df = concat([neg_train, pos_train]).sample(frac=1, random_state=42).reset_index(drop=True)
val_df = concat([neg_val, pos_val]).sample(frac=1, random_state=42).reset_index(drop=True)
test_df = concat([neg_test, pos_test]).sample(frac=1, random_state=42).reset_index(drop=True)


# 4. Vérification
print("Taille train :", len(train_df))
print("Taille validation :", len(val_df))
print("Taille test :", len(test_df))
print("Répartition train :\n", train_df["target"].value_counts())
print("Répartition validation :\n", val_df["target"].value_counts())
print("Répartition test :\n", test_df["target"].value_counts())

train_df.to_csv("train.csv", index=False, encoding="utf-8")
val_df.to_csv("validation.csv", index=False, encoding="utf-8")
test_df.to_csv("test.csv", index=False, encoding="utf-8")