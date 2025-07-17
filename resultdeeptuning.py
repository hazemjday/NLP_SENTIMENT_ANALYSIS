from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import pandas as pd
from transformers import BertTokenizer, BertModel
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import matplotlib.pyplot as plt

df_test = pd.read_csv("./test.csv", encoding="ISO-8859-1")
df_test['text'] = df_test['text'].astype(str)
df_test['label'] = df_test['target'].apply(lambda x: 1 if x == 4 else 0)
x_test = df_test['text']
y_test = df_test['label']

MAX_LEN = 256

if torch.cuda.is_available():       
    device = torch.device("cuda")
    print(f'There are {torch.cuda.device_count()} GPU(s) available.')
    print('Device name:', torch.cuda.get_device_name(0))


#tockenizer predifini dan bert
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
def preprocessing_for_bert(data):
    input_ids = []
    attention_masks = []

    for sent in data:
        encoded_sent = tokenizer.encode_plus(
            text=sent,  # Preprocess sentence
            add_special_tokens=True,        # Add `[CLS]` and `[SEP]`
            max_length=MAX_LEN,                  # Max length to truncate/pad
            padding="max_length",  
            truncation=True ,    
            return_attention_mask=True      # Return attention mask
            )
        # ajout des Ids de mots que le modele comprend et le attention mask
        input_ids.append(encoded_sent.get('input_ids'))
        attention_masks.append(encoded_sent.get('attention_mask'))

    #convertir les deux liste en tensors
    input_ids = torch.tensor(input_ids)
    attention_masks = torch.tensor(attention_masks)

    return input_ids, attention_masks





# Étape 2 : Prétraitement avec tokenizer
test_inputs, test_masks = preprocessing_for_bert(x_test)
test_labels = torch.tensor(y_test.values)

test_data = TensorDataset(test_inputs, test_masks, test_labels)
test_dataloader = DataLoader(test_data, batch_size=32)


class BertClassifier(nn.Module):
    def __init__(self, freeze_bert=False):
        #appele de la partie preentrainer
        super(BertClassifier, self).__init__()
        #specification des couches hiideen
        D_in, H, D_out = 768, 50, 2
        # Instantiate BERT model
        self.bert = BertModel.from_pretrained('bert-base-uncased')

        # Instantiate an one-layer feed-forward classifier
        
        self.classifier = nn.Sequential(
          nn.Linear(D_in, 256),     # 1ère couche cachée
          nn.ReLU(),
     # evite surappreentissage  
          nn.Dropout(0.3),          # Dropout 30%   
          nn.Linear(256, H),        # 2ème couche cachée
        #non linearite entre couches
          nn.ReLU(),
     
          nn.Linear(H, D_out)       # Couche de sortie
          )
        #les gradiants de bert prentra  iner sont conserver
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False
        
  
  
  # lier le modele bert predifini a notre modele

    def forward(self, input_ids, attention_mask):
        # Feed input to BERT
        outputs = self.bert(input_ids=input_ids,
                            attention_mask=attention_mask)
        
        # Extract the last hidden state of the token `[CLS]` for classification task
        last_hidden_state_cls = outputs[0][:, 0, :]

        # Feed input to classifier to compute logits
        logits = self.classifier(last_hidden_state_cls)

        return logits




# Étape 3 : Recharger le modèle sauvegardé
model = BertClassifier(freeze_bert=True)
model.load_state_dict(torch.load("./model_deep_fine/best_bert_classifier.pt", map_location=device))
model.to(device)
model.eval()

# Étape 4 : Prédictions
all_preds = []
all_true = []

with torch.no_grad():
    for batch in test_dataloader:
        b_input_ids, b_attn_mask, b_labels = tuple(t.to(device) for t in batch)
        logits = model(b_input_ids, b_attn_mask)
        preds = torch.argmax(logits, dim=1).flatten()
        all_preds.extend(preds.cpu().numpy())
        all_true.extend(b_labels.cpu().numpy())

# Étape 5 : Évaluation
print("\nClassification Report :\n")
print(classification_report(all_true, all_preds, target_names=["négatif", "positif"]))

print("\nMatrice de Confusion :\n")
cm = confusion_matrix(all_true, all_preds)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["négatif", "positif"], yticklabels=["négatif", "positif"])
plt.xlabel("Prédit")
plt.ylabel("Réel")
plt.title("Matrice de confusion - Test set")
plt.show()