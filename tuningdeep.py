#utulisation de la base bert avec reseaux de neurones pour entrainer le modele

import os
import re
from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from transformers import BertTokenizer, BertModel
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import torch.nn as nn
from transformers import BertModel
from transformers import  get_linear_schedule_with_warmup
import random
from torch.optim import AdamW

#importation et gestion des données
df_train = pd.read_csv("./train.csv", encoding="ISO-8859-1")
df_val = pd.read_csv("./validation.csv", encoding="ISO-8859-1")
df_test = pd.read_csv("./test.csv", encoding="ISO-8859-1")
df_train['label'] = df_train['target'].apply(lambda x: 1 if x == 4 else 0)
df_val['label'] = df_val['target'].apply(lambda x: 1 if x == 4 else 0)
df_test['label'] = df_test['target'].apply(lambda x: 1 if x == 4 else 0)
df_train['text'] = df_train['text'].astype(str)
df_val['text'] = df_val['text'].astype(str)
df_test['text'] = df_test['text'].astype(str)

#train validation data
x_train = df_train['text']
y_train = df_train['label']
x_val = df_val['text']
y_val = df_val['label']


# config du gpu
if torch.cuda.is_available():       
    device = torch.device("cuda")
    print(f'There are {torch.cuda.device_count()} GPU(s) available.')
    print('Device name:', torch.cuda.get_device_name(0))

else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")


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
            padding="max_length",        # Pad sentence to max length
            return_attention_mask=True      # Return attention mask
            )
        # ajout des Ids de mots que le modele comprend et le attention mask
        input_ids.append(encoded_sent.get('input_ids'))
        attention_masks.append(encoded_sent.get('attention_mask'))

    #convertir les deux liste en tensors
    input_ids = torch.tensor(input_ids)
    attention_masks = torch.tensor(attention_masks)

    return input_ids, attention_masks

# tockenisation des donnee train et validation
MAX_LEN = 256
train_inputs, train_masks = preprocessing_for_bert(x_train)
val_inputs, val_masks = preprocessing_for_bert(x_val)


train_labels = torch.tensor(y_train)
val_labels = torch.tensor(y_val)
#creation des dataloaders
batch_size = 32
train_data = TensorDataset(train_inputs, train_masks, train_labels)
train_dataloader = DataLoader(train_data, batch_size=batch_size)

# Create the DataLoader for our validation set
val_data = TensorDataset(val_inputs, val_masks, val_labels)
val_dataloader = DataLoader(val_data, batch_size=batch_size)





#creation du modele
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




def initialize_model(epochs=4):
    bert_classifier = BertClassifier(freeze_bert=True)
    bert_classifier.to(device)
  #Adamw et pas Adam resonsable de updater les poids
    optimizer = AdamW(bert_classifier.parameters(),
                      lr=5e-5,    # Default learning rate
                      eps=1e-8    # Default epsilon value
                      )
#dilminuer le learning rate au cours du temps    
    total_steps = len(train_dataloader) * epochs 
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=0, # Default value
                                                num_training_steps=total_steps)
    return bert_classifier, optimizer, scheduler


# Specify loss function
loss_fn = nn.CrossEntropyLoss()


random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

def train(model, train_dataloader, val_dataloader=None, epochs=4, evaluation=False):
    print("Start training...\n")
    best_val_accuracy = 0.0  # <- on garde la meilleure val_accuracy
    save_path = "./model_deep_fine"
    os.makedirs(save_path, exist_ok=True)
    for epoch_i in range(epochs):
        print(f"{'Epoch':^7} | {'Batch':^7} | {'Train Loss':^12} | {'Val Loss':^10} | {'Val Acc':^9} | {'Elapsed':^9}")
        print("-"*70)
        total_loss, batch_loss = 0, 0
        model.train()
        for step, batch in enumerate(train_dataloader):
           #tuple contenant les differents tensors
            b_input_ids, b_attn_mask, b_labels = tuple(t.to(device) for t in batch)
            model.zero_grad()
            #faire les prediction
            logits = model(b_input_ids, b_attn_mask)
           #calcule de loss et son incrementation
            loss = loss_fn(logits, b_labels)
            batch_loss += loss.item()
            total_loss += loss.item()
        # backword pour faire mis a jour du gradiant
            loss.backward()
          # max norme du gradiant est 1 pour eviter le exploding
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
       #mis ajours des poids
            optimizer.step()
            scheduler.step()
        avg_train_loss = total_loss / len(train_dataloader)
        if evaluation == True:
            val_loss, val_accuracy = evaluate(model, val_dataloader)      
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                #on ne sauvegarde qi=ue les poids dans le personalise
                torch.save(model.state_dict(), os.path.join(save_path, "best_bert_classifier.pt"))
                print(f"Meilleur modèle sauvegardé à epoch {epoch_i+1} avec val_acc = {val_accuracy:.2f}%")
            print(f"{epoch_i + 1:^7} | {'-':^7} | {avg_train_loss:^12.6f} | {val_loss:^10.6f} | {val_accuracy:^9.2f} ")
            print("-"*70)
        print("\n")
    
    print("Training complete!")


def evaluate(model, val_dataloader):

    # Put the model into the evaluation mode. The dropout layers are disabled during
    # the test time.
    model.eval()

    # Tracking variables
    val_accuracy = []
    val_loss = []

    # For each batch in our validation set...
    for batch in val_dataloader:
        # Load batch to GPU
        b_input_ids, b_attn_mask, b_labels = tuple(t.to(device) for t in batch)

        # Compute logits
        with torch.no_grad():
            logits = model(b_input_ids, b_attn_mask)

        # Compute loss
        loss = loss_fn(logits, b_labels)
        val_loss.append(loss.item())

        # Get the predictions
        preds = torch.argmax(logits, dim=1).flatten()

        # Calculate the accuracy rate
        accuracy = (preds == b_labels).cpu().numpy().mean() * 100
        val_accuracy.append(accuracy)

    # Compute the average accuracy and loss over the validation set.
    val_loss = np.mean(val_loss)
    val_accuracy = np.mean(val_accuracy)

    return val_loss, val_accuracy



bert_classifier, optimizer, scheduler = initialize_model(epochs=3)
train(bert_classifier, train_dataloader, val_dataloader, epochs=3, evaluation=True)
