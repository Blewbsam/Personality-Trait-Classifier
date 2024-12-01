import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from utils import Preprocessor, BertTextTokenizer, PersonalityDataset
from model import *


# Model parameters:
emb_dim = 512
attention_dim = 512
decoder_dim = 512
dropout = 0
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Training parameters:
epochs = 100
batch_size = 32




def train_model(model,train_loader,val_loader,optimizer,loss_fn,epochs):
    for epoch in range(epochs):
        model.train()
        train_loss = 0

        for batch in train_loader:
            input_ids = batch["inpuit_ids"]
            attention_mask = batch["attention_mask"]
            labels = batch["label"]

            optimizer.zero_grad()
            
            outputs = model(input_ids, attention_mask)
            loss = loss_fn(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        val_loss = validate_model(model, val_loader, loss_fn) if val_loader is not None else 0
        
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")


def validate_model(model,val_loader,loss_fn):
    model.eval()
    val_loss = 0

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            outputs = model(input_ids, attention_mask)
            loss = loss_fn(outputs, labels)
            
            val_loss += loss.item()
    
    return val_loss




def main():

    ## load data
    preprocessor = Preprocessor()
    data = preprocessor.split_text()
    train_texts, train_labels, val_texts, val_labels = preprocessor.train_test_split(data=data,num_rows=32,train_size=0.8)
    
    ## setup training pipeline and model
    tokenizer = BertTextTokenizer(max_length=emb_dim)
    train_dataset = PersonalityDataset(train_texts, train_labels, tokenizer)
    val_dataset = PersonalityDataset(val_texts, val_labels, tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset,batch_size=32)

    encoder = LSTMEncoder(input_size=768, hidden_size=512, num_layers=1, bidirectional=True)
    model = PersonalityClassifier(encoder=encoder, hidden_size=1024, output_size=5).to(device)

    ## Training setup
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    loss_fn = nn.BCELoss()  # For binary classification (sigmoid activation)

    train_model(model, train_loader, val_loader, optimizer, loss_fn, epochs=10)

if __name__ == "__main__":
    main()