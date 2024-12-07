import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from utils import Preprocessor, BertTextTokenizer, PersonalityDataset
from utils import Evaluation as eval

from model import *


# Model parameters:
# EMB_DIM = 512
# hidden_size = 512
EMB_DIM = 96
INPUT_SIZE = 1
hidden_size = 512
attention_dim = 512
decoder_dim = 512
dropout = 0
NUM_ROWS = 200
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Training parameters:
EPOCHS = 10
batch_size = 32





def train_model(model,train_loader,val_loader,optimizer,loss_fn,epochs):

    training_losses = []
    validation_losses = []
    accuracies = []


    for epoch in range(epochs):
        model.train()
        train_loss = 0
        accuracy_sum = 0
        batch_count = 0
        training_length = train_loader.__len__()


        for batch in train_loader:
            batch_count += 1
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            labels = batch["label"]

            input_ids = input_ids.unsqueeze(-1) #TODO move to data processing

            optimizer.zero_grad()

            lengths = torch.count_nonzero(attention_mask,dim=1)

            # forward pass
            outputs = model(input_ids.to(torch.float32), lengths)
            loss = loss_fn(outputs, labels)
            
            # backward pass
            loss.backward()
            optimizer.step()

            predicted_classes = (predictions >= 0.5).float()
            accuracy_sum += (predicted_classes == labels).float().mean()
            
            train_loss += loss.item()
            break
        
        val_loss = validate_model(model, val_loader, loss_fn) if val_loader is not None else 0
        accuracy = accuracy_sum / (training_length / batch_count)

        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Accuracy: {accuracy:.4f}")

        training_losses.append(train_loss)
        validation_losses.append(val_loss)
        accuracies.append(accuracy)



    eval.plot_loss(training_losses,validation_losses)
    eval.plot_acc(accuracies)


def validate_model(model,val_loader,loss_fn):
    model.eval()
    val_loss = 0

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            lengths = torch.count_nonzero(attention_mask,dim=1)
            input_ids = input_ids.unsqueeze(-1)
            
            outputs = model(input_ids.to(torch.float32),lengths)
            loss = loss_fn(outputs, labels)
            
            val_loss += loss.item()
    
    return val_loss

def main():

    ## load data
    preprocessor = Preprocessor()
    data = preprocessor.split_text()
    train_texts, train_labels, val_texts, val_labels = preprocessor.train_test_split(data=data,num_rows=NUM_ROWS,train_size=0.8)
    
    ## setup training pipeline and model
    tokenizer = BertTextTokenizer(max_length=EMB_DIM)
    train_dataset = PersonalityDataset(train_texts, train_labels, tokenizer)
    val_dataset = PersonalityDataset(val_texts, val_labels, tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset,batch_size=32)

    encoder = LSTMEncoder(input_size=INPUT_SIZE, hidden_size= hidden_size, num_layers=1, bidirectional=True)
    model = PersonalityClassifier(encoder=encoder, hidden_size=hidden_size*2, output_size=5).to(device)

    ## Training setup
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    loss_fn = nn.BCELoss()  # For binary classification (sigmoid activation)

    train_model(model, train_loader, val_loader, optimizer, loss_fn, epochs=EPOCHS)

if __name__ == "__main__":
    main()