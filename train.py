import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from utils import Preprocessor, TextTokenizer, PersonalityDataset
from utils import Evaluation as eval

from model import *


# Model parameters:
# EMB_DIM = 512
# HIDDEN_SIZE = 512
# EMB_DIM = 96
TOKENIZER_NAME = "distilbert-base-uncased"
EMB_DIM =  96
INPUT_SIZE = 1
SENTENCE_SPLIT_COUNT = 5
TRAIN_VAL_SPLIT = 0.8
HIDDEN_SIZE = 128
DROPOUT = 0.4
NUM_ROWS = 2000
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAVE_PATH = "model5-96-10.pth"

# Training parameters:
EPOCHS = 10
batch_size = 32





def train_model(model,train_loader,val_loader,optimizer,loss_fn,epochs,save_path=None):

    training_losses = []
    validation_losses = []
    validation_accuracies = []
    accuracies = []


    for epoch in range(epochs):
        model.train()
        train_loss = 0
        accuracy_sum = 0
        batch_count = 0

        for batch in train_loader:
            batch_count += 1
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            labels = batch["label"]


            optimizer.zero_grad()

            # lengths = torch.count_nonzero(attention_mask,dim=1)

            # forward pass
            outputs = model(input_ids, attention_mask)
            loss = loss_fn(outputs, labels)
            
            # backward pass
            loss.backward()
            optimizer.step()

            predicted_classes = (outputs >= 0.5).float()
            accuracy_sum += (predicted_classes == labels).float().mean()
            
            train_loss += loss.item()
        
        val_loss,val_acc = validate_model(model, val_loader, loss_fn) if val_loader is not None else 0
        val_loss /= (batch_count / 4) # TODO make split dependatnt
        val_acc /= (batch_count / 4)
        train_loss /= batch_count
        accuracy = accuracy_sum / batch_count
        print()
        print(f"Epoch {epoch+1}/{epochs} with {batch_count} batches")
        print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f} ")
        print(f"Train Accuracy: {accuracy:.4f}, Validation Accuracy {val_acc:.4f}")

        training_losses.append(train_loss)
        validation_losses.append(val_loss)
        accuracies.append(accuracy)  
        validation_accuracies.append(val_acc)



    eval.plot_loss(training_losses,validation_losses,SENTENCE_SPLIT_COUNT,NUM_ROWS)
    eval.plot_acc(accuracies,validation_accuracies,SENTENCE_SPLIT_COUNT,NUM_ROWS)

    if save_path:
        torch.save(model.state_dict(),save_path) 


def validate_model(model,val_loader,loss_fn):
    model.eval()
    val_loss = 0
    val_acc = 0

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            # lengths = torch.count_nonzero(attention_mask,dim=1)
            # input_ids = input_ids.unsqueeze(-1)
            
            outputs = model(input_ids,attention_mask)
            loss = loss_fn(outputs, labels)

            predicted_classes = (outputs >= 0.5).float()
            val_acc += (predicted_classes == labels).float().mean()
            val_loss += loss.item()
    
    return val_loss,val_acc

def main():

    ## load data
    preprocessor = Preprocessor()
    data = preprocessor.split_text(SENTENCE_SPLIT_COUNT)
    train_texts, train_labels, val_texts, val_labels = preprocessor.train_test_split(data=data,num_rows=NUM_ROWS,train_size=TRAIN_VAL_SPLIT)
    
    ## setup training pipeline and model
    tokenizerModel = DistilBertTokenizer.from_pretrained(TOKENIZER_NAME)
    tokenizer = TextTokenizer(tokenizerModel,EMB_DIM)

    train_dataset = PersonalityDataset(train_texts, train_labels, tokenizer)
    val_dataset = PersonalityDataset(val_texts, val_labels, tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset,batch_size=32)

    # encoder = LSTMEncoder(input_size=INPUT_SIZE, hidden_size= HIDDEN_SIZE, num_layers=1, bidirectional=True)
    # model = LSTMPersonalityClassifier(encoder=encoder, hidden_size=HIDDEN_SIZE*2, output_size=5).to(device)
    model = TransformerPersonalityClassifier(hidden_size=HIDDEN_SIZE,output_size=5,dropout_prob=DROPOUT)

    # ## Training setup
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    loss_fn = nn.BCELoss()  # For binary classification (sigmoid activation)

    train_model(model, train_loader, val_loader, optimizer, loss_fn, epochs=EPOCHS,save_path=SAVE_PATH)

if __name__ == "__main__":
    main()