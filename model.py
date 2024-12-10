
import torch
import torch.nn as nn
from transformers import DistilBertModel, DistilBertTokenizer

class LSTMEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, bidirectional=False, dropout=0.5):
        super(LSTMEncoder, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0
        )

    def overview(self):
        for name, param in self.lstm.named_parameters():
            print(name,param.shape)
    
    def forward(self, x, lengths):  
        ''' 
        x: (batch_size,sequence_length,input_size)
        lengths: 1d tensor of integers specifying how each inputs in the 
        batch is to be packed to ignore padding for efficien computation
        '''
        packed = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        out, (h_n, c_n) = self.lstm(x)

        # out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True) # unpack for testing

        # Return final hidden state (concat if bidirectional)
        if self.lstm.bidirectional:
            h_n = h_n.view(self.lstm.num_layers, 2, -1, self.lstm.hidden_size)
            h_n = torch.cat((h_n[-1, 0], h_n[-1, 1]), dim=-1)
        else:
            h_n = h_n[-1]
        return h_n


# class LSTMWithAttentionEncoder(nn.module):
    # pass    TODO


class TransformerPersonalityClassifier(nn.Module):
    def __init__(self, hidden_size, output_size, pretrained_model="distilbert-base-uncased",dropout_prob=0.3):
        super(TransformerPersonalityClassifier, self).__init__()
        # Load DistilBERT model
        self.encoder = DistilBertModel.from_pretrained(pretrained_model)
        self.fc = nn.Linear(self.encoder.config.hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout_prob) 
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, input_ids, attention_mask):
        # Get embeddings from DistilBERT
        assert input_ids.ndim == 2, f"Expected 2D input_ids, got {input_ids.ndim}D"
        assert attention_mask.ndim == 2, f"Expected 2D attention_mask, got {attention_mask.ndim}D"

        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0]  # CLS token representation
        # possibly add batch normalization
        l1 = torch.relu(self.fc(pooled_output)) 
        l1 = self.dropout(l1)
        l2 = self.fc2(l1)
        sig = torch.sigmoid(l2)  # Multi-label classification
        return sig



class LSTMPersonalityClassifier(nn.Module):
    def __init__(self, encoder, hidden_size, output_size):
        ## Expects LSTM encoder
        super(LSTMPersonalityClassifier, self).__init__()
        self.encoder = encoder
        self.fc = nn.Linear(hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size,output_size)
    
    def forward(self, x, lengths):
        assert x.ndim == 3
        assert lengths.ndim == 1
        encoded = self.encoder(x, lengths)
        l1 = self.fc(encoded)
        l2 = self.fc2(l1)
        sig = torch.sigmoid(l2)
        return sig  
