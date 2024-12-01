
import torch
import torch.nn as nn

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
        # Pack the sequence to set all lengths to be the same
        # x = x.unsqueeze(-1)

        print(f"Input shape to model: {x.shape}")
        # packed = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        print("Checkpoint.")
        self.overview()
        out, (h_n, c_n) = self.lstm(x)
        print("Checkpoint 2")
        # unpack the sequence
        # out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)
        # Return final hidden state (concat if bidirectional)
        if self.lstm.bidirectional:
            h_n = h_n.view(self.lstm.num_layers, 2, -1, self.lstm.hidden_size)
            h_n = torch.cat((h_n[-1, 0], h_n[-1, 1]), dim=-1)
        else:
            h_n = h_n[-1]
  
        return h_n


# class LSTMWithAttentionEncoder(nn.module):
    # pass    TODO

class PersonalityClassifier(nn.Module):
    def __init__(self, encoder, hidden_size, output_size):
        super(PersonalityClassifier, self).__init__()
        self.encoder = encoder
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x, lengths):
        encoded = self.encoder(x, lengths)
        print()
        print(f"Encoded shape: {encoded.shape}")  # Add this line to check the shape
        print()
        l1 = self.fc(encoded)
        sig = torch.sigmoid(l1)
        return sig  