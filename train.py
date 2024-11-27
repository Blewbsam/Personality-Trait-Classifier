from utils import Preprocessor, BertTextTokenizer



# Model parameters:
emb_dim = 512
attention_dim = 512
decoder_dim = 512
dropout = 0
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Training parameters:
epochs = 100
batch_size = 32




def main():

    ## load data
    
    ## setup models
    tokenizer = BertTextTokenizer(emb_dim)


    ## for fixed batch
    
    ## pass data to tokenizer


    ## pass result to model


    ## calculate and display loss



    ## backpropogater

    ## optimizer with Adam w




    


