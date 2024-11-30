import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer
import torch
import re

DATA_PATH = "data/BFB.csv"

TRAIT_LABELS = ["Extraversion","Agreeableness","Openness","Neuroticism","Conscientiousness"]

class Preprocessor():
    def __init__(self,path):
        self.data = pd.read_csv(path)



    def display_freq_bar(self):
        ''' 
        Displays bar plot with frequency of each trait in dataset
        '''
        counts = []
        for label in TRAIT_LABELS:
            count = len(self.data[self.data[label] == 1])
            counts.append(count)

        plt.bar(range(len(TRAIT_LABELS)), counts)

        plt.title(f'Frequency of personality traits in dataset.')
        plt.xlabel('Personality Trait')
        plt.ylabel('Frequency')
        plt.xticks(range(len(TRAIT_LABELS)), TRAIT_LABELS)


        plt.tight_layout()
        plt.show()
    
    def text_length_histogram(self,data):
        '''
        Displays historgram of sequence lengths in column of text in given pandas df
        sequence length is number of chars 
        '''
        
        string_lengths = data['text'].apply(lambda x: len(x) if pd.notna(x) else 0)


        embed_size = string_lengths.quantile(0.9)

        plt.hist(string_lengths)

        plt.title("Histogram of length of texts")
        plt.xlabel("Length of text")
        plt.ylabel("Frequency")
        plt.show()

    def sentence_count_histogram(self):
        '''
        Displays histogram of number of lines in sequence
        '''
        num_lines = self.data["num_lines"]

        plt.hist(num_lines,bins=40,range=(0,300))

        plt.title("Histogram of number of lines in text")
        plt.xlabel("Number of lines")
        plt.ylabel("Frequency")
        plt.show()



    def split_text(self, row_count=None, sentence_count=3):
        '''
        splits text column of first row_count rows of data to sentence_count
        row_count = None splits all rows.
        '''

        def spl(x):
            ''' 
            Splits given text to valid sequence of sentences
            '''
            split = re.split(r'(?<=[.!?]) +', x)
            res = []
            for i in range(0,len(split)-2,sentence_count):
                sent = split[i]
                sent += split[i+1]
                sent += split[i+2]
                res.append(sent)
            return res
            
        rows = self.data.iloc[0:] if (row_count == None) else self.data.iloc[0:row_count]
        rows["text"] = rows["text"].map(spl)
        rows = rows.explode("text")

        return rows


    def train_test_split(self, train_size):
        '''
        Splits training and testing data into two from original DataFrame.
        train_size: between 0 and 1.0
        '''

        label_df = self.data[["text", "Extraversion", "Agreeableness", "Openness", "Neuroticism", "Conscientiousness"]]
        
        train, test = train_test_split(label_df, train_size=train_size, shuffle=True)

        return train.iloc[:, 0], train.iloc[:, 1:], test.iloc[:, 0], test.iloc[:, 1:]


# pp = Preprocessor(DATA_PATH)
# print(pp.data.shape)
# rows = pp.split_text()
# pp.text_length_histogram(rows)
# print(rows.shape)





class BertTextTokenizer:
    def __init__(self,model_name="bert-base-uncased",max_length=512):

        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.max_length = max_length

    def tokenize(self,texts):
        encoded_inputs = self.tokenizer(
            texts,
            padding=True,         
            truncation=True,      
            max_length=self.max_length,
            return_tensors='pt', 
            return_attention_mask=True  
        )
        
        return encoded_inputs['input_ids'], encoded_inputs['attention_mask']
    
    def encode_text(self,texts):
        '''
        Process texts 
        '''
        return self.tokenize(texts)

    def decode_text(self,input_ids):
        '''
        Decodes single or batch of token ID's into texts
        '''
        if isinstance(input_ids, list) or len(input_ids.shape) == 1:
            return self.tokenizer.decode(input_ids, skip_special_tokens=True)
        else:
            return [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids]

    
    def process_batch(self, texts, batch_size=32):
        '''
        Processes the input text in batches.
        Args:
            texts: List of texts to tokenize.
            batch_size: Size of each batch of text.
        Returns:
            Batches of input_ids and attention_mask as PyTorch tensors.
        '''

        num_batches = len(texts) // batch_size + (1 if len(texts) % batch_size != 0 else 0)
        
        for i in range(num_batches):
            batch = texts[i*batch_size: (i+1)*batch_size]
            input_ids, attention_mask = self.tokenize(batch)
            yield input_ids, attention_mask



bt = BertTextTokenizer()

pp = Preprocessor(DATA_PATH)
rows = pp.split_text()
rows = list(rows.sample(1000)["text"])

input_ids,input_attention = bt.encode_text(rows)

print(input_ids.shape)
non_zero_counts = torch.count_nonzero(input_ids,dim=1)

print(non_zero_counts)
print(non_zero_counts.shape)


plt.hist(non_zero_counts)
plt.xlabel("Number of tokens")
plt.ylabel("Frequency")
plt.title("Histogram of token length for a sample of 1000 3 sentence texts")
plt.show()

