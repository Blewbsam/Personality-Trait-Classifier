import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer
import torch


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
    
    def text_length_histogram(self):
        '''
        Displays historgram of sequence lengths in column of text 
        '''
        
        string_lengths = self.data['text'].apply(lambda x: len(x) if pd.notna(x) else 0)


        embed_size = string_lengths.quantile(0.9)
        print(embed_size)

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


    def train_test_split(self, train_size):
        '''
        Splits training and testing data into two from original DataFrame.
        train_size: between 0 and 1.0
        '''

        label_df = self.data[["text", "Extraversion", "Agreeableness", "Openness", "Neuroticism", "Conscientiousness"]]
        
        train, test = train_test_split(label_df, train_size=train_size, shuffle=True)

        return train.iloc[:, 0], train.iloc[:, 1:], test.iloc[:, 0], test.iloc[:, 1:]


pp = Preprocessor(DATA_PATH)
pp.sentence_count_histogram()




class BertTextTokenizer:
    def __init__(self,model_name="bert-base-uncased",max_length=128):

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


