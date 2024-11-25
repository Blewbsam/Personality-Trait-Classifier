import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split




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


    def train_test_split(train_size):
        '''
        Splits training and testing data into two
        train_size: betweem 0 and 1.0
        '''

        train,test = train_test_split(self.data,train_size=train_size,shuffle=True)

p = Preprocessor("data/BFB.csv")
p.display_freq_bar()