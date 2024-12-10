from transformers import DistilBertTokenizer
from utils import Preprocessor, TextTokenizer


DATA_PATH = "data/BFB.csv"
SENTENCE_COUNT = 5
SAMPLE_SIZE = 1000

TOKENIZER_NAME = "distilbert-base-uncased"



p = Preprocessor()

# tokenizerModel = DistilBertTokenizer.from_pretrained(TOKENIZER_NAME)
# tokenizer = TextTokenizer(tokenizerModel,96)


# p.tokenized_sequence_length_histogram(tokenizer,SAMPLE_SIZE,SENTENCE_COUNT)
p.split_text_lengths_histogram(SENTENCE_COUNT)





