


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


    
    def process_text(self,text):
        '''
        Process a single inputtext
        '''
        return self.tokenize(text)

    
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

