import pickle
import pathlib
import nltk
from nltk.tokenize import sent_tokenize

def DataProcessor(self, parameter_list):
    """
    docstring
    """
    
    def __init__(self, input_text_path, output_text_path):
        self.input_text_path = input_text_path
        self.output_text_path = output_text_path
        self.sentecnes = []
        self.data = ""
        
    def read_data(self):
        f = open(self.input_text_path, "r")
        self.data = f.read()
        
    def tokenize(self):
        self.sentences = sent_tokenize(self.data)
        
    def clean_sentences(self, removals=[], fix_contractions=False):
        pass
    
    def save_tokenized_sents(self):
        with open(self.output_text_path, 'wb') as f:
            pickle.dump(self.sentecnes, f)
    
    
if __name__ == "__main__":
    input_text_path = pathlib.Path.home().joinpath('Data', 'LM_Data', 'harry.txt')
    output_text_path = pathlib.Path.home().joinpath('Data', 'LM_Data', 'harry_sentences.pkl')
    tokenizer = DataProcessor(input_text_path, output_text_path)
    tokenizer.read_data()
    tokenizer.tokenize()
    tokenizer.save_tokenized_sents()