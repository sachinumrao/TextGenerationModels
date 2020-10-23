import pickle
import pathlib
import nltk
from nltk.tokenize import sent_tokenize
import contractions
from tqdm import tqdm


class DataProcessor:
    """
    docstring
    """
    
    def __init__(self, input_text_path, output_text_path):
        self.input_text_path = input_text_path
        self.output_text_path = output_text_path
        self.sentecnes = []
        self.clean_sentences = []
        self.data = ""
        
    def read_data(self):
        f = open(self.input_text_path, "r")
        self.data = f.read()
        
    def tokenize(self):
        self.sentences = sent_tokenize(self.data)
        
    def get_clean_sentences(self, removals, fix_contractions=False):
        filters = removals.keys()
        for sent in tqdm(self.sentences, total=len(self.sentences), desc="Cleaning:"):
            for filter in filters:
                sent = sent.replace(filter, removals[filter])
                
            sent = contractions.fix(sent)
            self.clean_sentences.append(sent)
            
    def save_tokenized_sents(self):
        with open(self.output_text_path, 'wb') as f:
            pickle.dump(self.clean_sentences, f)
    
    
if __name__ == "__main__":
    input_text_path = pathlib.Path.home().joinpath('Data', 'LM_Data', 'harry.txt')
    output_text_path = pathlib.Path.home().joinpath('Data', 'LM_Data', 'harry_sentences.pkl')
    processor = DataProcessor(input_text_path, output_text_path)
    processor.read_data()
    processor.tokenize()
    filter_strings = {'\n': ' ',
                      '\u3000': '',
                      '\xa0': '',
                      '\'': "'"}
    processor.get_clean_sentences(filter_strings, fix_contractions=True)
    processor.save_tokenized_sents()