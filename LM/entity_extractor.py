import spacy
import pickle
import pandas as pd

class EntityExtractor:
    """
    docstring
    """
    
    def __init__(self, input_file, output_file):
        self.input_file = input_file
        self.output_file = output_file
        self.entity_file = None
        self.fact_triplets = None
        
        self.sentences = []
        self.raw_triplets = []
        self.entity_list = []
        
    
    def load_sentence_corpus(self):
        with open(self.input_file, 'rb') as f:
            self.sentences = pickle.load(f)
        
    def check_entity_disambiguation(self, entity):
        pass
        
        
    def find_entities(self, sent):
        pass
    
    def format_triplets(self):
        self.fact_triplets = pd.DataFrame(columns=['Head', 'Relation', 'Tail'])
        pass
    
    def save_entity_list(self):
        with open(self.entity_file, 'w') as f:
            for item in self.entity_list:
                f.write("%s\n" % item)