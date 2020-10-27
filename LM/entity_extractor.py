import spacy
import pickle
import pandas as pd
import pathlib
from tqdm import tqdm

class EntityExtractor:
    """
    docstring
    """
    
    def __init__(self, input_file, output_file):
        self.input_file = input_file
        self.output_file = output_file
        self.fact_triplet_df = None
        
        self.sentences = []
        self.raw_triplets = []
        self.entity_list = []
        
    
    def load_sentence_corpus(self):
        with open(self.input_file, 'rb') as f:
            self.sentences = pickle.load(f)
        
    def _check_entity_disambiguation(self, entity):
        pass
        
        
    def find_entities(self, sent):
        pass
    
    def save_triplets(self):
        self.fact_triplet_df = pd.DataFrame(columns=['Head', 'Relation', 'Tail'])
        
        # self.fact_triplet_df.to_csv(self.output_file, index=False)
        pass
    
    def save_entity_list(self, entity_file_name):
        with open(entity_file_name, 'w') as f:
            for item in self.entity_list:
                f.write("%s\n" % item)
                
                
if __name__ == "__main__":
    corpus_file =  pathlib.Path.home().joinpath('Data', 'LM_Data', 'harry_sentences.pkl')
    output_file = pathlib.Path.home().joinpath('Data', 'LM_Data', 'fact_triplets.csv')
    entity_file = pathlib.Path.home().joinpath('Data', 'LM_Data', 'harry_potter_entities.txt')
    entity_finder = EntityExtractor(corpus_file, 
                                    output_file)
    
    entity_finder.load_sentence_corpus()
    
    pass