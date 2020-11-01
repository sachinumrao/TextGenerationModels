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
        
        self.nlp = spacy.load('en_core_web_sm')
    
    def load_sentence_corpus(self):
        with open(self.input_file, 'rb') as f:
            self.sentences = pickle.load(f)
        
    def _check_entity_disambiguation(self, entity):
        pass
        
        
    def _find_triplet(self, sent):
        doc = self.nlp(sent)
        tok_dict = {tok.dep_: tok for tok in doc }
        head = tok_dict.get('nsubj', None)
        relationship = tok_dict.get('ROOT', None)
        if relationship is not None:
            relationship = relationship.lemma_
        tail = tok_dict.get('dobj', None)
        return (head, relationship, tail)
    
    def _validate_triplet(self, triplet):
        return all(triplet)
    
    def _align_triplet(self, triplet):
        head = str(triplet[0]).lower()
        tail = str(triplet[2]).lower()
        relationship = str(triplet[1]).lower()
        return (head, relationship, tail)
        
    def find_fact_triplets(self):
        self.raw_triplets = []
        for sent in tqdm(self.sentences, total=len(self.sentences), desc="Factifying:"):
            triplet = self._find_triplet(sent)
            is_valid = self._validate_triplet(triplet)
            if is_valid:
                triplet = self._align_triplet(triplet)
                self.raw_triplets.append(triplet)
            
    
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