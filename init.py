
from nltk.parse.stanford import StanfordDependencyParser
import nltk
nltk.download('wordnet')
nltk.download('sentiwordnet')

class DependencyParser():
    def __init__(self, model_path, parser_path):
        self.model = StanfordDependencyParser(path_to_jar=parser_path, path_to_models_jar=model_path)

    def raw_parse(self, text):
        parse_result = self.model.raw_parse(text)
        result = [list(parse.triples()) for parse in parse_result]
        return result[0]
    
#set parameters for review processing
model_path = 'config/stanford-corenlp-4.5.7/stanford-corenlp-4.5.7-models.jar'
parser_path = 'config/stanford-corenlp-4.5.7/stanford-corenlp-4.5.7.jar'
dep_parser = DependencyParser(model_path, parser_path)