import os
import pandas as pd
import spacy
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy import sparse
import numpy as np 

from lxml import etree
from bs4 import BeautifulSoup

# from sentence_transformers import SentenceTransformer

class DatasetDP(object):
    DATAPATH = '/home/ec2-user/SageMaker/mariano/datasets/displaced_persons/files/labeled_data_latest_08072022.csv'
    VECTORIZER_PATH='/home/ec2-user/SageMaker/mariano/datasets/displaced_persons/representations/'
    nlp = spacy.load('en_core_web_lg', disable=['textcat', 'ner', 'parser', 'tager', ])
    distilbert_path = '/home/ec2-user/SageMaker/mariano/huggingface/pretrained/distilbert-base-uncased/'

    def get_data():
        data = {
            'text': [],
            'label': [],
            'id':[],
        }
        for line in open(DatasetDP.DATAPATH, 'r').read().splitlines()[1:]:
            id_, label = line.split(';')
            assert label==DataItemDP.RELEVANT_LABEL or label== DataItemDP.IRRELEVANT_LABEL
            data['id'].append(id_)
            data['label'].append(label)
            data['text'].append(DataItemDP(id_).get_text())
        
        return pd.DataFrame(data)
    
    def get_DP_labeled_collection():
        collection = []
        df = DatasetDP.get_data()
        for idx in range(df.shape[0]):
            id_ = df['id'].iloc[idx]
            label = df['label'].iloc[idx]
            collection.append(DataItemDP(id_))
            if label == DataItemDP.RELEVANT_LABEL:
                collection[-1].set_relevant()
            else:
                collection[-1].set_irrelevant()

        return collection
    
    def get_DP_unlabeled_collection():        
        collection = []
        df = DatasetDP.get_data()
        for idx in range(df.shape[0]):
            id_ = df['id'].iloc[idx]
            collection.append(DataItemDP(id_))
        return collection
    
    def get_DP_unlabeled_collection_train_test(seed=1234, train_proportion=.8):
        rng = np.random.default_rng(seed=1234)
        unlabeled = DatasetDP.get_DP_unlabeled_collection()
        train = rng.choice(unlabeled, size=int(len(unlabeled)*train_proportion),replace=False)
        train_ids=set([item.id_ for item in train])
        test  = [item for item in unlabeled if not item.id_ in train_ids]
        return train,test
    
    def get_DP_oracle():
#         assert category in set(os.listdir(DatasetDP.DATAPATH))
        oracle = {}
        collection = []
        df = DatasetDP.get_data()
        for idx in range(df.shape[0]):            
            id_ = df['id'].iloc[idx]
            label = df['label'].iloc[idx]
            oracle[id_] = label
        return oracle
    
#     def get_X(type_='bow'):
        
#         representations = DatasetDP.get_DP_representations(type_=type_)
#         df = DatasetDP.get_data()
        
#         if type_=='bow':
#             m = sparse.vstack([representations[f'{label}/{id_}'] for label, id_ in zip(df.iloc[:,1], df.iloc[:,2])])
#             return m
    
    def _preprocessor(text):
        return ' '.join([token.lemma_.lower() for token in DatasetDP.nlp(text) if token.lemma_.isalpha()])
    
    def get_DP_representations(type_='bow'):
        df = DatasetDP.get_data()
        representations = {}
        representations_path = os.path.join(DatasetDP.VECTORIZER_PATH, f'DP_representations_{type_}.pickle')
        if os.path.isfile(representations_path):
#             print(f'representations file found, loading pickle ({representations_path}) ... ')
            representations = pickle.load(open(representations_path, 'rb'))
        else:
            if type_=='bow':
#                 print('Using BOW representation')
                vectorizer_path = os.path.join(DatasetDP.VECTORIZER_PATH, f'DP_vectorizer_{type_}.pickle')
#                 print('representations file NOT found, creating from vectorizer... ')
                if not os.path.isfile(vectorizer_path):
#                     print('vectorizer file NOT found, creating ... ')

                    stopwords=['ll', 've']+\
                            list(DatasetDP.nlp.Defaults.stop_words.difference({'\'ll','\'ve','further','regarding','used','using',}))
                    vectorizer = TfidfVectorizer(lowercase=True, 
                                                 preprocessor=DatasetDP._preprocessor, 
                                                 stop_words=stopwords,
                                                 ngram_range=(1,3), 
                                                 use_idf=True,
                                                 max_features=10000)         
                    X = vectorizer.fit_transform(df['text'])
                    pickle.dump(vectorizer, open(vectorizer_path, 'wb'))
                else:    
#                     print('vectorizer file  found, loading pickle ... ')  
                    vectorizer = pickle.load(open(vectorizer_path, 'rb'))            
                    X = vectorizer.transform(df['text'])
#                 print('Creating representations file')
                for idx in range(len(df)):
                    id_ = df['id'].iloc[idx]
                    label = df['label'].iloc[idx]
                    representations[f'{id_}']=X[idx,:]
#                 print('Dumping representations file to disk')
                pickle.dump(representations, open(representations_path, 'wb'))
            elif type_=='glove':
#                 print('Computing glove ...')
                for idx in range(len(df)):
                    id_ = df['id'].iloc[idx]
                    label = df['label'].iloc[idx]
                    representations[f'{id_}']=DatasetDP.nlp(df['text'].iloc[idx]).vector
                pickle.dump(representations, open(representations_path, 'wb'))
            elif type_=='sbert':
#                 print('Computing sbert ...')
                model = SentenceTransformer(DatasetDP.distilbert_path)
                    
                vectors =  model.encode(df['text'])
        
                for idx,vector in enumerate(vectors):
                    id_ = df['id'].iloc[idx]
                    label = df['label'].iloc[idx]
                    representations[f'{id_}'] = vector

                    
                pickle.dump(representations, open(representations_path, 'wb'))
            
        return representations
    


class DataItemDP(object):
    UNKNOWN_LABEL='U'
    RELEVANT_LABEL='R'
    IRRELEVANT_LABEL='I'
    
#     def compute_representation(type_='bow'):

    
#     def get_representation(self, type_='bow'):
#         if type_=='bow':
#             vectorizer_path = os.path.join(VECTORIZER_PATH,)  
    
    def __init__(self, id_):
        self.id_=id_
        self.label=DataItemDP.UNKNOWN_LABEL

    def is_relevant(self, ):
        return self.label==DataItemDP.RELEVANT_LABEL
    def is_irrelevant(self, ):
        return self.label==DataItemDP.IRRELEVANT_LABEL
    def is_unknown(self, ):
        return self.label==DataItemDP.UNKNOWN_LABEL
    
    def set_relevant(self, ):
        self.label=DataItemDP.RELEVANT_LABEL
    def set_irrelevant(self, ):
        self.label=DataItemDP.IRRELEVANT_LABEL
    def set_unknown(self, ):
        self.label=DataItemDP.UNKNOWN_LABEL
    
    
    def get_text(self, ):
        tree = etree.parse(self.get_filename())
        root = tree.getroot()
        if root.find('.//HiddenText') is not None:
            text = (root.find('.//HiddenText').text)

        elif root.find('.//Text') is not None:
            text = (root.find('.//Text').text)

        else:
            text = None
        title = root.find('.//Title').text
        concated_text = ''
        if not title is None:
            concated_text = f'{title}. '
        if not text is None:
            concated_text += f'{text}'
        return BeautifulSoup(concated_text,'html.parser').get_text()
#         return open(self.get_filename(), 'r', encoding='latin-1').read()
    
    def get_htmldocview(self, ):
        self.get_text()
    
    def get_filename(self, ):
        for folder in os.listdir('/home/ec2-user/SageMaker/data/'):
            if os.path.isfile(os.path.join('/home/ec2-user/SageMaker/data/', folder, self.id_+ '.xml')):
                return os.path.join('/home/ec2-user/SageMaker/data/', folder, self.id_+'.xml')
    
    def __str__(self, ):
        return f'<DP Dataset, id={self.id_}, label={self.label}, >'
    
    def _vector_filename(self, ):
        return 'N/A'
    
