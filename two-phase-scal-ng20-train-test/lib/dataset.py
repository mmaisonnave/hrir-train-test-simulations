import os
import pandas as pd
import spacy
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np 

class Dataset20NG(object):
    DATAPATH = '/home/ec2-user/SageMaker/mariano/datasets/20news-18828/files/'
    VECTORIZER_PATH='/home/ec2-user/SageMaker/mariano/datasets/20news-18828/representations/'
    distilbert_path = '/home/ec2-user/SageMaker/mariano/huggingface/pretrained/distilbert-base-uncased/'

    nlp = spacy.load('en_core_web_lg', disable=['textcat', 'ner', 'parser', 'tager', ])

    def get_data():
        data = {
            'text': [],
            'label': [],
            'id':[],
        }

        for folder in os.listdir(Dataset20NG.DATAPATH): 
            files = os.listdir(os.path.join(Dataset20NG.DATAPATH,folder))
            data['text'] += [open(os.path.join(Dataset20NG.DATAPATH,folder,file), 'r', encoding='latin-1').read() for file in files]
            data['label'] += [folder]*len(files)
            data['id'] += [file.split('/')[-1] for file in files]
        
        return pd.DataFrame(data)
    

    
    def get_20newsgroup_labeled_collection(category):
        assert category in set(os.listdir(Dataset20NG.DATAPATH))
        collection = []
        for folder in os.listdir(Dataset20NG.DATAPATH): 
            for file in os.listdir(os.path.join(Dataset20NG.DATAPATH,folder)):
                collection.append(DataItem20NG(id_=file, category=folder, ), )
                if folder==category:
                    collection[-1].set_relevant()
                else:
                    collection[-1].set_irrelevant()
        return collection
    
    def get_20newsgroup_unlabeled_collection():
        collection = []
        for folder in os.listdir(Dataset20NG.DATAPATH): 
            for file in os.listdir(os.path.join(Dataset20NG.DATAPATH,folder)):
                collection.append(DataItem20NG(id_=file, category=folder, ), )
        return collection
    
    def get_20newsgroup_unlabeled_collection_train_test(seed=1234, train_proportion=.8):
        rng = np.random.default_rng(seed=1234)
        unlabeled = Dataset20NG.get_20newsgroup_unlabeled_collection()
        train = rng.choice(unlabeled, size=int(len(unlabeled)*train_proportion),replace=False)
        train_ids=set([item.id_ for item in train])
        test  = [item for item in unlabeled if not item.id_ in train_ids]
        return train,test
    
    def get_20newsgroup_oracle(category):
        assert category in set(os.listdir(Dataset20NG.DATAPATH))
        oracle = {}
        for folder in os.listdir(Dataset20NG.DATAPATH): 
            for file in os.listdir(os.path.join(Dataset20NG.DATAPATH,folder)):
                oracle[f'{folder}/{file}'] = DataItem20NG.RELEVANT_LABEL if folder==category else DataItem20NG.IRRELEVANT_LABEL
        
        return oracle
    
#     def get_X(type_='bow'):
        
#         representations = Dataset20NG.get_20newsgroup_representations(type_=type_)
#         df = Dataset20NG.get_data()
        
#         if type_=='bow':
#             m = sparse.vstack([representations[f'{label}/{id_}'] for label, id_ in zip(df.iloc[:,1], df.iloc[:,2])])
#             return m
    
    def _preprocessor(text):
        return ' '.join([token.lemma_.lower() for token in Dataset20NG.nlp(text) if token.lemma_.isalpha()])
    
    def get_20newsgroup_representations(type_='bow'):
        df = Dataset20NG.get_data()
        representations = {}
        representations_path = os.path.join(Dataset20NG.VECTORIZER_PATH, f'20NG_representations_{type_}.pickle')
        if os.path.isfile(representations_path):
#             print(f'representations file found, loading pickle ({representations_path}) ... ')
            representations = pickle.load(open(representations_path, 'rb'))
        else:
            if type_=='bow':
#                 print('Using BOW representation')
                vectorizer_path = os.path.join(Dataset20NG.VECTORIZER_PATH, f'20NG_vectorizer_{type_}.pickle')
                print('representations file NOT found, creating from vectorizer... ')
                if not os.path.isfile(vectorizer_path):
                    print('vectorizer file NOT found, creating ... ')

                    stopwords=['ll', 've']+\
                            list(Dataset20NG.nlp.Defaults.stop_words.difference({'\'ll','\'ve','further','regarding','used','using',}))
                    vectorizer = TfidfVectorizer(lowercase=True, 
                                                 preprocessor=Dataset20NG._preprocessor, 
                                                 stop_words=stopwords,
                                                 ngram_range=(1,3), 
                                                 use_idf=True,
                                                 max_features=10000)         
                    X = vectorizer.fit_transform(df['text'])
                    pickle.dump(vectorizer, open(vectorizer_path, 'wb'))
                else:    
                    print('vectorizer file  found, loading pickle ... ')  
                    vectorizer = pickle.load(open(vectorizer_path, 'rb'))            
                    X = vectorizer.transform(df['text'])
                print('Creating representations file')
                for idx in range(len(df)):
                    id_ = df['id'].iloc[idx]
                    label = df['label'].iloc[idx]
                    representations[f'{label}/{id_}']=X[idx,:]
                print('Dumping representations file to disk')
                pickle.dump(representations, open(representations_path, 'wb'))
            elif type_=='glove':
                print('Computing glove ...')
                for idx in range(len(df)):
                    id_ = df['id'].iloc[idx]
                    label = df['label'].iloc[idx]
                    representations[f'{label}/{id_}']=Dataset20NG.nlp(df['text'].iloc[idx]).vector
                pickle.dump(representations, open(representations_path, 'wb'))
            elif type_=='sbert':
                print('Computing sbert ...')
                model = SentenceTransformer(Dataset20NG.distilbert_path)
                    
                vectors =  model.encode(df['text'])
        
                for idx,vector in enumerate(vectors):
                    id_ = df['id'].iloc[idx]
                    label = df['label'].iloc[idx]
                    representations[f'{label}/{id_}'] = vector

                    
                pickle.dump(representations, open(representations_path, 'wb'))
            
        return representations
    


class DataItem20NG(object):
    UNKNOWN_LABEL='U'
    RELEVANT_LABEL='R'
    IRRELEVANT_LABEL='I'
    
#     def compute_representation(type_='bow'):

    
#     def get_representation(self, type_='bow'):
#         if type_=='bow':
#             vectorizer_path = os.path.join(VECTORIZER_PATH,)  
    
    def __init__(self, id_, category):
        self.id_=f'{category}/{id_}'
        self.label=DataItem20NG.UNKNOWN_LABEL
        self.category=category
    def is_relevant(self, ):
        return self.label==DataItem20NG.RELEVANT_LABEL
    def is_irrelevant(self, ):
        return self.label==DataItem20NG.IRRELEVANT_LABEL
    def is_unknown(self, ):
        return self.label==DataItem20NG.UNKNOWN_LABEL
    
    def set_relevant(self, ):
        self.label=DataItem20NG.RELEVANT_LABEL
    def set_irrelevant(self, ):
        self.label=DataItem20NG.IRRELEVANT_LABEL
    def set_unknown(self, ):
        self.label=DataItem20NG.UNKNOWN_LABEL
    
    
    def get_text(self, ):
        return open(self.get_filename(), 'r', encoding='latin-1').read()
    
    def get_htmldocview(self, ):
        self.get_text()
    
    def get_filename(self, ):
        return os.path.join(Dataset20NG.DATAPATH, self.id_, )
    
    def __str__(self, ):
        return f'<id={self.id_}, label={self.label}, >'
    
    def _vector_filename(self, ):
        return 'N/A'
    
# class DataItem20NG(object):
#     UNKNOWN_LABEL=-1
#     VALID_LABELS=set(os.listdir(Dataset20NG.datapath))
#     def __init__(self, id_ = None, label = None ):
#         assert label is None or label in DataItem20NG.VALID_LABELS
#         self.id=id_
#         self.label=UNKNOWN_LABEL if label is None else label
        
#     def set_id(self, id_):
#         self.id_ = id_
        
#     def set_label(self, label):
#         self.label = label
        
#     def get_text(self, ):
#         return open(self.get_filename(),  'r', encoding='latin-1', ).read()    
    
#     def get_filename(self, ):
#         assert self.id!=None and self.label!=DataItem20NG.UNKNOWN_LABEL, 'Requested filename for an incomplete DataItem'
#         return os.path.join(Dataset20NG.datapath, self.label, self.id, )
