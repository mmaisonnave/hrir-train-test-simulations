import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, confusion_matrix
from lib.dataset import  DataItem20NG
from lib.models import LogisticRegression20NG, SVM20NG
import datetime
import pytz
####################################################################################
#                                20 newsgroup SCAL                                 #
####################################################################################
class SCAL20NG(object):
    HOME_FOLDER=f'sessions/scal/'
    
    def __init__(self,
                 session_name=None,
                 labeled_collection=None,
                 unlabeled_collection=None,
                 batch_size_cap=10,
                 random_sample_size=10000,
                 target_recall=0.8,
                 ranking_function='relevance',
                 item_representation=None,
                 model_type='logreg',
                 oracle=None,
                 smart_start=True,
                 seed=2022):

        # Labeled data has labels
        assert all([oracle[id_]==DataItem20NG.RELEVANT_LABEL or oracle[id_]==DataItem20NG.IRRELEVANT_LABEL for id_ in oracle])
        self.model_type= model_type
        self.B=1
        self.oracle=oracle
        self.target_recall=target_recall
        self.ranking_function=ranking_function
        self.n=batch_size_cap
        self.N=random_sample_size
        self.item_representation=item_representation
        
        self.labeled_collection = labeled_collection
        self.full_unlabeled_collection = unlabeled_collection
        

        self.seed = seed
        # <--------------------------------------------------------------------------------------------------------------
        #############################################################
        # CORRECTED. REPLACED RANDOM FOR HALF AND HALF USING ORACLE #
        #############################################################
        
        self.ran = np.random.default_rng(self.seed)
        full_N=min(self.N,len(unlabeled_collection))
        N1=min(int(full_N/2), len([item for item in unlabeled_collection if  DataItem20NG.RELEVANT_LABEL==self.oracle[item.id_]]))
        N2=int(full_N-N1)
        # DataItem20NG.RELEVANT_LABEL==self.oracle[item.id_]
        if smart_start:
            relevant_items=[item for item in unlabeled_collection if  DataItem20NG.RELEVANT_LABEL==self.oracle[item.id_]]
            irrelevant_items=[item for item in unlabeled_collection if  DataItem20NG.IRRELEVANT_LABEL==self.oracle[item.id_]]

            assert len(relevant_items)+len(irrelevant_items)==len(unlabeled_collection)

            self.sample_unlabeled_collection = list(self.ran.choice(relevant_items, 
                                                               size=N1, 
                                                               replace=False))
            self.sample_unlabeled_collection += list(self.ran.choice(irrelevant_items, 
                                                               size=N2, 
                                                               replace=False))
            
        else: # RANDOM START
            self.sample_unlabeled_collection = self.ran.choice(unlabeled_collection, 
                                                               size=full_N, 
                                                               replace=False)
        # <--------------------------------------------------------------------------------------------------------------
        
        self.full_U = self.sample_unlabeled_collection
        
        self.cant_iterations = SCAL20NG._cant_iterations(self.N)        
        self.Rhat=np.zeros(shape=(self.cant_iterations,))
        
        self.j=0            
        self.removed = []
        self.models=[]
        self.precision_estimates=[]
        
#         self.all_texts = [item.get_htmldocview() for item in labeled_collection]
#         self.all_labels = [SCAL.RELEVANT_LABEL if item.is_relevant() else SCAL.IRRELEVANT_LABEL for item in labeled_collection]

        
    def run(self,):
        
        while len(self.sample_unlabeled_collection)>0:
            self.loop()
            self.after_loop()
        self.finish()
#         print('finish')
        return self.results
            
#         self.loop()
    
    def _select_highest_scoring_docs(self, function):
        """
            valid functions: "relevance" "uncertainty" "avg_distance" "min_distance"
        """
        # RELEVANCE 
        if function=='relevance':
            yhat = self.models[-1].predict(self.sample_unlabeled_collection, item_representation=self.item_representation)
            args = np.argsort(yhat)[::-1]
            
        # UNCERTAINTY
        elif function=='uncertainty':
            yhat = self.models[-1].predict(self.sample_unlabeled_collection, item_representation=self.item_representation)
            args = np.argsort(np.abs(yhat-0.5))
            
        elif function=='1quarter_relevance_3quarters_uncertainty':
            current_proportion = len(self.labeled_collection)/(self._total_effort()+1)
            if current_proportion<0.25:
                yhat = self.models[-1].predict(self.sample_unlabeled_collection, item_representation=self.item_representation)
                args = np.argsort(yhat)[::-1]
            else:
                yhat = self.models[-1].predict(self.sample_unlabeled_collection, item_representation=self.item_representation)
                args = np.argsort(np.abs(yhat-0.5))
                
        # RANDOM
        elif function=='random':
            args = list(range(len(self.sample_unlabeled_collection)))
            np.random.shuffle(args)
     
        else:
            assert False, f'Invalid ranking function: {function}'

        return [self.sample_unlabeled_collection[arg] for arg in args[:self.B]]
    
    def _extend_with_random_documents(self):    
        
        assert all([item.is_unknown() for item in self.sample_unlabeled_collection]), f'B={self.B} - j={self.j}'
        
        extension = self.ran.choice(self.sample_unlabeled_collection, size=min(100,len(self.sample_unlabeled_collection)), replace=False)
        list(map(lambda x: x.set_irrelevant(), extension))
        assert all([item.is_irrelevant() for item in extension])
        return extension
    
    def _label_as_unknown(collection):
        list(map(lambda x: x.set_unknown(), collection))
        
    def _build_classifier(self, training_collection):
        assert self.model_type=='logreg' or self.model_type=='svm'
        if self.model_type=='logreg':
            model = LogisticRegression20NG()
        elif self.model_type=='svm':
            model = SVM20NG()
            
        model.fit(training_collection, item_representation=self.item_representation)
        return model
    
      
    def _remove_from_unlabeled(self,to_remove):
        to_remove = set(to_remove)
        return list(filter(lambda x: not x in to_remove, self.sample_unlabeled_collection))

    
    def _get_Uj(self,j):
        to_remove = set([elem for list_ in self.removed[:(j+1)]  for elem in list_])
        Uj = [elem for elem in self.full_U if not elem in to_remove]
        return Uj
    
    def _total_effort(self):  
#         if not hasattr(self, 'labeling_budget'):
        B=1
        it=1
        effort=0
#             len_unlabeled=self._unlabeled_in_sample()
        len_unlabeled=min(self.N, len(self.full_unlabeled_collection))
        while (len_unlabeled>0):        
            b = B if B<=self.n else self.n
            effort+=min(b,len_unlabeled)
            len_unlabeled = len_unlabeled - B
            B+=int(np.ceil(B/10))
            it+=1
        self.labeling_budget = effort
        
        return self.labeling_budget   
    def _cant_iterations(len_unlabeled):    
        B=1
        it=0
        while len_unlabeled>0:        
            len_unlabeled = len_unlabeled - B
            B+=int(np.ceil(B/10))
            it+=1
        return it
    

    def loop(self):
        self.b = self.B if (self.Rhat[self.j]==1 or self.B<=self.n) else self.n
#         precision = f'{self.precision_estimates[-1]:4.3f}' if len(self.precision_estimates)>0 else 'N/A'       
        
        
        extension = self._extend_with_random_documents()
        
        # new model created, 
        self.models.append(self._build_classifier(list(extension)+list(self.labeled_collection)))

        SCAL20NG._label_as_unknown(extension)
        self.sorted_docs = self._select_highest_scoring_docs(function=self.ranking_function)

        self.random_sample_from_batch = self.ran.choice(self.sorted_docs, size=self.b, replace=False)
                  
                  
#         yhat = self.models[-1].predict(self.random_sample_from_batch, item_representation=self.item_representation)
        
        
                  
#         text_for_label = [suggestion.get_htmldocview(highlighter=None, confidence_score=confidence)
#                           for suggestion,confidence in zip(self.random_sample_from_batch,yhat)]
        
#         self.all_texts += text_for_label
#         self.after_loop()
        
        
        
    def after_loop(self):
#         self.all_labels += [ SCAL.RELEVANT_LABEL if Oracle.is_relevant(item) else  SCAL.IRRELEVANT_LABEL 
#                               for item in self.random_sample_from_batch ] 
                  
        # ADD LABELS TO self.random_sample_from_batch
        
        for item in self.random_sample_from_batch:
            if DataItem20NG.RELEVANT_LABEL==self.oracle[item.id_]:
                item.set_relevant()
            else:
                item.set_irrelevant()
        
        self.labeled_collection = list(self.labeled_collection) + list(self.random_sample_from_batch)
      
#         for item,label in zip(self.labeled_collection, self.all_labels):
#             assert label==SCAL.RELEVANT_LABEL or label==SCAL.IRRELEVANT_LABEL
#             label = DataItem.REL_LABEL if label==SCAL.RELEVANT_LABEL else DataItem.IREL_LABEL
#             item.assign_label(label)  
                  
        self.sample_unlabeled_collection = self._remove_from_unlabeled(self.sorted_docs)
 
        self.removed.append([elem for elem in self.sorted_docs ])
                  
        r = len([item for item in self.random_sample_from_batch if item.is_relevant()])
#         new_labels_str = [f'({item.id_},{label})' for label,item in zip(self.all_labels[-self.b:], self.random_sample_from_batch)]
        assert self.b==len(self.random_sample_from_batch)
                  
                  
        Uj = [elem for elem in self.full_U if not elem in set([elem for list_ in self.removed for elem in list_])]
        
        tj = np.min(self.models[self.j].predict([elem for elem in self.full_U if not elem in Uj],\
                                                                item_representation=self.item_representation))

        self.size_of_Uj = len(Uj)
        self.precision_estimates.append(r/self.b)
        self.Rhat[self.j] = (r*self.B)/self.b
        assert (r*self.B)/self.b>=r
        if self.j-1>=0:
            self.Rhat[self.j] += self.Rhat[self.j-1]
        
        self.B += int(np.ceil(self.B/10))
        self.B = min(self.B, len(self.sample_unlabeled_collection))


        self.j+=1
        
#         print(f'j={self.j:2} - B={self.B:5} - b={self.b:2} - len(labeled)={len(self.labeled_collection):6} - '\
#               f'len(unlabeled)={len(self.sample_unlabeled_collection):6} - precision={self.precision_estimates[-1]:4.3f} - '\
#               f'Rhat={self.Rhat[self.j-1]:6.2f} - tj={tj:06.3}')

        

    def finish(self):

        self.prevalecence = (1.05*self.Rhat[self.j-1]) / self.N
        
        no_of_expected_relevant = self.target_recall * self.prevalecence * self.N
        j=0
        while j<len(self.Rhat) and self.Rhat[j]<no_of_expected_relevant:
            j+=1
 
        Uj = self._get_Uj(j)  
        

#         print(f'{self.models[j].predict([elem for elem in self.full_U if not elem in Uj], item_representation=self.item_representation)}')

        
        t = np.min(self.models[j].predict([elem for elem in self.full_U if not elem in Uj], item_representation=self.item_representation))
        self.threshold=t
                  
#         with open(os.path.join(self.home_folder, f'data/labeled_data'+time.strftime("%Y-%m-%d_%H-%M")+'.csv'), 'w') as writer:
#                   writer.write('\n'.join([';'.join([item.id_,item.label]) for item in self.labeled_collection]))
                  
        # FINAL CLASSIFIER
        self.models.append(self._build_classifier(self.labeled_collection))
        
        labeled_ids= {item.id_ for item in self.labeled_collection}

        final_unlabeled_collection = [item for item in self.full_unlabeled_collection if not item.id_ in labeled_ids]

        
        yhat = self.models[-1].predict(final_unlabeled_collection, item_representation=self.item_representation)

        
        relevant = yhat>=t           
#         print(f'Size of predictions={len(relevant)} (relevant={len(relevant[relevant==True])})')


#         print(f'Size of labeled={len(self.labeled_collection)} (relevant='\
#                         f'{len([item for item in self.labeled_collection if item.is_relevant()])})')
#         print(f'Size of unlabeled={len(final_unlabeled_collection)}')
        
#         no_of_synthetic = len([item for item in self.labeled_collection if item.is_synthetic()])
        relevant_data = [item for item in self.labeled_collection if item.is_relevant()]
        confidence = [1.0]*len(relevant_data)
        
        no_of_labeled_rel = len(relevant_data)
        
        relevant_data += [item for item,y in zip(final_unlabeled_collection,yhat) if y>=t]
        confidence +=list([y for item,y in zip(final_unlabeled_collection,yhat) if y>=t])
        
        assert len(relevant_data)==len(confidence)

#         with open(filename, 'w') as writer:
#             writer.write('URL,relevant_or_suggested,confidence\n')
#             count=0
#             for item,confidence_value in zip(relevant_data,confidence):
#                 if count<no_of_labeled_rel:
#                     writer.write(f'https://proquest.com/docview/{item.id_},rel,{confidence_value:4.3f}\n')  
#                 else:
#                     writer.write(f'https://proquest.com/docview/{item.id_},sugg,{confidence_value:4.3f}\n')  
#                 count+=1
                                     
                
        # METRICS
        ytrue = np.array([1 if self.oracle[item.id_]==DataItem20NG.RELEVANT_LABEL else 0  for item in final_unlabeled_collection])
#         print(f'size of ytrue={len(ytrue)} (relevant={len(ytrue[ytrue==1])})')
        acc = accuracy_score(ytrue,yhat>=t)
        prec = precision_score(ytrue, yhat>=t)
        rec = recall_score(ytrue, yhat>=t)
        f1 = f1_score(ytrue, yhat>=t)
        tn, fp, fn, tp = confusion_matrix(ytrue, yhat>=t, labels=[True,False]).ravel()
#         print(f'prevalence  = {self.prevalecence:4.3f}')
#         print(f'accuracy    = {acc:4.3f}')
#         print(f'precision   = {prec:4.3f}')
#         print(f'recall      = {rec:4.3f}')
#         print(f'f1-score    = {f1:4.3f}')
# #         print(f'Rhat        = {self.Rhat}')
#         print(f'*right* j   = {j}')
#         print(f'threshold   = {t}')
            
            
        date=datetime.datetime.now(pytz.timezone('America/Halifax'))
        self.results={'Date':[':'.join(str(date).split(':')[:-2])],
                      'Seed': [self.seed], 
                      'Model': [self.model_type],
#                       'Representation': ['TF-IDF'],
                      'Ranking Function': [self.ranking_function],
                      'Dataset': ['20newsgroup'],   
                      'N': [self.N],
                      'n': [self.n],
                      'Effort': [self._total_effort()],
                      'Accuracy': [acc],
                      'Precision':[prec],
                      'Recall': [rec],
                      'F1-Score':[f1],
                      'Relevant labeled': [len([item for item in self.labeled_collection if item.is_relevant()])],                    
                      'TP': [tp],
                      'TN': [tn],
                      'FP': [fp],
                      'FN': [fn],
                      'Prevalence': [self.prevalecence],
                     }
        return relevant_data, confidence
        

        
        




