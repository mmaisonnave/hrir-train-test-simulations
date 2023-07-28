###########
# IMPORTS #
###########
import pandas as pd
import numpy as np
import os

import pickle
import sys
# repository_path=open('/home/ec2-user/SageMaker/mariano/repositories/train-test-split/smart-phase-scal-dp-train-test/config/repository_path.txt','r').read()

with open('../config/repository_path.txt', 'r') as reader:
    repository_path = reader.read().strip()
    
sys.path.append(repository_path)

from lib.dataset import DatasetDP,DataItemDP
from lib.scal import SCALDP
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix, accuracy_score
import argparse

###########
# PARAMS  #
###########

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='DP SCAL simulation')
    
    ###########
    # CHOICES #
    ###########
    rf_choices=['relevance', ]
    #              'relevance_with_avg_diversity', 'relevance_with_min_diversity', 'half_relevance_half_uncertainty', 
#              '1quarter_relevance_3quarters_uncertainty', 'random', '3quarter_relevance_1quarters_uncertainty', 'avg_distance', 
#              'uncertainty', 'min_distance', 'uncertainty_with_avg_diversity', 'uncertainty_with_min_diversity']


    
    parser.add_argument('--N', dest='N', type=int, help='Size of random sample', required=True)
    
    
    parser.add_argument('--n', dest='n', type=int, help='Batch size cap first round', required=True)
    parser.add_argument('--seed', dest='seed', type=int, help='Seed number', required=True)
    parser.add_argument('--target-recall', dest='target_recall', type=float, help='Target recall for the SCAL process first round', required=True)
    parser.add_argument('--model-type', dest='model_type', type=str, help='Model type', choices=['svm', 'logreg'], required=True)
    parser.add_argument('--smart', dest='smart', type=bool, help='smart or regular two phase', required=True)
    parser.add_argument('--representation',dest='representation', type=str, help='vectors',choices=['bow', 'sbert', 'glove'],required=True)

    parser.add_argument('--result-file',dest='output_file', type=str, help='output vector (append)',required=True)
            
    parser.add_argument('--ranking-function', dest='ranking_function', type=str, 
                        help='Ranking function.', choices=rf_choices, required=True)
    
    args = parser.parse_args()
    


    representation=args.representation
    if representation == 'bow':
        representation_file = os.path.join(repository_path, 'embeddings', 'item_representation_tfidf.pickle')
    elif representation == 'glove':
        representation_file = os.path.join(repository_path, 'embeddings', 'item_representation_glove.pickle')
    else:
        representation_file = os.path.join(repository_path, 'embeddings', 'item_representation_sentence_bert.pickle')
    
    with open(representation_file, 'rb') as reader:
        representations = pickle.load(reader)
    
    representations = {str(id_):representations[id_] for id_ in representations}
    
    oracle = DatasetDP.get_DP_oracle()    

    n=args.n
    Ni=args.N
    smart=args.smart
    target_recall=args.target_recall
    
    model=args.model_type
    ranking_function=args.ranking_function

    seed = args.seed
    
#     print(f'first_round_ni=  {first_round_ni}')
#     print(f'second_round_ni= {second_round_ni}')
#     print(f'Ni=              {Ni}')
#     print(f'first_round_tg=  {first_round_tg}')
#     print(f'second_round_tg= {second_round_tg}')
#     print(f'representation=  {representation}')
#     print(f'model=           {model}')
#     print(f'ranking_function={ranking_function}')
#     print(f'seed=            {seed}')


    ##################################################
    #                 TWO-PHASE SCAL                 #
    ##################################################
    # UNLABELED 
#     unlabeled = DatasetDP.get_DP_unlabeled_collection()
    train, test = DatasetDP.get_DP_unlabeled_collection_train_test(seed=args.seed)
    total_instance_count = len(train)

    # LABELED
    relevants = [item for item in train if oracle[item.id_]==DataItemDP.RELEVANT_LABEL]
    rng = np.random.default_rng(seed)
    labeled = list(rng.choice(relevants, size=1))
    for item in labeled:
        item.set_relevant()
    labeled_ids = {item.id_ for item in labeled}

    # REMOVING NEWLY LABELED FROM UNLABELED
    train = [item for item in train if not item.id_ in labeled_ids]

    assert len(train) == (total_instance_count-len(labeled))

    scal_model = SCALDP(session_name='two round scal',
                          labeled_collection=labeled,
                          unlabeled_collection=train,
                          batch_size_cap=n,
                          random_sample_size=Ni,
                          target_recall=target_recall,
                          ranking_function=ranking_function,
                          item_representation=representations,
                          oracle=oracle,
                          smart_start=smart,
                          model_type=model,
                          seed=seed)

    results = scal_model.run()
    results['target-recall']=[target_recall]
    results['SCAL-round-no']=[1]
    results['SCAL type']=['smart scal' if smart else 'double scal']
    results['threshold']=[scal_model.threshold]

    #######################
    # TEST SET EVALUATION #
    #######################
    scores = scal_model.models[-1].predict(test,item_representation=representations)
    final_suggestions = [item for item,score in zip(test,scores) if score>scal_model.threshold]

    final_suggestions_ids=set([item.id_ for item in final_suggestions])
    
    ytrue = [oracle[elem.id_]=='R' for elem in test]
    ypred = [elem.id_ in final_suggestions_ids for elem in test]


    tn, fp, fn, tp = confusion_matrix(ytrue, ypred,).ravel()
    
    results['Accuracy (held-out)']=[accuracy_score(ytrue,ypred)]
    results['Precision (held-out)']=[precision_score(ytrue,ypred)]
    results['Recall (held-out)']=[recall_score(ytrue,ypred)]
    results['F1-Score (held-out)']=[f1_score(ytrue,ypred)]
    results['TP (held-out)']=[tp]
    results['FP (held-out)']=[fp]
    results['TN (held-out)']=[tn]
    results['FN (held-out)']=[fn]
    
    results['Representation']=[representation]

    assert results['Effort'][0]+1==len(scal_model.labeled_collection)


    

    new_labeled = scal_model.labeled_collection
    labeled_ids=set([item.id_ for item in new_labeled])

    yhat = scal_model.models[-1].predict(train,item_representation=representations)
    suggestions = [item for item,score in zip(train,yhat) if score>scal_model.threshold if not item.id_ in labeled_ids]

    new_unlabeled = suggestions

    
    effort=scal_model._total_effort()
    ########################
    # SECOND ROUND LABELED #
    ########################
    assert scal_model.models[-1].trained
    new_df = pd.DataFrame(results)


    full_path_output= os.path.join(repository_path,'results',args.output_file)
    if os.path.isfile(full_path_output):
        df = pd.read_csv(full_path_output)
        df = df.append(new_df)
        df.to_csv(full_path_output, index=False)
    else:
        new_df.to_csv(full_path_output, index=False,)

    
    
