###########
# IMPORTS #
###########
import pandas as pd
import numpy as np
import os
import pickle

import sys
repository_path=open('/home/ec2-user/SageMaker/mariano/repositories/train-test-split/two-phase-scal-ng20-train-test/config/repository_path.txt','r').read()
sys.path.append(repository_path)

from lib.dataset import Dataset20NG,DataItem20NG
from lib.scal import SCAL20NG
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix, accuracy_score
import argparse

###########
# PARAMS  #
###########

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='20newsgroup SCAL simulation')
    
    ###########
    # CHOICES #
    ###########
    rf_choices=['relevance', ]
    #              'relevance_with_avg_diversity', 'relevance_with_min_diversity', 'half_relevance_half_uncertainty', 
#              '1quarter_relevance_3quarters_uncertainty', 'random', '3quarter_relevance_1quarters_uncertainty', 'avg_distance', 
#              'uncertainty', 'min_distance', 'uncertainty_with_avg_diversity', 'uncertainty_with_min_diversity']

    category_choices = ['alt.atheism','comp.graphics','comp.os.ms-windows.misc','comp.sys.ibm.pc.hardware','comp.sys.mac.hardware','comp.windows.x','misc.forsale','rec.autos','rec.motorcycles',
             'rec.sport.baseball','rec.sport.hockey','sci.crypt','sci.electronics','sci.med','sci.space','soc.religion.christian','talk.politics.guns','talk.politics.mideast', 
             'talk.politics.misc', 'talk.religion.misc']
    
    parser.add_argument('--N', dest='N', type=int, help='Size of random sample', required=True)
    parser.add_argument('--category', dest='category', choices=category_choices, type=str, help='One of the 20 categories of the 20NG dataset.', required=True)
    
    
    parser.add_argument('--n1', dest='n1', type=int, help='Batch size cap first round', required=True)
    parser.add_argument('--n2', dest='n2', type=int, help='Batch size cap second round', required=True)
    parser.add_argument('--seed', dest='seed', type=int, help='Seed number', required=True)
    parser.add_argument('--target-recall-1', dest='target_recall_1', type=float, help='Target recall for the SCAL process first round', required=True)
    parser.add_argument('--target-recall-2', dest='target_recall_2', type=float, help='Target recall for the SCAL process second round', required=True)
    parser.add_argument('--model-type', dest='model_type', type=str, help='Model type', choices=['svm', 'logreg'], required=True)
    parser.add_argument('--representation',dest='representation', type=str, help='vectors',choices=['bow', 'sbert', 'glove'],required=True)

    parser.add_argument('--result-file',dest='output_file', type=str, help='output vector (append)',required=True)
            
    parser.add_argument('--ranking-function', dest='ranking_function', type=str, 
                        help='Ranking function.', choices=rf_choices, required=True)
    
    args = parser.parse_args()
    

    # BEGIN REPRESENTATIONS
    representation=args.representation
#     representations = Dataset20NG.get_20newsgroup_representations(type_=representation) # CHANGE <<<<<<<<<
    #     representations = DatasetDP.get_DP_representations(type_=representation) # CHANGE <<<<<<<<<
    if representation == 'bow':
        representation_file = os.path.join(repository_path, 'embeddings', 'item_representation_tfidf.pickle')
    elif representation == 'glove':
        representation_file = os.path.join(repository_path, 'embeddings', 'item_representation_glove.pickle')
    else:
        representation_file = os.path.join(repository_path, 'embeddings', 'item_representation_sentence_bert.pickle')
    
    with open(representation_file, 'rb') as reader:
        representations = pickle.load(reader)
        
    if representation == 'bow':
        representations = {key:representations[key].toarray() for key in representations}

    representations = {str(id_):representations[id_] for id_ in representations}
    # END REPRESENTATIONS
    # END REPRESENTATIONS
    category = args.category
    
    oracle = Dataset20NG.get_20newsgroup_oracle(category=category)    
    
    first_round_ni=args.n1
    second_round_ni=args.n2
    Ni=args.N
    first_round_tg=args.target_recall_1
    second_round_tg=args.target_recall_2
    
    model=args.model_type
    ranking_function=args.ranking_function

    seed = args.seed
    
#     print(f'first_round_ni=  {first_round_ni}')
#     print(f'second_round_ni= {second_round_ni}')
#     print(f'Ni=              {Ni}')
#     print(f'first_round_tg=  {first_round_tg}')
#     print(f'second_round_tg= {second_round_tg}')
#     print(f'category=        {category}')
#     print(f'representation=  {representation}')
#     print(f'model=           {model}')
#     print(f'ranking_function={ranking_function}')
#     print(f'seed=            {seed}')


    ##################################################
    #                 TWO-PHASE SCAL                 #
    ##################################################
    # UNLABELED 
#     unlabeled = Dataset20NG.get_20newsgroup_unlabeled_collection()
    train, test = Dataset20NG.get_20newsgroup_unlabeled_collection_train_test(seed=args.seed)
    total_instance_count = len(train)

    # LABELED
    relevants = [item for item in train if oracle[item.id_]==DataItem20NG.RELEVANT_LABEL]
    rng = np.random.default_rng(seed)
    labeled = list(rng.choice(relevants, size=1))
    for item in labeled:
        item.set_relevant()
    labeled_ids = {item.id_ for item in labeled}

    # REMOVING NEWLY LABELED FROM UNLABELED
    train = [item for item in train if not item.id_ in labeled_ids]

    assert len(train) == (total_instance_count-len(labeled))

    scal_model = SCAL20NG(session_name='two round scal',
                          labeled_collection=labeled,
                          unlabeled_collection=train,
                          batch_size_cap=first_round_ni,
                          random_sample_size=Ni,
                          target_recall=first_round_tg,
                          ranking_function=ranking_function,
                          item_representation=representations,
                          oracle=oracle,
                          model_type=model,
                          seed=seed)

    results_1st_round = scal_model.run()
    results_1st_round['category']=[category]
    results_1st_round['target-recall']=[first_round_tg]
    results_1st_round['SCAL-round-no']=[1]
    results_1st_round['SCAL type']=['double scal']
    results_1st_round['threshold']=[scal_model.threshold]

    #######################
    # TEST SET EVALUATION #
    #######################
    scores = scal_model.models[-1].predict(test,item_representation=representations)
    final_suggestions = [item for item,score in zip(test,scores) if score>scal_model.threshold]

    final_suggestions_ids=set([item.id_ for item in final_suggestions])
    
    ytrue = [oracle[elem.id_]=='R' for elem in test]
    ypred = [elem.id_ in final_suggestions_ids for elem in test]


    tn, fp, fn, tp = confusion_matrix(ytrue, ypred, labels=[True,False]).ravel()
    
    results_1st_round['Accuracy (test)']=[accuracy_score(ytrue,ypred)]
    results_1st_round['Precision (test)']=[precision_score(ytrue,ypred)]
    results_1st_round['Recall (test)']=[recall_score(ytrue,ypred)]
    results_1st_round['F1-Score (test)']=[f1_score(ytrue,ypred)]
    results_1st_round['TP (test)']=[tp]
    results_1st_round['FP (test)']=[fp]
    results_1st_round['TN (test)']=[tn]
    results_1st_round['FN (test)']=[fn]

    assert results_1st_round['Effort'][0]+1==len(scal_model.labeled_collection)


    

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

    scal_model = SCAL20NG(session_name='two round scal(b)',
                          labeled_collection=new_labeled,
                          unlabeled_collection=new_unlabeled,
                          batch_size_cap=second_round_ni,
                          random_sample_size=Ni,
                          target_recall=second_round_tg,
                          ranking_function=ranking_function,
                          item_representation=representations,
                          oracle=oracle,
                          model_type=model,
                          seed=seed)


    results_2nd_round = scal_model.run()
    results_2nd_round['category']=[category]
    results_2nd_round['target-recall']=[second_round_tg]
    results_2nd_round['SCAL-round-no']=[2]
    results_2nd_round['SCAL type']=['double scal']
    results_2nd_round['threshold']=[scal_model.threshold]
    assert results_2nd_round['Effort'][0]+ results_1st_round['Effort'][0]+1==len(scal_model.labeled_collection)

    labeled_ids=set([item.id_ for item in scal_model.labeled_collection])


#     possible_suggestions = [item for item in new_unlabeled if not item.id_ in labeled_ids]
#     scores = scal_model.models[-1].predict(possible_suggestions,item_representation=representations)


#     final_suggestions = [item for item,score in zip(possible_suggestions,scores) if score>scal_model.threshold]





#     final_unlabeled = [item for item in Dataset20NG.get_20newsgroup_unlabeled_collection() if not item.id_ in labeled_ids]

    #######################
    # TEST SET EVALUATION #
    #######################
    scores = scal_model.models[-1].predict(test,item_representation=representations)
    final_suggestions = [item for item,score in zip(test,scores) if score>scal_model.threshold]

    final_suggestions_ids=set([item.id_ for item in final_suggestions])
    
    ytrue = [oracle[elem.id_]=='R' for elem in test]
    ypred = [elem.id_ in final_suggestions_ids for elem in test]


    tn, fp, fn, tp = confusion_matrix(ytrue, ypred, labels=[True,False]).ravel()


    results_2nd_round['Accuracy (test)']=[accuracy_score(ytrue,ypred)]
    results_2nd_round['Precision (test)']=[precision_score(ytrue,ypred)]
    results_2nd_round['Recall (test)']=[recall_score(ytrue,ypred)]
    results_2nd_round['F1-Score (test)']=[f1_score(ytrue,ypred)]
    results_2nd_round['TP (test)']=[tp]
    results_2nd_round['FP (test)']=[fp]
    results_2nd_round['TN (test)']=[tn]
    results_2nd_round['FN (test)']=[fn]
    
#     print('--- TEST RESULTS SCAL SECOND ROUND ---')
#     print(f'Accuracy  = {accuracy_score(ytrue,ypred):4.3f}')
#     print(f'Precision = {precision_score(ytrue,ypred):4.3f}')
#     print(f'Recall    = {recall_score(ytrue,ypred):4.3f}')
#     print(f'F1-Score  = {f1_score(ytrue,ypred):4.3f}')
#     print(f'TP        = {tp}')
#     print(f'FP        = {fp}')
#     print(f'TN        = {tn}')
#     print(f'FN        = {fn}')
    
    ###########################################################
    #                 SINGLE-PHASE SCAL    (FIRST)  (ni)      #
    ###########################################################
    def get_effort(N, n):
        B=1
        it=1
        len_unlabeled=N
        effort=0
        while len_unlabeled>0:
            b=min(B,n)
            effort+=min(len_unlabeled, b)
            len_unlabeled-=B
            B+=int(np.ceil(B/10))
            it+=1
        return effort

    total_effort=results_1st_round['Effort'][0]+results_2nd_round['Effort'][0]

    ni = first_round_ni

    assert get_effort(Ni, ni)<total_effort

    while get_effort(Ni, ni)<total_effort and get_effort(Ni,ni)<Ni:
        ni+=1

    #############################
    # RUN SCAL WITH ni and ni-1 #
    #############################

    # UNLABELED 
    unlabeled = Dataset20NG.get_20newsgroup_unlabeled_collection()
    total_instance_count=len(unlabeled)


    # LABELED
    relevants = [item for item in unlabeled if oracle[item.id_]==DataItem20NG.RELEVANT_LABEL]
    rng = np.random.default_rng(2022)
    labeled = list(rng.choice(relevants, size=1))
    for item in labeled:
        item.set_relevant()
    labeled_ids = {item.id_ for item in labeled}

    # REMOVING NEWLY LABELED FROM UNLABELED
    unlabeled = [item for item in unlabeled if not item.id_ in labeled_ids]

    assert len(unlabeled)==len(Dataset20NG.get_20newsgroup_unlabeled_collection())-len(labeled)

    scal_model = SCAL20NG(session_name='two round scal',
                          labeled_collection=labeled,
                          unlabeled_collection=unlabeled,
                          batch_size_cap=ni,
                          random_sample_size=Ni,
                          target_recall=second_round_tg,
                          ranking_function=ranking_function,
                          item_representation=representations,
                          oracle=oracle,
                          model_type=model,
                          seed=seed)


    results_single_round = scal_model.run()
    results_single_round['category']=[category]
    results_single_round['target-recall']=[second_round_tg]
    results_single_round['SCAL-round-no']=[1]
    results_single_round['SCAL type']=['single scal']
    results_single_round['threshold']=[scal_model.threshold]







    #######################
    # TEST SET EVALUATION #
    #######################
    scores = scal_model.models[-1].predict(test,item_representation=representations)
    final_suggestions = [item for item,score in zip(test,scores) if score>scal_model.threshold]
    
    final_suggestions_ids=set([item.id_ for item in final_suggestions])
    
    ytrue = [oracle[elem.id_]=='R' for elem in test]
    ypred = [elem.id_ in final_suggestions_ids for elem in test]


    tn, fp, fn, tp = confusion_matrix(ytrue, ypred, labels=[True,False]).ravel()
    
    results_single_round['Accuracy (test)']=[accuracy_score(ytrue,ypred)]
    results_single_round['Precision (test)']=[precision_score(ytrue,ypred)]
    results_single_round['Recall (test)']=[recall_score(ytrue,ypred)]
    results_single_round['F1-Score (test)']=[f1_score(ytrue,ypred)]
    results_single_round['TP (test)']=[tp]
    results_single_round['FP (test)']=[fp]
    results_single_round['TN (test)']=[tn]
    results_single_round['FN (test)']=[fn]
    
    results={}

    for key in results_1st_round:
        results[key]=results_1st_round[key]+results_2nd_round[key]+results_single_round[key]

    results['Total effort']=[results['Effort'][0],results['Effort'][0]+results['Effort'][1], results_single_round['Effort'][0] ]

#     results_2nd_round['Accuracy (corrected)']=[accuracy_score(ytrue,ypred)]
#     results_2nd_round['Precision (corrected)']=[precision_score(ytrue,ypred)]
#     results_2nd_round['Recall (corrected)']=[recall_score(ytrue,ypred)]
#     results_2nd_round['F1-Score (corrected)']=[f1_score(ytrue,ypred)]
#     results_2nd_round['TP (corr)']=[tp]
#     results_2nd_round['FP (corr)']=[fp]
#     results_2nd_round['TN (corr)']=[tn]
#     results_2nd_round['FN (corr)']=[fn]
    
#     print('--- TEST RESULTS SCAL ONE ROUND ---')
#     print(f'Accuracy  = {accuracy_score(ytrue,ypred):4.3f}')
#     print(f'Precision = {precision_score(ytrue,ypred):4.3f}')
#     print(f'Recall    = {recall_score(ytrue,ypred):4.3f}')
#     print(f'F1-Score  = {f1_score(ytrue,ypred):4.3f}')
#     print(f'TP        = {tp}')
#     print(f'FP        = {fp}')
#     print(f'TN        = {tn}')
#     print(f'FN        = {fn}')
    

    ##############################################################
    #                 SINGLE-PHASE SCAL    (SECOND)  (ni-1)      #
    ##############################################################

    # UNLABELED 
    unlabeled = Dataset20NG.get_20newsgroup_unlabeled_collection()
    total_instance_count=len(unlabeled)


    # LABELED
    relevants = [item for item in unlabeled if oracle[item.id_]==DataItem20NG.RELEVANT_LABEL]
    rng = np.random.default_rng(2022)
    labeled = list(rng.choice(relevants, size=1))
    for item in labeled:
        item.set_relevant()
    labeled_ids = {item.id_ for item in labeled}

    # REMOVING NEWLY LABELED FROM UNLABELED
    unlabeled = [item for item in unlabeled if not item.id_ in labeled_ids]

    assert len(unlabeled)==len(Dataset20NG.get_20newsgroup_unlabeled_collection())-len(labeled)

    scal_model = SCAL20NG(session_name='two round scal',
                          labeled_collection=labeled,
                          unlabeled_collection=unlabeled,
                          batch_size_cap=ni-1,
                          random_sample_size=Ni,
                          target_recall=second_round_tg,
                          ranking_function=ranking_function,
                          item_representation=representations,
                          oracle=oracle,
                          model_type=model,
                          seed=seed)


    results_single_round = scal_model.run()
    results_single_round['category']=[category]
    results_single_round['target-recall']=[second_round_tg]
    results_single_round['SCAL-round-no']=[1]
    results_single_round['SCAL type']=['single scal']
    results_single_round['threshold']=[scal_model.threshold]





    
    #######################
    # TEST SET EVALUATION #
    #######################
    scores = scal_model.models[-1].predict(test,item_representation=representations)
    final_suggestions = [item for item,score in zip(test,scores) if score>scal_model.threshold]
    final_suggestions_ids=set([item.id_ for item in final_suggestions])

    ytrue = [oracle[elem.id_]=='R' for elem in test]
    ypred = [elem.id_ in final_suggestions_ids for elem in test]


    tn, fp, fn, tp = confusion_matrix(ytrue, ypred, labels=[True,False]).ravel()

    results_single_round['Accuracy (test)']=[accuracy_score(ytrue,ypred)]
    results_single_round['Precision (test)']=[precision_score(ytrue,ypred)]
    results_single_round['Recall (test)']=[recall_score(ytrue,ypred)]
    results_single_round['F1-Score (test)']=[f1_score(ytrue,ypred)]
    results_single_round['TP (test)']=[tp]
    results_single_round['FP (test)']=[fp]
    results_single_round['TN (test)']=[tn]
    results_single_round['FN (test)']=[fn]
    
    for key in results_single_round:
        results[key].append(results_single_round[key][0])
    results['Total effort'].append(results_single_round['Effort'][0])
    
    new_df = pd.DataFrame(results)
    
#     results_2nd_round['Accuracy (corrected)']=[accuracy_score(ytrue,ypred)]
#     results_2nd_round['Precision (corrected)']=[precision_score(ytrue,ypred)]
#     results_2nd_round['Recall (corrected)']=[recall_score(ytrue,ypred)]
#     results_2nd_round['F1-Score (corrected)']=[f1_score(ytrue,ypred)]
#     results_2nd_round['TP (corr)']=[tp]
#     results_2nd_round['FP (corr)']=[fp]
#     results_2nd_round['TN (corr)']=[tn]
#     results_2nd_round['FN (corr)']=[fn]
    
#     print('--- TEST RESULTS SCAL ONE ROUND (less effort) ---')
#     print(f'Accuracy  = {accuracy_score(ytrue,ypred):4.3f}')
#     print(f'Precision = {precision_score(ytrue,ypred):4.3f}')
#     print(f'Recall    = {recall_score(ytrue,ypred):4.3f}')
#     print(f'F1-Score  = {f1_score(ytrue,ypred):4.3f}')
#     print(f'TP        = {tp}')
#     print(f'FP        = {fp}')
#     print(f'TN        = {tn}')
#     print(f'FN        = {fn}')
    
    if os.path.isfile(args.output_file):
        df = pd.read_csv(args.output_file)
        df = df.append(new_df)
        df.to_csv(args.output_file, index=False)
    else:
        new_df.to_csv(args.output_file, index=False,)

    
    