#!/bin/bash

ENV=imm
REPOSITORY_FOLDER=$(cat /home/ec2-user/SageMaker/mariano/repositories/train-test-split/ranking_function_ng_simulations/config/repository_path.txt)
OUTPUT_FILE='ranking_function_simulations.csv'

CATEGORIES_FILE=$REPOSITORY_FOLDER/files/categories.txt

PYTHON_SCRIPT=$REPOSITORY_FOLDER'/lib/ranking_function_simulations.py'

# CHECK IF ENVIRONMENT ACTIVATE, OTHERWISE EXIT WITH ERROR
if [ $(conda info --envs | grep \* | sed 's/^\([^\ ]*\).*/\1/g') != $ENV ]; 
then 
    echo Scripts expect conda environment set "$ENV"
    exit 1
fi
echo STARTING SIMULATION ...


# NO_OF_SEEDS=3
MODEL_TYPE=logreg
REPRESENTATION=sbert
#         5% 10%  20%  25%  50%  75% 100%
for N in 291 583 1165 1456 2913 4369 5825
do
    for n in 1 3 5 10 20
    do
        for RANKING_FUNCTION in relevance uncertainty half_and_half random
        do
#             for ((repetition=1; repetition<=$NO_OF_SEEDS;repetition++))
            for CATEGORY in $(cat $CATEGORIES_FILE)
            do
                SEED=$RANDOM
                echo $(date) N=$N    n=$n    Representation=$REPRESENTATION Category=$CATEGORY   SEED=$SEED \(repetition=$repetition\)
                python $PYTHON_SCRIPT --N $N \
                                      --n $n \
                                      --seed $SEED \
                                      --category $CATEGORY \
                                      --smart False \
                                      --target-recall 0.8 \
                                      --model-type $MODEL_TYPE \
                                      --representation $REPRESENTATION \
                                      --ranking-function $RANKING_FUNCTION \
                                      --result-file $OUTPUT_FILE \
                                      
            done
        done
    done
done
         

#         5% 10%  20%  25%  50%  75% 100%
# for N in 291 583 1165 1456 2913 4369 5825
# do
#     for n in 1 3 5 10 20
#     do
#         for s in {1..5}
#         do
#             SEED=$RANDOM
#             echo [$(date)] N=$N ni=$n seed=$SEED
#             python $PYTHON_SCRIPT --N $N \
#                                   --n1 $n \
#                                   --n2 $n \
#                                   --seed $SEED \
#                                   --target-recall-1 0.9 \
#                                   --target-recall-2 0.8 \
#                                   --model-type $MODEL_TYPE \
#                                   --representation $REPRESENTATION \
#                                   --ranking-function $RANKING_FUNCTION \
#                                   --result-file $OUTPUT_FILE
#         done
#     done
# done

echo Done!
