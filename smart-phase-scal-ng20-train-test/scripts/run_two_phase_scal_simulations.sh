#!/bin/bash

ENV=imm
REPOSITORY_FOLDER=$(cat /home/ec2-user/SageMaker/mariano/repositories/train-test-split/smart-phase-scal-ng20-train-test/config/repository_path.txt)
OUTPUT_FILE=$REPOSITORY_FOLDER'/results/two_phase_scal_sim_results.csv'

MODEL_TYPE=logreg
REPRESENTATION=bow
RANKING_FUNCTION=relevance
CATEGORIES=$(cat $REPOSITORY_FOLDER'/data/20newsgroup/categories.txt' | xargs)

PYTHON_SCRIPT=$REPOSITORY_FOLDER'/lib/two_phase_scal_simulation.py'

# CHECK IF ENVIRONMENT ACTIVATE, OTHERWISE EXIT WITH ERROR
if [ $(conda info --envs | grep \* | sed 's/^\([^\ ]*\).*/\1/g') != $ENV ]; 
then 
    echo Scripts expect conda environment set "$ENV"
    exit 1
fi
echo STARTING SIMULATION ...


#         5%  10%  20%  25%  50%   75%  100%
for N in 753 1506 3012 3766 7531 11297 15062
do
    for n in 1 3 5 10 20
    do
        for category in $CATEGORIES
        do
            echo [$(date)] N=$N ni=$n category=$category
            python $PYTHON_SCRIPT --N $N \
                                  --category $category \
                                  --n1 $n \
                                  --n2 $n \
                                  --seed $RANDOM \
                                  --target-recall-1 0.9 \
                                  --target-recall-2 0.8 \
                                  --model-type $MODEL_TYPE \
                                  --representation $REPRESENTATION \
                                  --ranking-function $RANKING_FUNCTION \
                                  --result-file $OUTPUT_FILE
        done
    done
done

echo Done!
