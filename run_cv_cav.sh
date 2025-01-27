#!/bin/bash

source .venv/bin/activate

datasets=(cav)
classifiers=("SVM" "XGB" "KNN" "ADA" "RF")

seeds=(0 1 2 3 4 5 6 7 8 9)
splits=(0 1 2 3 4 5 6 7 8 9)

dir_dataset='dataset'
dir_score='cv'
dir_matrix='matrix'

for dataset in "${datasets[@]}"; do
  echo $dataset
  for split in "${splits[@]}"; do
    for classifier in "${classifiers[@]}"; do
        echo $classifier
        for seed in "${seeds[@]}"; do        
            python 05_cv.py --dataset $dataset --seed $seed --dir_score $dir_score --dir_dataset $dir_dataset --dir_matrix $dir_matrix --split $split $classifier
        done
    done
  done
done