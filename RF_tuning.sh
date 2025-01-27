#!/bin/bash

source .venv/bin/activate

# Definisci i nomi dei dataset, i classificatori e i seed
datasets=("cav" "nav" "erg")
seeds=(0 1 2 3 4 5 6 7 8 9)
splits=(0 1 2 3 4 5 6 7 8 9)
n_estimators_set=(100 150 200)
max_features_set=(10 50)

for dataset in "${datasets[@]}"; do
  for split in "${splits[@]}"; do
    for seed in "${seeds[@]}"; do
      for n_estimators in "${n_estimators_set[@]}"; do
        for max_features in "${max_features_set[@]}"; do
            python 05_cv.py --dataset $dataset --seed $seed --split $split RF --n_estimators $n_estimators --max_features $max_features
        done
      done
    done
  done
done
