import numpy as np
import re
import pandas as pd
import os

from sklearn.model_selection import StratifiedKFold

from src.data_manage import get_Xy
from src.cls_configs import clsf_params, clsf
from src.paths import DIR_CV, DIR_DATASET, DIR_MATRIX, define_cls_folder

def main(args):
    X, y = get_Xy(dir_dataset=args.dir_dataset, dataset=args.dataset, dir_matrix=args.dir_matrix, split=args.split,
                  return_train=True, return_test=False)

    diz_args = vars(args)

    ## Questo blocco serve per fixare il tipo di variabile del parametro max_features del RF
    if args.cls == 'RF':
        if re.search(r'^[0-9]*\.{1}[0-9]+$', diz_args['max_features']): # '0.86' '.98'
            diz_args['max_features'] = float(diz_args['max_features'])
        elif re.search(r'^[0-9]+$', diz_args['max_features']):
            diz_args['max_features'] = int(diz_args['max_features'])

    cls_args = {param:diz_args[param] for param in clsf_params[args.cls]}
    
    scores = pd.Series(index = y.index, dtype=float )
    fnc_cls = clsf[args.cls]

    folder_cls = define_cls_folder(args.cls, cls_args)
    folder_dataset = f'dataset={args.dataset}-split={args.split}'
    csvName = f'seed={args.seed}-fold={args.n_fold}.csv'

    skf = StratifiedKFold(n_splits=args.n_fold, random_state=args.seed, shuffle=True)
    for fold, (train_index, valid_index) in enumerate(skf.split(X, y)):
        cls = fnc_cls(cls_args)
        X_train, y_train = X.iloc[train_index], y.iloc[train_index]
        X_valid = X.iloc[valid_index]

        cls.fit(X_train.values, y_train.values)
        preds = cls.predict_proba(X_valid.values)[:,1]

        scores.iloc[valid_index]=preds
    
    os.makedirs(
        os.path.join(args.dir_score, folder_cls, folder_dataset), 
        exist_ok=True
        )
    
    scores.to_csv(
        os.path.join(args.dir_score, folder_cls, folder_dataset, csvName),
        index=True, header=False, index_label='molID'
    )

if "__main__" == __name__ :
    import argparse

    parser = argparse.ArgumentParser(description='Parameters')

    parser.add_argument('--dataset', type=str, required=True,
                        help='dataset')

    parser.add_argument('--dir_dataset', type=str, required=False, default=DIR_DATASET,
                        help='directory for dataset')

    parser.add_argument('--dir_matrix', type=str, required=False, default=DIR_MATRIX,
                        help='directory for matrix output storage')
    
    parser.add_argument('--dir_score', type=str, required=False, default=DIR_CV,
                        help='directory for score cv output storage')
  
    parser.add_argument('--n_fold', type=int, required=False, default=5,
                        help='number of folds for CV')
    
    parser.add_argument('--seed', type=int, required=False, default=0,
                        help='random state for CV')
    
    parser.add_argument('--split', type=int, required=False, default=0,
                        help='Defined split for train/test dataset')
   
    subparser=parser.add_subparsers(title='Classifier', dest='cls', help='classifier employed for CV')

    for cls_name in clsf_params:
        parser_cls = subparser.add_parser(cls_name)
        for param, param_setup in clsf_params[cls_name].items():
            parser_cls.add_argument(f'--{param}', **param_setup)

    args,unk = parser.parse_known_args()
    if unk:
        print("Unknown arguments passed:", ", ".join(unk))
    
    main(args)
    