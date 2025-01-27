import numpy as np
import re
import pandas as pd
import os
from src.cls_configs import clsf, clsf_params
from src.paths import DIR_TEST, DIR_DATASET, DIR_MATRIX, define_cls_folder
from src.data_manage import get_Xy

def main(args):
    X_train, y_train, X_test, y_test = get_Xy(dir_dataset=args.dir_dataset, dataset=args.dataset, dir_matrix=args.dir_matrix, split=args.split,
                  return_train=True, return_test=True)

    diz_args = vars(args)

    # These lines fix input type for RF max_feature parameter
    if args.cls == 'RF':
        if re.search(r'^[0-9]*\.{1}[0-9]+$', diz_args['max_features']): # '0.86' '.98'
            diz_args['max_features'] = float(diz_args['max_features'])
        elif re.search(r'^[0-9]+$', diz_args['max_features']):
            diz_args['max_features'] = int(diz_args['max_features'])

    cls_args = {param:diz_args[param] for param in clsf_params[args.cls]}
    
    scores = pd.Series(index = y_test.index, dtype=float )
    fnc_cls = clsf[args.cls]

    folder_cls = define_cls_folder(args.cls, cls_args)
    file_dataset = f'dataset={args.dataset}-split={args.split}.csv'

    ## cls=RF-max_features=100-n_estimators=200/dataset=cav-split=1.csv

    cls = fnc_cls(cls_args)
    cls.fit(X_train.values, y_train.values)

    preds = cls.predict_proba(X_test.values)[:,1]

    scores.iloc[:]=preds
    
    os.makedirs(
        os.path.join(args.dir_score, folder_cls), 
        exist_ok=True
        )
    
    scores.to_csv(
        os.path.join(args.dir_score, folder_cls, file_dataset ),
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
    
    parser.add_argument('--dir_score', type=str, required=False, default=DIR_TEST,
                        help='directory for score cv output storage')
    
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
    