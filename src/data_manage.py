import pandas as pd
import os
from .paths import DIR_DATASET, DIR_MATRIX, DIR_PICKLE
from .target_configs import DATASET
import pickle

def get_Xy(dataset, split, dir_dataset=DIR_DATASET, dir_matrix=DIR_MATRIX,
           return_train=True, return_test=False):

    path_data = os.path.join(dir_dataset, dataset + '.csv')
    path_matrix = os.path.join(dir_matrix, dataset + '.csv')

    X = pd.read_csv(path_matrix, header=None)

    ## ritorna il dataset intero senza che ci sia nessuno split
    if (not return_train and not return_test) or split == -1:
        y = pd.read_csv(path_data, usecols=['label']).squeeze()
        return X,y

    data = pd.read_csv(path_data, usecols=['label',f'split{split}'])
    if return_train:
        query_str = f'split{split} == "train" '
        y_train = data.query(query_str).drop(columns=f'split{split}').squeeze()
        X_train = X.loc[y_train.index]
        y_train = y_train

    if return_test:
        query_str = f'split{split} == "test" '
        y_test = data.query(query_str).drop(columns=f'split{split}').squeeze()
        X_test = X.loc[y_test.index]
        y_test = y_test

    if return_train and return_test:
        return X_train,y_train,   X_test,y_test
    elif return_train:
        return X_train,y_train
    elif return_test:
        return X_test,y_test

def get_all_labels(dir_dataset=DIR_DATASET):
    labels_dataset = {
    data: pd.read_csv(
        os.path.join(dir_dataset, data +'.csv'),
        usecols=['label']
        ).squeeze()
    for data in DATASET # TODO replace this list with a for cycle to recognize csv files
}
    return labels_dataset

def get_smiles_train(dataset, dir_dataset = DIR_DATASET):
    smiles_train= pd.read_csv(
        os.path.join(dir_dataset, dataset +'.csv'),
        usecols=['elab_smiles']
        ).squeeze()
    
    return smiles_train