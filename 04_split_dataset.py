import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from src.paths import DIR_DATASET

def main(args):

    path_data = os.path.join(args.dir_dataset, args.dataset)
    data = pd.read_csv(path_data + '.csv')

    label = data['label']
    table_split = pd.DataFrame(index=label.index)

    for split in range(args.n_split):
        table_split[f'split{split}']=None
        train_idxs, test_idxs = train_test_split(label.index, test_size=.2, random_state=split, stratify=label)

        table_split.iloc[train_idxs,split]='train'
        table_split.iloc[test_idxs,split]='test'

    new_data = data.join(table_split)

    new_data.to_csv(path_data+'.csv', index=False)
    
if "__main__" == __name__ :
    import argparse

    parser = argparse.ArgumentParser(description='Parameters')

    parser.add_argument('--dataset', type=str, required=True,
                        help='dataset')

    parser.add_argument('--dir_dataset', type=str, required=False, default=DIR_DATASET,
                        help='directory for dataset')

    parser.add_argument('--n_split', type=int, required=False, default=10,
                        help='Number of splitting from initial dataset')

    args,unk = parser.parse_known_args()
    if unk:
        print("Unknown arguments passed:", ", ".join(unk))
    
    main(args)