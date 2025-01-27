from src.paths import DIR_DATASET, DIR_MATRIX
from compchemkit import fingerprints
import numpy as np
import pandas as pd
import os

def main(args):

    SMARTS=np.loadtxt("SMARTS_LIST_NEW.csv", dtype=str, comments=None)
    CSFP = fingerprints.FragmentFingerprint(substructure_list=SMARTS.tolist())

    pathData = os.path.join(args.dir_dataset, args.dataset)
    smiles = pd.read_csv(pathData+'.csv', usecols=['elab_smiles']).squeeze()
    
    matrix = CSFP.transform_smiles(smiles)

    matrix=pd.DataFrame.sparse.from_spmatrix(matrix)
    
    pathMatrix = os.path.join(args.dir_matrix, args.dataset)
    os.makedirs(args.dir_matrix, exist_ok=True)

    matrix.to_csv(pathMatrix + '.csv', index=False, header=False)

    
if "__main__" == __name__ :
    import argparse

    parser = argparse.ArgumentParser(description='Parameters')

    parser.add_argument('--dataset', type=str, required=True,
                        help='dataset')

    parser.add_argument('--dir_dataset', type=str, required=False, default=DIR_DATASET,
                        help='directory for dataset')

    parser.add_argument('--dir_matrix', type=str, required=False, default=DIR_MATRIX,
                        help='directory for matrix output storage')

    
    args,unk = parser.parse_known_args()
    if unk:
        print("Unknown arguments passed:", ", ".join(unk))
    
    main(args)