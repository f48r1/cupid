import pandas as pd
import numpy as np
from scipy import stats
# import math

from src.smiles_elaboration import ElabSmiles

def sigma (va):
    va=np.log10(np.array(va))
    return stats.tstd(va)

### Funzione che filtra con il procedimento "vecchio"
## non tiene conto dei duplicati per molregno o/e smiles
def filteringDF(df):
    df['elab_smiles']=df['canonical_smiles'].apply(ElabSmiles)
    df["avg"]=9 - df["va"].apply(lambda x: stats.tmean(np.log10(np.array(x))))
    df["sigma"]=df["va"].apply(sigma)
    df["log_va"]=df["va"].apply(lambda x: np.log10(np.array(x)).astype(float))

    maskSigma=(df['sigma']<1) | (pd.isna(df['sigma']))
    maskAvg=df['avg']!=5

    df = df[(maskSigma) & (maskAvg)]

    return df

## se presente piu di 1 organismo includendo Homo sapiens, ritiene solo i valori inerenti al saggio su Umano
def keepHomoSapiens(row):

    if len(row['organism'])>1 and 'Homo sapiens' in row['organism']:
        idxHomo = row['organism'].index('Homo sapiens')
        row['va']=(row['va'][idxHomo],)
        row['organism']=(row['organism'][idxHomo],)

    return row

from itertools import chain

## pulisce il dato raggruppando prima per molregno e poi smiles
def keepUnique(df):
    print ('iniziali: ', len(df))
    grpMolr = df.groupby('molregno', as_index=False).agg({'va':tuple, 'organism':tuple})
    print ('doppioni per molregno:', len(grpMolr))
    grpMolr = grpMolr.apply(keepHomoSapiens, axis=1)
    grpMolr['va']=grpMolr['va'].apply(lambda x: list(chain(*x)))

    grpMolr["log_va"]=grpMolr["va"].apply(lambda x: np.log10(np.array(x)))
    grpMolr["avg"]=9 - grpMolr["log_va"].apply(stats.tmean)
    grpMolr["sigma"]=grpMolr["log_va"].apply(stats.tstd)

    maskSigma=(grpMolr['sigma']<1) | (pd.isna(grpMolr['sigma']))
    # maskAvg=(grpMolr['avg']!=5) & ~(grpMolr['avg']==np.inf)
    maskAvg= ~(grpMolr['avg']==np.inf)
    grpMolr = grpMolr[(maskSigma) & (maskAvg)]
    print ('rimozione per sigma ed infiniti:', len(grpMolr))

    grpMolr=grpMolr.merge(df[['molregno','canonical_smiles','pref_name']], on='molregno', how='inner').drop_duplicates('molregno')
    grpMolr['elab_smiles']=grpMolr['canonical_smiles'].apply(ElabSmiles)

    grpSmi = grpMolr.groupby('elab_smiles',as_index=False).agg({'avg':tuple, 'molregno':tuple, 
                                                               'organism':lambda x: tuple(chain(*x)),
                                                               'pref_name':lambda x: tuple(set(x)),
                                                               'canonical_smiles':tuple,}
                                                               )
    print ('check elab_smiles:', len(grpSmi))
    grpSmi = grpSmi.query('elab_smiles != "mixture" ')
    print ('dopo rimozione mixture:', len(grpSmi))
    grpSmi['sigma']=grpSmi['avg'].apply(stats.tstd)
    grpSmi['avg2']=grpSmi['avg'].apply(stats.tmean)

    count0=(grpSmi['avg2']<5).sum()
    count1=(grpSmi['avg2']>5).sum()

    if count0 < count1:
        grpSmi['label']=grpSmi['avg2'].apply(lambda x: 0 if x <= 5 else 1)
    elif count0 > count1:
        grpSmi['label']=grpSmi['avg2'].apply(lambda x: 0 if x < 5 else 1)
    else:
        print('Not expected case ! Check here')
        return None

    return grpSmi


if __name__ == '__main__':
    import argparse
    import os
    import pandas as pd

    parser = argparse.ArgumentParser(description='Parameters to initialize raw dataset')
    from cls_configs import DIR_DATASET, DIR_RAWDATASET

    parser.add_argument('--dataset', type=str, required=True,
                        help='dataset')

    parser.add_argument('--dir_rawdataset', type=str, required=False, default=DIR_RAWDATASET,
                        help='directory for raw dataset files')

    parser.add_argument('--dir_dataset', type=str, required=False, default=DIR_DATASET,
                        help='directory for dataset files')
    
    args,unk = parser.parse_known_args()
    if unk:
        print("Unknown arguments passed:", ", ".join(unk))

    os.makedirs(
        os.path.join(args.dir_dataset), 
        exist_ok=True
        )
    
    from ast import literal_eval
    
    rawdata = pd.read_csv(os.path.join(args.dir_rawdataset,args.dataset) + '.csv', 
        converters={'va':literal_eval})

    cured_data = keepUnique(rawdata)

    cured_data.to_csv(os.path.join(args.dir_dataset, args.dataset) + '.csv', index=False)