import pandas as pd

DATASET = ['erg','nav','cav']

TARGET_NAMES = {
    'nav':'Naᵥ1.5',
    'cav':'Caᵥ1.2',
    'erg':'ERG'
}

# [ ] set tids as tuple of integers and then fix rawdataset initialization
TIDS = {
    'cav':'169,10513,10514,17056,106600',
    'nav':'11480,12606',
    'erg':'165,101236',
}

TARGET_DF = pd.DataFrame(
          [TARGET_NAMES, TIDS],
    index=['name',      'tid'],
    columns=DATASET,
)