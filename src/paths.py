import os

DIR_ROOT = './'
DIR_RAWDATASET = os.path.join(DIR_ROOT,'raw_dataset')
DIR_DATASET = os.path.join(DIR_ROOT,'dataset')
DIR_CV = os.path.join(DIR_ROOT,'cv')
DIR_TEST = os.path.join(DIR_ROOT,'preds')
DIR_MATRIX = os.path.join(DIR_ROOT,'matrix')
DIR_PICKLE = os.path.join(DIR_ROOT,'pickle')
DIR_IMGS = os.path.join(DIR_ROOT,'imgSMARTS')
DIR_SMARTS = os.path.join(DIR_ROOT,'SMARTS_LIST_NEW.csv')

def define_cls_folder(cls_name:str, cls_args:dict):
    strings=[f'cls={cls_name}']

    for key in sorted(cls_args.keys()):
        strings.append(
            key + '=' + str(cls_args[key])
                    )
        
    return '-'.join(strings)