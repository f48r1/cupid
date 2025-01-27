import pandas as pd
import numpy as np
from compchemkit import fingerprints

import os
import shap

from .data_manage import get_Xy, get_smiles_train
from .cls_configs import clsf, clsf_params
from .paths import DIR_CV, DIR_PICKLE, define_cls_folder, DIR_SMARTS
from .target_configs import TARGET_NAMES
import pickle

from matplotlib import pyplot as plt
from matplotlib import colors as mcolors
from matplotlib import cm

def get_scores_pred(dataset, split, seed, dir_scores=DIR_CV, cls_folder='cls=RF-max_features=sqrt-n_estimators=100'):
    path = os.path.join(dir_scores, cls_folder,
                        f'dataset={dataset}-split={split}',
                        f'seed={seed}-fold=5.csv')
    
    scores=pd.read_csv(path, header=None, index_col=0)
    return scores

def load_csfp():
    SMARTS=np.loadtxt(DIR_SMARTS, dtype=str, comments=None)
    CSFP = fingerprints.FragmentFingerprint(substructure_list=SMARTS.tolist())
    return CSFP


class Cupido:
    
    def __init__(self, target, cls_name='RF', seeds=10, splits=10):

        self.target = target
        self.seeds= seeds
        self.splits = splits
        self.cls_name = cls_name
        self.cls_params = {k:v['default'] for k,v in clsf_params[cls_name].items()}
        self.cls_folder = define_cls_folder(self.cls_name, self.cls_params)

        X, y = get_Xy(target, split=-1)

        self.X = X
        self.y = y

        self.smiles_train = get_smiles_train(target)

        self.load_scores()
        self.load_reliability()
        self.load_cls()
        self.load_explainer()

    def load_scores(self):
        target = self.target
        seeds = self.seeds
        splits = self.splits
    
        cls_folder = self.cls_folder
        scores_path = os.path.join(DIR_PICKLE, cls_folder, 'scores')
        score_path = os.path.join(scores_path, self.target + '.pkl')

        if os.path.exists(score_path):
            with open(score_path, 'rb') as f:
                self.scores = pickle.load(f)

        else:
            self.scores = pd.DataFrame(index=self.y.index)

            for seed in range(seeds):
                for split in range(splits):
                    tmp_scores = get_scores_pred(target, split, seed, cls_folder=self.cls_folder)
                    self.scores = pd.concat([ self.scores, tmp_scores ], axis=1, ignore_index=True)

            os.makedirs(scores_path, exist_ok=True)
            with open(score_path, 'wb') as f:
                pickle.dump(self.scores, f)

    def load_reliability(self):
        self.pred_labels = self.scores.map(lambda x: np.nan if pd.isna(x) else (0 if x < .5 else 1) )

        X,y = self.X, self.y
        pred_labels = self.pred_labels
        scores = self.scores

        mask_correct = np.expand_dims(y,1) == pred_labels
        mask_nan = scores.isna()
        
        scores_correct = scores.values.flatten()[ mask_correct.values.flatten() & ~mask_nan.values.flatten() ]
        scores_uncorrect = scores.values.flatten()[ ~mask_correct.values.flatten() & ~mask_nan.values.flatten() ]

        bins_value, bin_edges = np.histogram(scores_correct,
                                            bins=11, range=(0,1))
        bins_value_inc, _ = np.histogram(scores_uncorrect,
                                        bins=11, range=(0,1))

        self.ratio = bins_value/(bins_value+bins_value_inc)
        self.bin_edges = bin_edges


    def load_cls(self):
        X,y = self.X, self.y
        cls_folder = self.cls_folder
        models_path = os.path.join(DIR_PICKLE, cls_folder, 'models')
        model_path = os.path.join(models_path, self.target + '.pkl')

        if os.path.exists(model_path):
            with open(model_path, 'rb') as f:
                self.cls = pickle.load(f)

        else:
            self.cls = clsf[self.cls_name](self.cls_params)
            self.cls.fit(X.values, y.values)

            os.makedirs(models_path, exist_ok=True)
            with open(model_path, 'wb') as f:
                pickle.dump(self.cls, f)

        self.predict = self.cls.predict
        self.predict_proba = self.cls.predict_proba

    def load_explainer(self):
        X = self.X
        cls_folder = self.cls_folder
        explainers_path = os.path.join(DIR_PICKLE, cls_folder, 'explainers')
        explainer_path = os.path.join(explainers_path, self.target + '.pkl')

        # BUG bug when reload this explainer
        if False: # if os.path.exists(explainer_path):
            with open(explainer_path, 'rb') as f:
                self.explainer = pickle.load(f)

        else:
            if self.cls_name == 'RF':
                self.explainer = shap.TreeExplainer(
                        self.cls,
                        data=X.values,
                        model_output='probability', # [ ] is it ok ?
                        feature_names=X.columns.values.astype(str).tolist(),
                        )
            else: # dedicated to the XGB, currently ...
                self.explainer = shap.TreeExplainer(
                        self.cls,
                        data=X.values,
                        # model_output='log_loss', # [ ] to wonder if it should be set to the probability too
                        feature_names=X.columns.values.astype(str).tolist(),
                        )

            # os.makedirs(explainers_path, exist_ok=True)
            # with open(explainer_path, 'wb') as f:
            #     pickle.dump(self.explainer, f)

    def compute_shap_values(self, array):
        shap_values = self.explainer(array, check_additivity=False if self.target == 'erg' else True)

        if self.cls_name == 'RF':
            shap_values = shap_values[...,1]

        # XXX auto fix output shap values shape ? If 1 molecule it returns 1 row, else entire molecular dataset
        if array.shape[0] == 1:
            shap_values = shap_values[0]

        return shap_values
    
    def isin_train(self, smi):
        return smi in self.smiles_train
    
    def reliability(self, pred_score):
        bin_edges = self.bin_edges
        width = bin_edges[1]- bin_edges[0]
        ratio = self.ratio

        idx_bin = int(pred_score // width)
        reliability=ratio[idx_bin]

        return reliability

    def reliability_fig(self, pred_score, reliability):

        bin_edges = self.bin_edges
        width = bin_edges[1]- bin_edges[0]
        ratio = self.ratio

        # Plot the ratio
        norm = mcolors.Normalize(vmin=ratio.min(), vmax=ratio.max())
        #cmap = cm.PiYG
        cmap=cm.Blues
        # You can choose any other colormap from Matplotlib

        idx_bin = int(pred_score // width)
        reliability=ratio[idx_bin]

        fig, ax = plt.subplots(figsize=(10,6))

        ax.bar(bin_edges[:-1], ratio, width=width, edgecolor='black', color=cmap(norm(ratio)), align='edge')
        ax.bar(pred_score, reliability, width=.01, edgecolor='black', color='red' , align='center')
        ax.set_xlabel('Probability Bins')
        ax.set_ylabel('Ratio of Correct to Total')
        ax.set_ylim(0.5,1)
        ax.set_xlim(0,1)
        ax.set_title(TARGET_NAMES[self.target])
        ax.set_xticks([ t/10 for t in range(11)])
    
        #Ratio of Correct Predictions to Total Predictions in Each Probability Bin
        plt.close(fig)

        return fig