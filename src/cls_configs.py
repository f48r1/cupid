from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier

clsf_params = {
    'RF':{
        'n_estimators':{
            'type':int,
            'default':100,
                        },
        'max_features':{
            'type':str,
            'default':'sqrt'
        }
    },
    'SVM':{},
    'XGB':{},
    'KNN':{},
    'ADA':{},
}
    
clsf={
    'RF': lambda x: RandomForestClassifier(**x, random_state=0),
    'SVM': lambda x: svm.SVC(kernel='linear', probability=True, **x),
    'XGB': lambda x: GradientBoostingClassifier(**x),
    'KNN': lambda x: KNeighborsClassifier (n_neighbors=3, **x),
    'ADA': lambda x: AdaBoostClassifier (**x)
}