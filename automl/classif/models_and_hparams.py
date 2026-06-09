from typing import Literal

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

try:
    from xgboost import XGBClassifier  # type: ignore
except:
    XGBClassifier = None
try:
    from lightgbm import LGBMClassifier  # type: ignore
except:
    LGBMClassifier = None
try:
    from catboost import CatBoostClassifier  # type: ignore
except:
    CatBoostClassifier = None

def get_models(class_weight: Literal['balanced'] | None, random_state: int, plan_id: int = 3):
    models = {
        "LogisticRegression": LogisticRegression(max_iter=1000, class_weight=class_weight),
        "RandomForest": RandomForestClassifier(random_state=random_state, class_weight=class_weight),
        "SVC": SVC(probability=True, class_weight=class_weight),
        "KNN": KNeighborsClassifier(),
        "DecisionTree": DecisionTreeClassifier(random_state=random_state, class_weight=class_weight),
        "NaiveBayes": GaussianNB()
    }

    if plan_id >= 2:
        models["GradientBoosting"] = GradientBoostingClassifier(random_state=random_state)
        if XGBClassifier:
            models["XGBoost"] = XGBClassifier(random_state=random_state, use_label_encoder=False, eval_metric="logloss")
        if LGBMClassifier:
            models["LightGBM"] = LGBMClassifier(random_state=random_state, verbosity=-1)
        if CatBoostClassifier:
            models["CatBoost"] = CatBoostClassifier(verbose=0, random_state=random_state)
    
    return models
