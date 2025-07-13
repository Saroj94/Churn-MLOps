from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

MODEL_REGISTRY = {
    "LogisticRegression": LogisticRegression,
    "SVC": SVC,
    "XGBClassifier":XGBClassifier,
    "AdaBoostClassifier": AdaBoostClassifier,
    "DecisionTreeClassifier": DecisionTreeClassifier,
    "LGBMClassifier": LGBMClassifier
}