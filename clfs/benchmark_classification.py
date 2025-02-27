import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import LabelBinarizer
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import cross_val_predict

from sklearn.neighbors import KNeighborsClassifier

"""
Benchmark classification
------------------------
The parameters are defined according to the results of the OPTUNA trials.
The commented parameters are for the unaugmented data!
The defined parameters are for the augmented data!
"""

# parameters for unaugmented dataset
best_xgb_param_unaug = {'booster': 'gbtree', 'lambda': 1.1068723944171659e-08, 'alpha': 8.160758817453953e-08,
        'subsample': 0.7402331626687324, 'colsample_bytree': 0.5025353594627584, 'max_depth': 22, 'min_child_weight': 2,
        'eta': 0.050901063251071806, 'gamma': 5.515409506712251e-08, 'grow_policy': 'lossguide', "verbosity": 0,
        "objective": "multi:softmax", "num_class": 30}
best_KNN_param_unaug = {'n_neighbors': 27, 'metric': 'manhattan', 'leaf_size': 64}
best_rf_param_unaug = {'max_depth': 14, 'n_estimators': 177}
best_gb_param_unaug = {'n_estimators': 26, 'learning_rate': 0.03515322582815399, 'max_depth': 10}

# parameters for augmented dataset
best_xgb_param_aug = {'booster': 'gbtree', 'lambda': 0.27402306472106963, 'alpha': 1.0337469639524462e-07,
                       'subsample': 0.8763224738010359, 'colsample_bytree': 0.8497549372024669, 'max_depth': 18,
                       'min_child_weight': 3, 'eta': 0.17676685584319898, 'gamma': 4.5337669584553865e-07,
                       'grow_policy': 'lossguide', "verbosity": 0, "objective": "multi:softmax", "num_class": 30}
best_KNN_param_aug = {"n_neighbors": 15, "metric": 'manhattan', "leaf_size": 11}
best_rf_param_aug = {"max_depth":15, "n_estimators" : 186}
best_gb_param_aug = {"n_estimators":15, "learning_rate" : 0.08372870472978572, "max_depth" : 12}

def clf_class(X_train, X_test, y_train, y_test, unaugmented=False, xgb_clf=False, KNN_clf=False, rf_clf=False):
    # choose one of these classifiers
    if xgb_clf:
        print("using XGB")
        best_params = best_xgb_param_unaug if unaugmented else best_xgb_param_aug
        clf = XGBClassifier(**best_params)

    elif KNN_clf:
        print("using KNN")
        best_params = best_KNN_param_unaug if unaugmented else best_KNN_param_aug
        clf = KNeighborsClassifier(**best_params)
    elif rf_clf:
        print("using RF")
        best_params = best_rf_param_unaug if unaugmented else best_rf_param_aug
        clf = RandomForestClassifier(**best_params)
    else:
        print("using GB")
        best_params = best_gb_param_unaug if unaugmented else best_gb_param_aug
        clf = GradientBoostingClassifier(**best_params)

    # train classifier
    clf.fit(X_train, y_train)
    y_pred = cross_val_predict(clf, X_train, y_train, cv=5, method="predict_proba")
    y_pred_max = np.argmax(y_pred, axis=1)
    report_train = classification_report(y_train, y_pred_max) # you can print the classification report for the training set
    # AUC
    y_proba = clf.predict_proba(X_test)
    lb = LabelBinarizer()
    y_test_binarized = lb.fit_transform(y_test)
    auc_scores = []
    for i in range(y_proba.shape[1]):
        auc = roc_auc_score(y_test_binarized[:, i], y_proba[:, i])
        auc_scores.append(auc)
        print(f"class {i}: {auc:.4f}")
    macro_auc = roc_auc_score(y_test_binarized, y_proba, average="macro")
    print(f"Macro-Average AUC-ROC: {macro_auc:.4f}")

    # predict on test set
    if xgb_clf:
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dvalid = xgb.DMatrix(X_test, label=y_test)
        test_clf = xgb.train(best_params, dtrain)
        y_pred_val = test_clf.predict(dvalid)

    else :
        y_pred_val = clf.predict(X_test)
    report_test = classification_report(y_test, y_pred_val)
    print(report_test)
    return macro_auc, report_test
