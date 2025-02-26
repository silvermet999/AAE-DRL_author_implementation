import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import LabelBinarizer
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import cross_val_predict

from sklearn.neighbors import KNeighborsClassifier


def clf_class(X_train, X_test, y_train, y_test, xgb_clf=False, KNN_clf=False, rf_clf=False):
    if xgb_clf:
        print("using XGB")
        best_params = {'booster': 'gbtree', 'lambda': 0.27402306472106963, 'alpha': 1.0337469639524462e-07,
                       'subsample': 0.8763224738010359, 'colsample_bytree': 0.8497549372024669, 'max_depth': 18,
                       'min_child_weight': 3, 'eta': 0.17676685584319898, 'gamma': 4.5337669584553865e-07,
                       'grow_policy': 'lossguide', "verbosity": 0, "objective": "multi:softmax", "num_class": 30}
        # best_params = {'booster': 'gbtree', 'lambda': 1.1068723944171659e-08, 'alpha': 8.160758817453953e-08,
        # 'subsample': 0.7402331626687324, 'colsample_bytree': 0.5025353594627584, 'max_depth': 22, 'min_child_weight': 2,
        # 'eta': 0.050901063251071806, 'gamma': 5.515409506712251e-08, 'grow_policy': 'lossguide', "verbosity": 0,
        # "objective": "multi:softmax", "num_class": 30}
        clf = XGBClassifier(**best_params)

    elif KNN_clf:
        print("using KNN")
        clf = KNeighborsClassifier(n_neighbors= 15, metric= 'manhattan', leaf_size= 11) # 'n_neighbors': 27, 'metric': 'manhattan', 'leaf_size': 64
    elif rf_clf:
        print("using RF")
        clf = RandomForestClassifier(max_depth=15, n_estimators=186) # 'max_depth': 14, 'n_estimators': 177
    else:
        print("using GB")
        clf = GradientBoostingClassifier(n_estimators=15, learning_rate=0.08372870472978572, max_depth=12) # 'n_estimators': 26, 'learning_rate': 0.03515322582815399, 'max_depth': 10

    clf.fit(X_train, y_train)
    y_pred = cross_val_predict(clf, X_train, y_train, cv=5, method="predict_proba")
    y_pred_max = np.argmax(y_pred, axis=1)
    report_train = classification_report(y_train, y_pred_max)
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
