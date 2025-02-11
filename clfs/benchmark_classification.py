import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import LabelBinarizer
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import cross_val_predict

import utils
from data import main_u
from sklearn.neighbors import KNeighborsClassifier



df = pd.DataFrame(pd.read_csv("/home/silver/PycharmProjects/AAEDRL/AAE/ds_fin2.csv"))
df_disc, df_cont = main_u.df_type_split(df)
_, mainX_cont = main_u.df_type_split(main_u.X)
X_inv = utils.inverse_sc_cont(mainX_cont, df_cont)
X = df_disc.join(X_inv)

y_rl = pd.DataFrame(pd.read_csv("/home/silver/PycharmProjects/AAEDRL/clfs/labels.csv"))
y_rl = y_rl[y_rl["attack_cat"] != 2]
y = pd.concat([y_rl, main_u.y], axis=0)
y = y.squeeze()


X_train, X_test, y_train, y_test = main_u.vertical_split(X, y[:173252])
dtrain = xgb.DMatrix(X_train, label=y_train)
dvalid = xgb.DMatrix(X_test, label=y_test)

def clf_class(xgb_clf=False, KNN_clf=False, rf_clf=False):
    if xgb_clf:
        print("using XGB")
        best_params = {'booster': 'gbtree', 'lambda': 0.27402306472106963, 'alpha': 1.0337469639524462e-07,
                       'subsample': 0.8763224738010359, 'colsample_bytree': 0.8497549372024669, 'max_depth': 18,
                       'min_child_weight': 3, 'eta': 0.17676685584319898, 'gamma': 4.5337669584553865e-07,
                       'grow_policy': 'lossguide', "verbosity": 0, "objective": "multi:softmax", "num_class": 30}
        clf = XGBClassifier(**best_params)

    elif KNN_clf:
        print("using KNN")
        clf = KNeighborsClassifier(n_neighbors= 15, metric= 'manhattan', leaf_size= 11)
    elif rf_clf:
        print("using RF")
        clf = RandomForestClassifier(max_depth=15, n_estimators=186)
    else:
        print("using GB")
        clf = GradientBoostingClassifier(n_estimators=12, learning_rate=0.08372870472978572, max_depth=15)

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
        test_clf = xgb.train(best_params, dtrain)
        y_pred_val = test_clf.predict(dvalid)

    else :
        y_pred_val = clf.predict(X_test)
    report_test = classification_report(y_test, y_pred_val)
    return macro_auc, report_test
