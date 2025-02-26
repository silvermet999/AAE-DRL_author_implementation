import argparse
import sys

import pandas as pd
import xgboost as xgb
from clfs import benchmark_classification
import utils
from data import main_u


def parse_args(args):
    parser = argparse.ArgumentParser()

    parser.add_argument('--features', default="/home/silver/PycharmProjects/AAEDRL/AAE/ds_fin2.csv")
    parser.add_argument('--labels', default="/home/silver/PycharmProjects/AAEDRL/clfs/labels.csv")

    parser.add_argument("--xgb_clf", default= False, type=bool)
    parser.add_argument("--KNN_clf", default= False, type=bool)
    parser.add_argument("--rf_clf", default= False, type=bool)

    return parser.parse_args(args)

if __name__ == "__main__":
    args = sys.argv[1:]
    args = parse_args(args)
    df = pd.DataFrame(pd.read_csv(args.features))
    df_disc, df_cont = main_u.df_type_split(df)
    _, mainX_cont = main_u.df_type_split(main_u.X)
    X_inv = utils.inverse_sc_cont(mainX_cont, df_cont)
    X = df_disc.join(X_inv)

    y_rl = pd.DataFrame(pd.read_csv(args.labels))
    y_rl = y_rl[y_rl["attack_cat"] != 2]
    y = pd.concat([y_rl, main_u.y], axis=0)
    y = y.squeeze()
    X_train, X_test, y_train, y_test = main_u.vertical_split(X, y[:173252])

    benchmark_classification.clf_class(X_train, X_test, y_train, y_test, xgb_clf=args.xgb_clf, KNN_clf=args.KNN_clf,
                                       rf_clf=args.rf_clf)
