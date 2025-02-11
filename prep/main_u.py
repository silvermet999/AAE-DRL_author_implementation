"""-----------------------------------------------import libraries-----------------------------------------------"""
from collections import defaultdict
from scipy import stats
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MaxAbsScaler, MinMaxScaler
from sklearn.feature_selection import RFE, RFECV
from sklearn.svm import SVR

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold
from fitter import Fitter, get_common_distributions
import warnings

warnings.filterwarnings('ignore')





"""--------------------------------------------data exploration/cleaning--------------------------------------------"""
train = pd.read_csv('/home/silver/UNSW_NB15_training-set.csv')
test = pd.read_csv('/home/silver/UNSW_NB15_testing-set.csv')
extra = pd.read_csv("/home/silver/PycharmProjects/AAEDRL/data/dataset.csv")
extra = extra.rename(columns={
    'sintpkt': 'sinpkt',
    'dintpkt': 'dinpkt',
    'ct_src_ ltm': "ct_src_ltm",
    'Label': "label"
})

dfs = [train, test, extra]
df = pd.concat(dfs, ignore_index=True)
df = df.drop(df.columns[df.nunique() == 1], axis = 1) # no change
df = df.drop(df.columns[df.nunique() == len(df)], axis = 1) # no change
df = df.drop(["id", "rate"], axis=1)

df["is_ftp_login"] = df["is_ftp_login"].replace([4, 2], 1).astype(int)

df = df[df['proto'] != 'a/n']
df = df[df['service'] != '-']
df = df[df["state"] != "no"]

# df["proto"].replace("a/n", np.nan, inplace=True)
# df["service"].replace("-", np.nan, inplace=True)
# df["state"].replace("no", np.nan, inplace=True)


df["attack_cat"] = df["attack_cat"].replace([' Fuzzers', ' Fuzzers '], "Fuzzers")
df["attack_cat"] = df["attack_cat"].replace("Backdoors", "Backdoor")
df["attack_cat"] = df["attack_cat"].replace(" Reconnaissance ", "Reconnaissance")
df["attack_cat"] = df["attack_cat"].replace(" Shellcode ", "Shellcode")

le = LabelEncoder()
cols_le = ["attack_cat", "proto", "service", "state"]
mappings = {}

for col in cols_le:
    df[col] = le.fit_transform(df[col])
#     mappings[col] = dict(zip(le.classes_, le.transform(le.classes_)))
# #
# for col, mapping in mappings.items():
#     print(f"Mapping for {col}: {mapping}")

for col in cols_le:
    value_counts = df[col].value_counts()
    singletons = value_counts[value_counts == 1].index
    df = df[~df[col].isin(singletons)]



def corr(df):
    correlation = df.corr()
    f_corr = {}
    for column in correlation.columns:
        correlated_with = list(correlation.index[(correlation[column] >= 0.75) | (correlation[column] <= -0.75)])
        for corr_col in correlated_with:
            if corr_col != column:
                df_corr = correlation.loc[corr_col, column]
                f_corr[(column, corr_col)] = df_corr

    plt.figure(figsize=(30, 20))
    sns.heatmap(correlation, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1)
    plt.title("Correlation Matrix Heatmap")
    plt.savefig("corr.png")
    return f_corr

df = df.drop(["service", "dttl", "sttl", "is_sm_ips_ports", 'ct_ftp_cmd', 'ct_flw_http_mthd', "dload", "stcpb", "dtcpb",
              "dmean", "ct_dst_src_ltm"], axis=1)

normal = df[df['attack_cat'].isin([6])]
noridx = normal.tail(240000).index
df = df.drop(noridx)
generic = df[df['attack_cat'].isin([5])]
genidx = generic.tail(240000).index
df = df.drop(genidx)
df["attack_cat"] = df["attack_cat"].replace([2, 4, 7, 0, 8, 1], 0)
df["attack_cat"] = df["attack_cat"].replace(5, 1)
df["attack_cat"] = df["attack_cat"].replace(3, 2)
df["attack_cat"] = df["attack_cat"].replace(6, 3)
# profiler = ProfileReport(df)
# profiler.to_file("report.html")

"""-----------------------------------------------vertical data split-----------------------------------------------"""
def vertical_split(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=45)
    return X_train, X_test, y_train, y_test


X = df.drop(["attack_cat", "label"], axis = 1)
y = df["attack_cat"]

X_train, X_test, y_train, y_test = vertical_split(X, y)

def df_type_split(df):
    X_cont = df.drop(['state', 'ct_state_ttl', "trans_depth", "proto", "is_ftp_login",
                      # 'service', 'dttl', "is_sm_ips_ports", "ct_ftp_cmd", "ct_flw_http_mthd", 'sttl',
                      ], axis=1)
    X_disc = df[['state', 'ct_state_ttl', "trans_depth", "proto", "is_ftp_login"
                      # 'service', 'dttl', "is_sm_ips_ports", "ct_ftp_cmd", "ct_flw_http_mthd", 'sttl',
                      ]]
    return X_disc, X_cont

def prep(X_disc, X_cont):
    cont_cols = X_cont.columns
    cont_index = X_cont.index
    scaler = MinMaxScaler()
    X_cont = scaler.fit_transform(X_cont)
    X_cont = pd.DataFrame(X_cont, columns=cont_cols, index=cont_index)
    X_sc = pd.concat([X_disc, X_cont], axis=1)
    return X_sc

X_disc, X_cont = df_type_split(X)
X_disc_train, X_cont_train = df_type_split(X_train)
X_disc_test, X_cont_test = df_type_split(X_test)

X_sc = prep(X_disc, X_cont)
X_train_sc = prep(X_disc_train, X_cont_train)
X_test_sc = prep(X_disc_test, X_cont_test)


def optimize_features_rfe(X, y):
    estimator = SVR(kernel="linear")
    cv = StratifiedKFold(n_splits=5)
    rfecv = RFECV(estimator=estimator, step=1, cv=cv, scoring='accuracy')
    rfecv.fit(X, y)
    # selector = RFE(estimator, n_features_to_select=30, step=10)
    # selector.fit(X, y)
    return rfecv.ranking_


#
def kolmogorov_smirnov_test(column, dist='norm'):
    D, p_value = stats.kstest(column, dist)
    return p_value > 0.05

def anderson_darling_test(column, dist='norm'):
    result = stats.anderson(column, dist=dist)
    return result.statistic < result.critical_values[2]

def fit_distributions(column):
    f = Fitter(column, distributions=get_common_distributions())
    f.fit()
    f.summary()
    best_fit = f.get_best(method='sumsquare_error')

    return best_fit

# results = {}
#
# for idx, column in enumerate(X_train_sc.columns):
#     col_results = []
#     if kolmogorov_smirnov_test(X_train_sc[column]):
#         col_results.append(f"Column {column} follows the specified distribution (Kolmogorov-Smirnov test).")
#     if anderson_darling_test(X_train_sc[column]):
#         col_results.append(f"Column {column} follows the specified distribution (Anderson-Darling test).")
#     best_fit = fit_distributions(X_train_sc[column])
#     col_results.append(f"{best_fit}")
#     results[idx] = col_results
#
# for idx in range(31):
#     if idx in results:
#         print(f"{idx}")
#         for result in results[idx]:
#             print(result)
#         print()
#     else:
#         print(f"{idx}: No data\n")
