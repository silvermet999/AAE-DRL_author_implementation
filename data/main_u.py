"""-----------------------------------------------import libraries-----------------------------------------------"""
from collections import defaultdict
from scipy import stats
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.feature_selection import RFE, RFECV
from sklearn.svm import SVR

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold
from fitter import Fitter, get_common_distributions
import warnings

from ydata_profiling import ProfileReport

warnings.filterwarnings('ignore')





"""--------------------------------------------data exploration/cleaning--------------------------------------------"""

"""Read CSV files and join datasets"""
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


"""Clean dataset"""
# Check for all unique and one unique values in each column
df = df.drop(df.columns[df.nunique() == 1], axis = 1)
df = df.drop(df.columns[df.nunique() == len(df)], axis = 1)

# Drop non common columns between datasets
df = df.drop(["id", "rate"], axis=1)

# This column is supposedly binary => https://unsw-my.sharepoint.com/:x:/r/personal/z5025758_ad_unsw_edu_au/_layouts/15/Doc.aspx?sourcedoc=%7B975B24E4-7E36-4CE1-B98A-9FBE4BB521B7%7D&file=NUSW-NB15_features.csv&action=default&mobileredirect=true
df["is_ftp_login"] = df["is_ftp_login"].replace([4, 2], 1).astype(int)

# Drop null values
df = df[df['proto'] != 'a/n']
df = df[df['service'] != '-']
df = df[df["state"] != "no"]

# Discard spaces
df["attack_cat"] = df["attack_cat"].replace([' Fuzzers', ' Fuzzers '], "Fuzzers")
df["attack_cat"] = df["attack_cat"].replace("Backdoors", "Backdoor")
df["attack_cat"] = df["attack_cat"].replace(" Reconnaissance ", "Reconnaissance")
df["attack_cat"] = df["attack_cat"].replace(" Shellcode ", "Shellcode")

# Encode categorical columns (for visualization)
le = LabelEncoder()
cols_le = ["attack_cat", "proto", "service", "state"]
mappings = {}

for col in cols_le:
    df[col] = le.fit_transform(df[col])
#     mappings[col] = dict(zip(le.classes_, le.transform(le.classes_)))
# #
# for col, mapping in mappings.items():
#     print(f"Mapping for {col}: {mapping}")

# Discard singletons
for col in cols_le:
    value_counts = df[col].value_counts()
    singletons = value_counts[value_counts == 1].index
    df = df[~df[col].isin(singletons)]

# Outlier capping (was not applied)
def cap_outliers_iqr(df, factor=40):
    capped_df = df.copy()
    for column in df.select_dtypes(include=[np.number]).columns:
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_threshold = Q1 - factor * IQR
        upper_threshold = Q3 + factor * IQR
        print(lower_threshold)
        print(upper_threshold)
        capped_df[column] = np.where(df[column] < lower_threshold, lower_threshold, df[column])
        capped_df[column] = np.where(capped_df[column] > upper_threshold, upper_threshold, capped_df[column])
    return capped_df

# Correlation analysis
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

# Get portfolio
# profiler = ProfileReport(df)
# profiler.to_file("report.html")

# Drop non-important features according to RFE
df = df.drop(["service", "dttl", "sttl", "is_sm_ips_ports", 'ct_ftp_cmd', 'ct_flw_http_mthd', "dload", "stcpb", "dtcpb",
              "dmean", "ct_dst_src_ltm"], axis=1)

"""------------------------------------------------data preprocessing------------------------------------------------"""

# Balance dataset
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

# Train test split
def vertical_split(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=45)
    return X_train, X_test, y_train, y_test

X = df.drop(["attack_cat", "label"], axis = 1)
y = df["attack_cat"]

X_train, X_test, y_train, y_test = vertical_split(X, y)

# Data type separation
def df_type_split(df):
    X_cont = df.drop(['state', 'ct_state_ttl', "trans_depth", "proto", "is_ftp_login"
                      ], axis=1)
    X_disc = df[['state', 'ct_state_ttl', "trans_depth", "proto", "is_ftp_login"
                      ]]
    return X_disc, X_cont

# Feature scaling
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

# Dimensionality reduction
def optimize_features_rfe(X, y):
    estimator = SVR(kernel="linear")
    cv = StratifiedKFold(n_splits=5)
    rfecv = RFECV(estimator=estimator, step=1, cv=cv, scoring='accuracy')
    rfecv.fit(X, y)
    # selector = RFE(estimator, n_features_to_select=30, step=10)
    # selector.fit(X, y)
    return rfecv.ranking_

