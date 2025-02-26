# Import pandas
import pandas as pd

# Read dataset
df1 = pd.read_csv("/home/silver/UNSW-NB15_1.csv")
df2 = pd.read_csv("/home/silver/UNSW-NB15_2.csv")
df3 = pd.read_csv("/home/silver/UNSW-NB15_3.csv")
df4 = pd.read_csv("/home/silver/UNSW-NB15_4.csv")

# Define column names
df1.columns = ["srcip", "sport", "dstip", "dsport",
               "proto", "state", "dur", "sbytes", "dbytes", "sttl", "dttl",
"sloss", "dloss", "service", "sload", "dload", "spkts", "dpkts", "swin", "dwin", "stcpb", "dtcpb", "smean",
"dmean", "trans_depth", "response_body_len", "sjit", "djit", "Stime", "Ltime", "sintpkt", "dintpkt", "tcprtt", "synack",
"ackdat", "is_sm_ips_ports", "ct_state_ttl", "ct_flw_http_mthd", "is_ftp_login", "ct_ftp_cmd", "ct_srv_src",
"ct_srv_dst", "ct_dst_ltm", "ct_src_ ltm", "ct_src_dport_ltm", "ct_dst_sport_ltm", "ct_dst_src_ltm", "attack_cat",
"Label"]
df2.columns = df1.columns
df3.columns = df1.columns
df4.columns = df1.columns

# Join datasets
dfs = [df1, df2, df3, df4]
df = pd.concat(dfs, ignore_index=True)

# Drop non common columns (not available in the partition dataset)
df = df.drop(["srcip", "sport", "dstip", "dsport", "Stime", "Ltime"], axis=1)

# Fill null values
df["attack_cat"] = df["attack_cat"].fillna(0)
df["is_ftp_login"] = df["is_ftp_login"].fillna(0)
df["ct_flw_http_mthd"] = df["ct_flw_http_mthd"].fillna(0)
mapping_cmd = {' ': 0, '1': 1, '2': 2, '4': 4}
df['ct_ftp_cmd'] = df['ct_ftp_cmd'].replace(mapping_cmd)

# Normal behavior samples (we get the samples from the first dataset thus we commented out the others) 
# you can drop normal samples from the dataset using df[df['attack_cat'].isin([0])] => .tail(#n-samples-to-drop).index => .drop(genidx)
df["attack_cat"] = df["attack_cat"].replace(0, "Normal")

