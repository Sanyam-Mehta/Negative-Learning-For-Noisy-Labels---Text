import json
import pandas as pd
import numpy as np
from utils import read_trec6_ATS

TEXT_COL, LABEL_COL = 'text', 'label'
colnames = [LABEL_COL, TEXT_COL]

with open('Data/trec6_ATS/train.json') as f:
    data = json.load(f)

df_train = pd.read_json('Data/trec6_ATS/train.json')
df_test = pd.read_json('Data/trec6_ATS/test.json')
# print(data)

df2 = pd.read_csv("Data/trec/trec6.csv", names=colnames)
# print(df2)
#
# print(df_train)
#
# print(df_test)

# print(np.where(df_test['actual_label'] == df_test['predicted_label'], 1, 0).tolist())

# print(df_train['actual_label'].equals(df_train['predicted_label']))

df = read_trec6_ATS("Data/trec6_ATS")
print(df)