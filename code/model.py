import numpy as np
import pandas as pd
import sklearn.linear_model as sk

df_pred = pd.read_csv('/home/arizzo/Projects/Iris/data/processed/iris_predictors.csv')
df_resp = pd.read_csv('/home/arizzo/Projects/Iris/data/processed/iris_response.csv')

print(df_pred)
print(df_resp)

sk.LogisticRegressionCV(df_pred,cv=10,multi_class=’ovr’)