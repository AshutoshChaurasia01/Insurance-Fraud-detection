import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report

import warnings
import pickle
from scipy import stats
df=pd.read_csv("insurance_claims.csv")
df.replace("?", np.nan, inplace=True)
sns.boxplot(x=df['policy_annual_premium'])
plt.show()
IQR=[]

IQR.append(df['age'].quantile(0.75)-df['age'].quantile(0.25))

IQR.append(df['policy_annual_premium'].quantile(0.75)-df['policy_annual_premium'].quantile(0.25))
IQR.append(df['umbrella_limit'].quantile(0.75)-df['umbrella_limit'].quantile(0.25))
IQR.append(df['total_claim_amount'].quantile(0.75)-df['total_claim_amount'].quantile(0.25))

IQR.append(df['property_claim'].quantile(0.75)-df['property_claim'].quantile(0.25))

IQR
upper=[]

upper.append(df['age'].quantile (0.75)+1.5* (IQR[0]) )

upper.append(df['policy_annual_premium'].quantile(0.75)+1.5 * (IQR[1]) )

upper.append(df['umbrella_limit'].quantile(0.75)+1.5*(IQR[2]))

upper.append(df['total_claim_amount'].quantile(0.75)+1.5*(IQR[3]))

upper.append(df['property_claim'].quantile(0.75)+1.5*(IQR[4]))

upper
df_num_features = df.select_dtypes(include=['int64','float64'])
for k in df_num_features.columns:


  print(k)

  sns.boxplot(x=df_num_features[k].dropna())

  plt.show()
