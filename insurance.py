import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


import warnings
import pickle
from scipy import stats
df=pd.read_csv("insurance_claims.csv")
df.replace("?", np.nan, inplace=True)
df = df.drop('_c39', axis=1)
num_cols = df.select_dtypes(include=['int64','float64']).columns
df[num_cols] = df[num_cols].fillna(df[num_cols].median())
df = df.dropna()
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
#by multivariate analysis highly corelated feature  dropped
df = df.drop(['months_as_customer','injury_claim','property_claim','vehicle_claim'], axis=1)


le = LabelEncoder()
daata1 = df.select_dtypes(include='object').columns
for i in daata1:
    df[i] = le.fit_transform(df[i])


X = df.iloc[:,0:30]
y = df.iloc[:,30]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0
)


Std_scaler = StandardScaler()
X_train = Std_scaler.fit_transform(X_train)
X_train = pd.DataFrame(X_train, columns=X.columns)
X_test = Std_scaler.transform(X_test)
X_test = pd.DataFrame(X_test, columns=X.columns)

#descision tree


dt=DecisionTreeClassifier()
dt.fit(X_train,y_train)
y_pred=dt.predict(X_test)
dt_train_acc=accuracy_score(y_train,dt.predict(X_train))
dt_test_acc=accuracy_score(y_test,y_pred)
print(f"Train accuracy: {dt_train_acc}")
print(f'Test accuracy: {dt_test_acc}')

# Random Forest Model

rf = RandomForestClassifier()
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
rf_train_acc = accuracy_score(y_train, rf.predict(X_train))
rf_test_acc = accuracy_score(y_test, y_pred)
print("Random Forest Train Accuracy:", rf_train_acc)
print("Random Forest Test Accuracy:", rf_test_acc)

# KNN Model
knn = KNeighborsClassifier(n_neighbors=30)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Logistic Regression Model
lg = LogisticRegression(max_iter=5000)
lg.fit(X_train, y_train)
lrg_pred = lg.predict(X_test)
print(confusion_matrix(y_test, lrg_pred))
print(classification_report(y_test, lrg_pred))


# Naive Bayes Model

gnb = GaussianNB()
model_2 = gnb.fit(X_train, y_train)
predict_log = model_2.predict(X_test)
print("Training Accuracy:", 100 * accuracy_score(model_2.predict(X_train), y_train))
print("Testing Accuracy:", 100 * accuracy_score(y_test, predict_log))

# SVM Model
svc = SVC()
svc.fit(X_train, y_train)
y_pred = svc.predict(X_test)
svc_train_acc = accuracy_score(y_train, svc.predict(X_train))
svc_test_acc = accuracy_score(y_test, y_pred)
print("Training accuracy of SVC :", svc_train_acc)
print("Test accuracy of SVC :", svc_test_acc)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

