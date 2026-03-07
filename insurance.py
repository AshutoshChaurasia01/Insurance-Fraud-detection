import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoost
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score
from skleang.metrics import classification_report, confusion_matri
import warnings
import pickle
from scipy import stats
df=pd.read_csv("insurance_claims.csv")
df.head()
