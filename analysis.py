import seaborn as sns
import matplotlib.pyplot as plt

sns.countplot(x='fraud_reported', data=df)
plt.title("Fraud Reported Count")
plt.show()

sns.countplot(x='incident_severity', data=df)
plt.title("Incident Severity Count")
plt.show()

sns.countplot(x='incident_severity', data=df)
plt.title("Incident Severity Distribution")
plt.show()

sns.countplot(x='insured_sex', data=df)
plt.title("Gender Distribution")
plt.show()

severity = df['incident_severity'].value_counts()
plt.pie(severity,
        labels=severity.index,
        autopct='%1.1f%%')
plt.title("Incident Severity Composition")
plt.show()

sns.histplot(df['age'], bins=20, kde=True)
plt.title("Age Distribution")
plt.xlabel("Age")
plt.show()

sns.kdeplot(df['age'], fill=True)
plt.title("Age Density Plot")
plt.show()

num_cols = df.select_dtypes(include=['int64','float64']).columns

#multivariate analysis

df_num = df.select_dtypes(include='number')
corr = df_num.corr()
plt.figure(figsize=(14,10))
sns.heatmap(corr,
            annot=True,
            cmap='coolwarm',
            fmt='.2f')

plt.title("Correlation Heatmap")
plt.show()
