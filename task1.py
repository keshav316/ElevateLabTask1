from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("titanic.csv")  # Replace with your actual path

# Basic info
# print(df.head())
# print(df.info())
# print(df.describe())

# print(df.isnull().sum())
df['Age'].fillna(df['Age'].median(), inplace=True)
# print(df.isnull().sum())
df.drop(columns=['Cabin'], inplace=True)
df['Embarked']=df['Embarked'].fillna(df['Embarked'].mode()[0])

# print(df.isnull().sum())


# Convert 'Sex' and 'Embarked' to numerical
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
df = pd.get_dummies(df, columns=['Embarked'], drop_first=True)

scaler = StandardScaler()
df[['Age', 'Fare']] = scaler.fit_transform(df[['Age', 'Fare']])
# # Boxplots
sns.boxplot(x=df['Age'])
# plt.show()

sns.boxplot(x=df['Fare'])
# plt.show()

# # Remove outliers using IQR
def remove_outliers(col):
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    return df[(df[col] >= lower) & (df[col] <= upper)]

df = remove_outliers('Fare')
df = remove_outliers('Age')
print(df.head())