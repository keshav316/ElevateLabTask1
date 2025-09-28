import pandas as pd

# Load dataset
df = pd.read_csv('titanic.csv')  # Replace with your actual path

# Preview
print(df.head())
print(df.info())
# Basic stats
print(df.describe(include='all'))

# Specific stats
print("Mean Age:", df['Age'].mean())
print("Median Fare:", df['Fare'].median())
print("Standard Deviation of Fare:", df['Fare'].std())
print("Standard deviation for AGE: ",df['Age'].std())

# step 3
import matplotlib.pyplot as plt
import seaborn as sns

# Histogram of Age
sns.histplot(df['Age'].dropna(), bins=10, kde=True)
plt.title('Age Distribution')
plt.show()

# Boxplot of Fare
sns.boxplot(x=df['Fare'])
plt.title('Fare Boxplot')
plt.show()


# step 4

# Pairplot (selected features)
sns.pairplot(df[['Age', 'Fare', 'Survived', 'Pclass']].dropna(), hue='Survived')
plt.show()

# Correlation matrix
corr = df[['Age', 'Fare', 'Survived', 'Pclass']].corr()
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title('Feature Correlation Matrix')
plt.show()


# step 5
 
# Survival rate by gender
sns.barplot(x='Sex', y='Survived', data=df)
plt.title('Survival Rate by Gender')
plt.show()

# Age vs Survival
sns.violinplot(x='Survived', y='Age', data=df)
plt.title('Age Distribution by Survival')
plt.show()

