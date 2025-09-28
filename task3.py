import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load dataset
df = pd.read_csv('Housing.csv')  # Replace with actual path

# Optional: Inspect data
# print(df.head())
# print(df.info())
# print(df.describe())
# print(df.isnull().sum())

# Step 1: Remove outliers from key numeric columns
def remove_outliers(colo):
    filtered_df = df.copy()
    for col in colo:
        Q1 = filtered_df[col].quantile(0.25)
        Q3 = filtered_df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        filtered_df = filtered_df[(filtered_df[col] >= lower) & (filtered_df[col] <= upper)]
    return filtered_df

df = remove_outliers(['area', 'price'])

# Step 2: Encode categorical variables
categorical_cols = ['mainroad', 'guestroom', 'basement', 'hotwaterheating',
                    'airconditioning', 'prefarea', 'furnishingstatus']

df_encoded = pd.get_dummies(df, columns=categorical_cols)

# Step 3: Define features and target
X = df_encoded.drop('price', axis=1)
y = df_encoded['price']

# Step 4: Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 6: Predict and evaluate
y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("MAE:", mae)
print("MSE:", mse)
print("R² Score:", r2)

# Step 7: Visualize predictions
plt.scatter(y_test, y_pred, color='blue', alpha=0.6)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted House Prices")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--')
plt.show()


# step 6

# If using one feature like 'SquareFeet'
X_simple = df[['area']]
y_simple = df['price']

X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(X_simple, y_simple, test_size=0.2, random_state=42)

model_s = LinearRegression()
model_s.fit(X_train_s, y_train_s)

plt.scatter(X_test_s, y_test_s, color='blue')
plt.plot(X_test_s, model_s.predict(X_test_s), color='red')
plt.xlabel('Square Feet')
plt.ylabel('Price')
plt.title('House Price vs Square Feet')
plt.show()
