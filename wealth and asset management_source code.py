# Importing necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV

# Step 1: Load the dataset
# Assuming the dataset is stored in a CSV file named 'farm_data.csv'
farm_data = pd.read_excel(r"C:\Users\Meghna\Documents\Davey\Project\Farm and wealth Dataset.xlsx")
# Step 2: Perform basic Exploratory Data Analysis (EDA)
print(farm_data.head())  # Display the first few rows of the dataset
print(farm_data.info())  # Get information about the dataset
print(farm_data.describe())  # Summary statistics of numerical variables

print(farm_data.index)

# Print column names to check for the exact name
print(farm_data.columns)

# Check the structure of your DataFrame
print(farm_data.head())
# Remove extra spaces from column names
farm_data.columns = farm_data.columns.str.strip()

# Data visualization
# plot the histogram
sns.histplot(farm_data['Net cash farm income'])
plt.title('Distribution of Net cash farm income')
plt.show()

# Box plot of Cash receipts
sns.boxplot(y='Cash receipts', data=farm_data)
plt.title('Box plot of Cash receipts')
plt.show()

# Step 3: Removal of unwanted features and missing data handling
# Check for missing values
print(farm_data.isnull().sum())
# Remove rows with missing values
farm_data_cleaned = farm_data.dropna()

# Check if missing values have been removed
print(farm_data_cleaned.isnull().sum())

# Now, you can proceed with your analysis using the cleaned DataFrame
sns.boxplot(y='Cash receipts', data=farm_data_cleaned)
plt.title('Box plot of Cash receipts (Cleaned)')
plt.show()

# Step 4: Checking data distribution using statistical techniques
# Convert columns to numeric data type, ignoring errors
farm_data_numeric = farm_data.apply(pd.to_numeric, errors='coerce')

# Check for columns that could not be converted
print("Columns with non-numeric values:")
print(farm_data_numeric.columns[farm_data_numeric.isnull().any()])

# Drop rows with missing values after conversion
farm_data_numeric_cleaned = farm_data_numeric.dropna()

# Check skewness and kurtosis of numeric columns
print("Skewness:")
print(farm_data_numeric_cleaned.skew())
print("\nKurtosis:")
print(farm_data_numeric_cleaned.kurtosis())

#  Using Python libraries for data interpretation and visualization
import numpy as np

# Convert non-numeric values to NaN
farm_data_numeric = farm_data.apply(pd.to_numeric, errors='coerce')

# Drop rows with NaN values
farm_data_numeric_cleaned = farm_data_numeric.dropna()

# Calculate the correlation matrix


selected_variables = ['Year', 'Cash receipts', 'Crops', 'Animals and products', 
                      'Federal Government direct farm program payments', 
                      'Gross cash farm income', 'Cash expenses', 'Net cash farm income', 
                      'Total gross farm income', 'Total expenses', 'Net farm income', 
                      'Farm assets', 'Farm equity', 'Debt-to-equity', 'Debt-to-asset']

# Filter the dataframe to include only selected variables
selected_data = farm_data[selected_variables]

# Convert non-numeric values to NaN
selected_data_numeric = selected_data.apply(pd.to_numeric, errors='coerce')

# Drop rows with NaN values
selected_data_numeric_cleaned = selected_data_numeric.dropna()

# Calculate the correlation matrix
corr_matrix_selected = selected_data_numeric_cleaned.corr()

# Plot the heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix_selected, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix of Selected Variables')
plt.show()

# Step 6: Splitting Dataset into Train and Test using sklearn
X = farm_data.drop('Net cash farm income', axis=1)
y = farm_data['Net cash farm income']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Step 7: Training a model using Classification techniques
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

# Drop non-numeric columns if necessary
# selected_numeric_data = selected_data_numeric.drop(columns=['Year'])


numeric_data = farm_data.select_dtypes(include='number')

# Convert non-numeric values to NaN
numeric_data = farm_data_numeric_cleaned.copy()  # Assuming farm_data_numeric_cleaned contains only numeric data
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(numeric_data)
print("Shape of X_imputed:", X_imputed.shape)
print("Shape of y:", y.shape)

# since the dimensions of x ,y weren't matchching we 
# Check the number of samples in X_imputed and y
print("Number of samples in X_imputed:", X_imputed.shape[0])
print("Number of samples in y:", y.shape[0])

# Ensure the number of samples in X_imputed and y match
min_samples = min(X_imputed.shape[0], y.shape[0])
X_imputed = X_imputed[:min_samples]
y = y[:min_samples]

# Check the shape again to confirm the matching number of samples
print("Shape of X_imputed after adjustment:", X_imputed.shape)
print("Shape of y after adjustment:", y.shape)

# Now, you can proceed with further processing or analysis using X_imputed.

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.2, random_state=42)

# Scale the features if necessary
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Shape of X_train:", X_train.shape)
print("Shape of y_train:", y_train.shape)
print("X_train:", X_train)
print("y_train:", y_train)

#Linear Regression
from sklearn.linear_model import LinearRegression

# Instantiate the Linear Regression model
linreg = LinearRegression()

# Fit the model to the training data
linreg.fit(X_train, y_train)

# Predict on the test data
y_pred = linreg.predict(X_test)


from sklearn.metrics import mean_squared_error

# Instantiate the Linear Regression model
linreg = LinearRegression()

# Define the parameter grid for tuning
param_grid = {'fit_intercept': [True, False]}

# Perform grid search with cross-validation
grid_search = GridSearchCV(linreg, param_grid, cv=4)

# Fit the grid search to the training data
from sklearn.model_selection import LeaveOneOut

loo = LeaveOneOut()
grid_search = GridSearchCV(linreg, param_grid, cv=loo)

grid_search.fit(X_train, y_train)
print("Best parameters:", grid_search.best_params_)

# Evaluate the model
y_pred = grid_search.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)

# model seems to be doing well in terms of predictive accuracy compared to a simple baseline model that predicts the mean of the target variable for all samples.

