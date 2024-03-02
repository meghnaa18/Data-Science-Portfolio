import pandas as pd
import matplotlib.pyplot as plt
import sweetviz
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler
from scipy.cluster.hierarchy import linkage,dendrogram
from sklearn.cluster import AgglomerativeClustering
from sklearn import  metrics
from clusteval import clusteval
import numpy as np
import psycopg2

from sqlalchemy import create_engine

uni = pd.read_excel(r"C:\Users\Meghna\University_Clustering.xlsx")

# Define your connection parameters
dbname = 'db01'
user = 'postgres'
password = 'Meghna1899'
host = 'localhost'  # If it's localhost, you can use 'localhost'
port = '5432'

# Establish a connection to the database
try:
    conn = psycopg2.connect(dbname=dbname, user=user, password=password, host=host, port=port)
    print("Connected to the database.")
    
    # Create a cursor object using the connection
    cursor = conn.cursor()
    
    # Example query
    cursor.execute("SELECT version();")
    
    # Fetch and print the result
    record = cursor.fetchone()
    print("You are connected to - ", record, "\n")
    # Close the cursor and connection
    cursor.close()
    conn.close()
    
except psycopg2.Error as e:
    print("Error connecting to the database:", e)
    
    from sqlalchemy import create_engine

# Define your connection parameters
dbname = 'db01'
user = 'postgres'
password = 'Meghna1899'
host = 'localhost'
port = '5432'

# Create SQLAlchemy engine
engine = create_engine(f'postgresql://{user}:{password}@{host}:{port}/{dbname}')

# Push DataFrame to a table in PostgreSQL
try:
    uni.to_sql('uni_table', engine, if_exists='replace', index=False)
    print("DataFrame successfully pushed to PostgreSQL table.")
except Exception as e:
    print("Error pushing DataFrame to PostgreSQL:", e)
# Define your SQL query
sql_query = "SELECT * FROM uni_table"

# Execute SQL query and load result into a DataFrame
try:
    df = pd.read_sql_query(sql_query, engine)
    print("DataFrame successfully created from SQL query.")
except Exception as e:
    print("Error creating DataFrame from SQL query:", e)
# Data types
df.info()
df.describe()

df.drop(['UnivID'],axis=1,inplace = True)
df.info()

#AutoEDA
import sweetviz
my_report = sweetviz.analyze([df,"df"])

my_report.show_html('Report.html')

#EDA report

df.plot(kind = 'box' , subplots = True, sharey = False, figsize = (15,8))

plt.subplots_adjust(wspace = 0.75)
plt.show()

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Print column names to verify
print(df.columns)

# Update column name if it's different
numerical_column = 'GradRate'  # Assuming 'GradRate' is the correct numerical column name

# Handling Missing Values
imputer = SimpleImputer(strategy='mean')
df[[numerical_column]] = imputer.fit_transform(df[[numerical_column]])

# Continue with the rest of your preprocessing steps...

# Update categorical column name if it's different
categorical_column = 'State'  # Assuming 'State' is the correct categorical column name

# One-hot encode categorical columns
onehot_encoder = OneHotEncoder()
encoded_categorical = onehot_encoder.fit_transform(df[[categorical_column]])

# Update numerical column name if it's different
numerical_column = 'Expenses'  # Assuming 'Expenses' is the correct numerical column name

# Scale numerical columns using StandardScaler
scaler = StandardScaler()
scaled_numerical = scaler.fit_transform(df[[numerical_column]])


# Concatenating transformed features
processed_features = np.concatenate((encoded_categorical.toarray(), scaled_numerical), axis=1)

# Define numerical and categorical columns
numerical_column = 'Expenses'  # Assuming 'Expenses' is the correct numerical column name
categorical_column = 'State'    # Assuming 'State' is the correct categorical column name

# Create a pipeline for numerical features
numerical_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

# Create a pipeline for categorical features
categorical_pipeline = Pipeline([
    ('onehot', OneHotEncoder())
])

# Combine numerical and categorical pipelines using ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_pipeline, [numerical_column]),
        ('cat', categorical_pipeline, [categorical_column])
    ])

# Example usage of the preprocessor in a final pipeline
# You can add additional steps like model training after preprocessing
# For demonstration purposes, we'll just transform the features

# Importing necessary libraries
from sklearn.cluster import AgglomerativeClustering
import numpy as np

# Preprocess the data using the defined preprocessing steps
processed_features = preprocessor.fit_transform(df)

# Build the agglomerative clustering model
# Define the number of clusters
n_clusters = 3  # Adjust the number of clusters as needed

# Initialize the agglomerative clustering model
agg_cluster = AgglomerativeClustering(n_clusters=n_clusters)

# Convert sparse matrix to dense numpy array
processed_features_dense = processed_features.toarray()

# Fit the model to the processed features
agg_cluster.fit(processed_features_dense)


# Get the labels assigned to each data point
cluster_labels = agg_cluster.labels_

# Print the cluster labels
print("Cluster Labels:", cluster_labels)
# Fit the model to the processed features
agg_cluster.fit(processed_features_dense)

# Convert linkage matrix to a numpy array with double-precision floating-point numbers
linkage_matrix = linkage(processed_features_dense, method='ward')

# Plot the dendrogram
plt.figure(figsize=(16, 8))
dendrogram(
    linkage_matrix,
    leaf_rotation=90.,
    leaf_font_size=8.,
)
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('Index')
plt.ylabel('Distance')
plt.show()
from sklearn.cluster import AgglomerativeClustering

# Define the number of clusters
n_clusters = 3

# Initialize the AgglomerativeClustering model
agg_cluster = AgglomerativeClustering(n_clusters=n_clusters)

# Fit the model to the processed features
agg_cluster.fit(processed_features_dense)

# Get the cluster labels assigned to each data point
cluster_labels = agg_cluster.labels_

# Print the cluster labels
print("Cluster Labels:", cluster_labels)

import pandas as pd

# Add cluster labels to the original DataFrame
df['Cluster'] = cluster_labels

# Display the DataFrame with cluster labels
print(df.head())
# Concatenate the original DataFrame with the cluster labels
clustered_df = pd.concat([df, pd.DataFrame({'Cluster': cluster_labels})], axis=1)

# Display the clustered DataFrame
print(clustered_df.head())