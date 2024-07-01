import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Paths
input_csv_path = 'data/raw/student_performance_classification.csv'
output_csv_path = 'data/processed/student_performance_pca.csv'

# Read the CSV file
df = pd.read_csv(input_csv_path)

# Convert 'Performance_Label' to numeric
df['Performance_Label'] = df['Performance_Label'].map({'Pass': 1, 'Fail': 0})

# Selecting features for scaling and PCA
features = ['Math_Score', 'English_Score', 'Science_Score']
x = df[features]

# Standardizing the features
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)

# Applying PCA
pca = PCA(n_components=3)
x_pca = pca.fit_transform(x_scaled)

# Creating a DataFrame with the PCA results
df_pca = pd.DataFrame(data=x_pca, columns=['PCA1', 'PCA2','PCA3'])

# Adding the Student_ID and Performance_Label back to the DataFrame
df_pca['Student_ID'] = df['Student_ID']
df_pca['Performance_Label'] = df['Performance_Label']

# Reordering columns
df_pca = df_pca[['Student_ID', 'PCA1', 'PCA2','PCA3','Performance_Label']]

# Writing the processed data to a new CSV file
df_pca.to_csv(output_csv_path, index=False)

print(f'Processed data saved to {output_csv_path}')
