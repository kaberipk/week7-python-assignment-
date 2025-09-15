# Cell 1: Import libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
import numpy as np

# Set style for better looking plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# Cell 2: Load and explore dataset
print("="*50)
print("TASK 1: LOAD AND EXPLORE THE DATASET")
print("="*50)

try:
    iris = load_iris()
    df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    df['species'] = iris.target
    df['species'] = df['species'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})
    
    print("Dataset loaded successfully!")
    print(f"Dataset shape: {df.shape}")
    
except Exception as e:
    print(f"Error loading dataset: {e}")

# Cell 3: Display data and check structure
print("\nFirst 5 rows:")
df.head()

print("\nDataset information:")
df.info()

print("\nMissing values:")
df.isnull().sum()

# Cell 4: Basic analysis
print("="*50)
print("TASK 2: BASIC DATA ANALYSIS")
print("="*50)

print("\nBasic statistics:")
df.describe()

print("\nMean values by species:")
species_stats = df.groupby('species').mean()
species_stats

# Cell 5: Create visualizations
print("="*50)
print("TASK 3: DATA VISUALIZATION")
print("="*50)

fig, axes = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle('Iris Dataset Analysis', fontsize=16, fontweight='bold')

# Line chart
axes[0, 0].plot(df.index[:50], df['sepal length (cm)'][:50], label='Sepal Length', marker='o')
axes[0, 0].plot(df.index[:50], df['petal length (cm)'][:50], label='Petal Length', marker='s')
axes[0, 0].set_title('Line Chart: Sepal vs Petal Length')
axes[0, 0].set_xlabel('Sample Index')
axes[0, 0].set_ylabel('Length (cm)')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Bar chart
species_means = df.groupby('species').mean()
x = np.arange(len(species_means.index))
width = 0.2

axes[0, 1].bar(x - width, species_means['sepal length (cm)'], width, label='Sepal Length')
axes[0, 1].bar(x, species_means['sepal width (cm)'], width, label='Sepal Width')
axes[0, 1].bar(x + width, species_means['petal length (cm)'], width, label='Petal Length')
axes[0, 1].set_title('Bar Chart: Average Measurements by Species')
axes[0, 1].set_xlabel('Species')
axes[0, 1].set_ylabel('Measurement (cm)')
axes[0, 1].set_xticks(x)
axes[0, 1].set_xticklabels(species_means.index)
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Histogram
axes[1, 0].hist(df['sepal length (cm)'], bins=15, alpha=0.7, edgecolor='black')
axes[1, 0].set_title('Histogram: Distribution of Sepal Length')
axes[1, 0].set_xlabel('Sepal Length (cm)')
axes[1, 0].set_ylabel('Frequency')
axes[1, 0].grid(True, alpha=0.3)

# Scatter plot
colors = {'setosa': 'red', 'versicolor': 'green', 'virginica': 'blue'}
for species in df['species'].unique():
    species_data = df[df['species'] == species]
    axes[1, 1].scatter(species_data['sepal length (cm)'], 
                      species_data['petal length (cm)'], 
                      label=species, 
                      alpha=0.7)
axes[1, 1].set_title('Scatter Plot: Sepal Length vs Petal Length')
axes[1, 1].set_xlabel('Sepal Length (cm)')
axes[1, 1].set_ylabel('Petal Length (cm)')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Cell 6: Findings
print("="*50)
print("FINDINGS AND OBSERVATIONS")
print("="*50)
print("""
Key Findings:
1. Dataset is clean with no missing values
2. Clear separation between species in measurements
3. Setosa has smallest measurements, Virginica the largest
4. Strong correlation between sepal and petal length
""")