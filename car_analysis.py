# car_analysis.py

# Step 1: Import Libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Step 2: Load the Data
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/car/car.data"
columns = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'class']
car_data = pd.read_csv(url, names=columns)

# Display the first few rows of the dataset
print("First few rows of the dataset:")
print(car_data.head())

# Step 3: Data Cleaning
# Check for Missing Values
print("\nMissing values in each column:")
print(car_data.isnull().sum())

# Convert Categorical Variables to Numeric
car_data = pd.get_dummies(car_data, drop_first=True)

# Check Data Types
print("\nData types of each column:")
print(car_data.dtypes)

# Step 4: Data Manipulation
# Basic Descriptive Statistics
print("\nDescriptive Statistics:")
print(car_data.describe())

# Group by Class
class_group = car_data.groupby('class_good').mean()
print("\nMean values by class:")
print(class_group)

# Step 5: Statistical Analysis
# Correlation Matrix
correlation = car_data.corr()
print("\nCorrelation Matrix:")
print(correlation)

# Visualize the Correlation Matrix
plt.figure(figsize=(10, 8))
sns.heatmap(correlation, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

# Step 6: Visualization
# Count Plot of Classes
sns.countplot(x='class_good', data=car_data)
plt.title('Count of Each Car Class')
plt.xlabel('Car Class')
plt.ylabel('Count')
plt.show()

# Bar Plot for Average Buying Price by Class
avg_buying_price = car_data.groupby('class_good')['buying'].mean().reset_index()
sns.barplot(x='class_good', y='buying', data=avg_buying_price)
plt.title('Average Buying Price by Car Class')
plt.xlabel('Car Class')
plt.ylabel('Average Buying Price')
plt.show()
