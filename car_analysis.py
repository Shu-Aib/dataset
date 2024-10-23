import pandas as pd  # For data manipulation
import seaborn as sns  # For data visualization
import matplotlib.pyplot as plt  # For creating plots
import matplotlib
import zipfile
import os

# Use a non-interactive backend
matplotlib.use('Agg')

# Define the path to the zip file and the extraction directory
zip_file_path = 'C:/Users/shubs/dataset/dataset/car+evaluation.zip'
extraction_path = 'C:/Users/shubs/dataset/dataset/car-evaluation/'  # Path for extraction

# Extract the zip file
with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
    zip_ref.extractall(extraction_path)

# Load the Data
# Specify the path to the dataset
data_path = os.path.join(extraction_path, 'car.data')  # Adjust if needed based on the extracted files
columns = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'class']

# Read the dataset and name the columns
car_data = pd.read_csv(data_path, names=columns)

# Show the first few rows to see what the data looks like
print("First few rows of the dataset:")
print(car_data.head())

# Data Cleaning
# Check for missing values in the dataset
print("\nMissing values in each column:")
print(car_data.isnull().sum())  # Count of missing values per column

# Convert categorical data into numerical format
car_data = pd.get_dummies(car_data, drop_first=True)  # Drop the first category to avoid redundancy

# Check the data types to ensure they are correct
print("\nData types of each column:")
print(car_data.dtypes)  # Print data types for each column

# Data Manipulation
# Get basic statistics about the dataset
print("\nDescriptive Statistics:")
print(car_data.describe())  # Summary statistics for numerical columns

# Group by 'class_good' and calculate average values
class_group = car_data.groupby('class_good').mean()
print("\nMean values by class:")
print(class_group)  # Display average values for each car class

# Statistical Analysis
# Calculate the correlation between numeric columns
correlation = car_data.corr()
print("\nCorrelation Matrix:")
print(correlation)  # Print correlation coefficients

# Visualize the Correlation Matrix with a heatmap
plt.figure(figsize=(10, 8))  # Set the size of the plot
sns.heatmap(correlation, annot=True, cmap='coolwarm')  # Create a heatmap with correlation values
plt.title('Correlation Matrix')  # Title for the heatmap
plt.savefig('correlation_matrix.png')  # Save the heatmap
plt.close()  # Close the plot to free memory

# Visualization
# Create a count plot to show the number of cars in each class
plt.figure(figsize=(10, 6))  # Set the size of the plot
sns.countplot(x='class_good', data=car_data)  # Count plot for car classes
plt.title('Count of Each Car Class')  # Title for the plot
plt.xlabel('Car Class')  # Label for the x-axis
plt.ylabel('Count')  # Label for the y-axis
plt.savefig('count_plot.png')  # Save the count plot
plt.close()  # Close the plot to free memory

# Calculate the average buying price for each car class
# Since 'buying' is converted to dummy variables, we can calculate the mean of those
avg_buying_price = car_data.groupby('class_good').mean()[['buying_low', 'buying_med', 'buying_vhigh']].reset_index()  # Group and calculate mean

# Create a bar plot to visualize the average buying price
plt.figure(figsize=(10, 6))  # Set the size of the plot
avg_buying_price_melted = avg_buying_price.melt(id_vars='class_good', var_name='buying_category', value_name='average_value')
sns.barplot(x='class_good', y='average_value', hue='buying_category', data=avg_buying_price_melted)  # Bar plot for average prices
plt.title('Average Buying Price by Car Class')  # Title for the plot
plt.xlabel('Car Class')  # Label for the x-axis
plt.ylabel('Average Buying Price')  # Label for the y-axis
plt.savefig('average_buying_price.png')  # Save the bar plot
plt.close()  # Close the plot to free memory
