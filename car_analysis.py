# Import necessary libraries
import pandas as pd  # For data manipulation
import seaborn as sns  # For data visualization
import matplotlib.pyplot as plt  # For creating plots

# Load the Data
# Specify the URL of the dataset and the column names
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/car/car.data"
columns = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'class'] 

# Read the dataset from the URL and name the columns
car_data = pd.read_csv(url, names=columns)  

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
plt.show()  # Show the heatmap

# Visualization
# Create a count plot to show the number of cars in each class
sns.countplot(x='class_good', data=car_data)  # Count plot for car classes
plt.title('Count of Each Car Class')  # Title for the plot
plt.xlabel('Car Class')  # Label for the x-axis
plt.ylabel('Count')  # Label for the y-axis
plt.show()  # Show the count plot

# Bar Plot for Average Buying Price by Class
# Calculate the average buying price for each car class
avg_buying_price = car_data.groupby('class_good')['buying'].mean().reset_index()  # Group and calculate mean

# Create a bar plot to visualize the average buying price
sns.barplot(x='class_good', y='buying', data=avg_buying_price)  # Bar plot for average prices
plt.title('Average Buying Price by Car Class')  # Title for the plot
plt.xlabel('Car Class')  # Label for the x-axis 
plt.ylabel('Average Buying Price')  # Label for the y-axis 
plt.show()  # Show the bar plot 
