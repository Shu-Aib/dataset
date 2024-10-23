# Import necessary libraries
import pandas as pd  # For data manipulation and analysis
import seaborn as sns  # For data visualization
import matplotlib.pyplot as plt  # For plotting graphs

# Load the Data
# Define the URL of the dataset and the column names
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/car/car.data"
columns = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'class'] 

# Read the dataset from the URL and assign the column names
car_data = pd.read_csv(url, names=columns) 

# Display the first few rows of the dataset to understand its structure
print("First few rows of the dataset:") 
print(car_data.head()) 

# Data Cleaning 
# Check for missing values in each column
print("\nMissing values in each column:") 
print(car_data.isnull().sum())  # This will show the count of missing values per column

# Convert categorical variables into numerical format using one-hot encoding
car_data = pd.get_dummies(car_data, drop_first=True)  # Avoiding dummy variable trap by dropping the first category

# Check the data types of each column to ensure they are correct
print("\nData types of each column:")
print(car_data.dtypes)  # This will print the data type of each column

# Data Manipulation 
# Generate basic descriptive statistics for the dataset
print("\nDescriptive Statistics:") 
print(car_data.describe())  # This provides summary statistics for numerical columns

# Group data by the 'class_good' column and calculate mean values for each group
class_group = car_data.groupby('class_good').mean()
print("\nMean values by class:") 
print(class_group)  # Display the mean values for each class

# Statistical Analysis 
# Calculate the correlation matrix for the dataset
correlation = car_data.corr()
print("\nCorrelation Matrix:") 
print(correlation)  # Print the correlation coefficients between numeric columns

# Visualize the Correlation Matrix using a heatmap
plt.figure(figsize=(10, 8))  # Set the figure size for the plot
sns.heatmap(correlation, annot=True, cmap='coolwarm')  # Create a heatmap with annotations
plt.title('Correlation Matrix')  # Set the title of the heatmap
plt.show()  # Display the heatmap

# Visualization
# Create a count plot to visualize the distribution of each car class
sns.countplot(x='class_good', data=car_data)  # Count plot of the 'class_good' variable
plt.title('Count of Each Car Class')  # Set the title for the count plot
plt.xlabel('Car Class')  # Label for the x-axis
plt.ylabel('Count')  # Label for the y-axis
plt.show()  # Display the count plot

# Bar Plot for Average Buying Price by Class
# Group by 'class_good' and calculate the mean buying price for each class
avg_buying_price = car_data.groupby('class_good')['buying'].mean().reset_index()  # Calculate mean buying price

# Create a bar plot to visualize the average buying price by car class
sns.barplot(x='class_good', y='buying', data=avg_buying_price)  # Bar plot of average buying price
plt.title('Average Buying Price by Car Class')  # Set the title for the bar plot
plt.xlabel('Car Class')  # Label for the x-axis
plt.ylabel('Average Buying Price')  # Label for the y-axis
plt.show()  # Display the bar plot