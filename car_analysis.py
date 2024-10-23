import pandas as pd  # Import pandas for data manipulation and analysis
import seaborn as sns  # Import seaborn for statistical data visualization
import matplotlib.pyplot as plt  # Import matplotlib for plotting graphs
import zipfile  # Import zipfile to handle ZIP file extraction
import os  # Import os for file path manipulation

# Set backend for matplotlib to avoid issues with certain environments
import matplotlib 
matplotlib.use('Agg')  # Use 'Agg' backend for non-GUI environments (like servers). 
# Allows you to generate plots in environments where a display is not available.

# Define the path to the ZIP file and the directory where it will be extracted
zip_file_path = 'C:/Users/shubs/dataset/dataset/car+evaluation.zip'
extraction_path = 'C:/Users/shubs/dataset/dataset/car-evaluation/'

# Extract the contents of the ZIP file to the specified directory
with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
    zip_ref.extractall(extraction_path)

# Load the Data
data_path = os.path.join(extraction_path, 'car.data')  # Create the full path to the data file
columns = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'class']  # Define column names

# Read the dataset into a pandas DataFrame and apply the column names
car_data = pd.read_csv(data_path, names=columns)

# Data Cleaning
print("First few rows of the dataset:")  # Print a message to indicate the next output
print(car_data.head())  # Display the first few rows of the DataFrame

print("\nMissing values in each column:")  # Print a message for the next output
print(car_data.isnull().sum())  # Display the count of missing values in each column

# Convert categorical data into numerical format using one-hot encoding
car_data = pd.get_dummies(car_data, drop_first=True)  # Convert categorical variables to dummy variables

# Check the data types of each column to ensure they are correct
print("\nData types of each column:")
print(car_data.dtypes)  # Display the data types of each column

# Ensure 'class_good' column exists; if not, notify the user
if 'class_good' not in car_data.columns:
    print("class_good column is missing. Check your DataFrame columns.")
else:
    # Data Manipulation
    print("\nDescriptive Statistics:")  # Print a message for the next output
    print(car_data.describe())  # Display descriptive statistics of the DataFrame

    # Group data by 'class_good' and calculate average values for each group
    class_group = car_data.groupby('class_good').mean()  # Group by 'class_good' and compute mean
    print("\nMean values by class:")
    print(class_group)  # Display the mean values for each car class

    # Statistical Analysis: Calculate the correlation matrix
    correlation = car_data.corr()  # Compute the correlation matrix
    print("\nCorrelation Matrix:")
    print(correlation)  # Display the correlation matrix

    # Visualize the Correlation Matrix with a heatmap
    plt.figure(figsize=(10, 8))  # Create a new figure with specified size
    sns.heatmap(correlation, annot=True, cmap='coolwarm')  # Create a heatmap of the correlation matrix
    plt.title('Correlation Matrix')  # Set the title for the heatmap
    plt.savefig('correlation_matrix.png')  # Save the heatmap as a PNG file
    plt.close()  # Close the current figure to free up memory

    # Visualization of the count of each car class
    plt.figure(figsize=(10, 6))  # Create a new figure
    sns.countplot(x='class_good', data=car_data)  # Create a count plot for car classes
    plt.title('Count of Each Car Class')  # Set the title for the count plot
    plt.xlabel('Car Class')  # Label for the x-axis
    plt.ylabel('Count')  # Label for the y-axis
    plt.savefig('count_plot.png')  # Save the count plot as a PNG file
    plt.close()  # Close the current figure

    # Calculate the average buying price for each car class
    avg_buying_price = car_data.groupby('class_good').mean()[['buying_low', 'buying_med', 'buying_vhigh']].reset_index()
    # Group by 'class_good' and compute the mean for buying price categories, then reset the index

    # Create a bar plot to visualize the average buying price
    plt.figure(figsize=(10, 6))  # Create a new figure
    avg_buying_price_melted = avg_buying_price.melt(id_vars='class_good', var_name='buying_category', value_name='average_value')
    # Reshape the data for plotting (from wide to long format)
    sns.barplot(x='class_good', y='average_value', hue='buying_category', data=avg_buying_price_melted)  # Create a bar plot
    plt.title('Average Buying Price by Car Class')  # Set the title for the bar plot
    plt.xlabel('Car Class')  # Label for the x-axis
    plt.ylabel('Average Buying Price')  # Label for the y-axis
    plt.savefig('average_buying_price.png')  # Save the average buying price plot as a PNG file
    plt.close()  # Close the current figure
