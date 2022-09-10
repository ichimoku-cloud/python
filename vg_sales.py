import pandas as pd 
from matplotlib import pyplot as plt 

# Load your data
df = pd.read_csv('vgsales.csv') # Reading the data

# Quick calculations and adding to existing dataset
df['Publisher NA Average'] = df.groupby('Publisher')['NA_Sales'].transform(lambda x: x.mean())


# Filtering and organizing your dataset
filter = pd.DataFrame(df, columns = ["Publisher", "Genre", "Name", "NA_Sales", "Publisher NA Average"]) # Organizing data into a seperate dataset
filter2 = filter[(filter['Publisher'] == 'Nintendo') | (filter['Publisher'] == 'Activision')]   #Filtering by different publishers

# Running the summary statistics
stats = pd.DataFrame(filter2.groupby('Publisher')['NA_Sales'].agg(['sum', 'mean', 'max', 'min']))


# Visualizing the data
aggregation = pd.DataFrame(filter2.groupby('Publisher')['NA_Sales'].sum()).plot(kind="bar") # Aggregating and plotting data
#mean = pd.DataFrame(filter2.groupby('Publisher')['NA_Sales'].mean()).plot(kind="bar") # Aggregating and plotting data
#max = pd.DataFrame(filter2.groupby('Publisher')['NA_Sales'].mean()).plot(kind="bar") # Aggregating and plotting data
#max = pd.DataFrame(filter2.groupby('Publisher')['NA_Sales'].mean()).plot(kind="bar") # Aggregating and plotting data

print(stats)
plt.xlabel("Publisher")
plt.ylabel("NA Sales")
plt.title("Total Publisher North American Sales")
plt.tight_layout()
plt.show()

# Checking unique column categories for publishers
print(filter["Publisher"].unique())
