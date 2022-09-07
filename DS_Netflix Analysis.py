from re import X
from numpy import sort
import pandas as pd  
from matplotlib import pyplot as plt 

# Loading in the Data
df = pd.read_csv('Datasets/netflix_titles.csv')      


# Cleaning up data and setting variables
subset = pd.DataFrame(df, columns=['title', 'rating', 'country', 'release_year'])   # Create a dataframe with specific columns
month = df['Month'] = pd.DatetimeIndex(df['date_added']).month    # Convert date of Netflix title's added to month


# Summarizing Data
df2 = pd.DataFrame(subset.groupby(['country'])['title'].agg('count'))
#.agg('count')).sort_values(by=subset, ascending=False)
#print(df2)
print(df.head(50))
#release_ratingbyyear = df.groupby('release_year')['rating'].count().sort_values(ascending=False)      # Counting the number of rating types by year


# Putting into a dataset for easy reading

# Visualizing the data
#plt.bar(x2['title'], 0.8)
#plt.xticks(country_title_df['country'])
#plt.show()




#print(country_title_df)
#print(x)
#print(x2.head(50))

#print(release_ratingbyyear.head(15))
# Printing the output of the data
#print(subset.head(30))
#print(country_titlecount.head(15))
#print(country_titlecount)
#print(country_title_df.head(15))
#print(country_title_df.columns)
#print(df.columns)