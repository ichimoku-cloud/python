# linear regression model
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split 
import matplotlib.pyplot as plt 
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import numpy as np 
import pandas as pd 


# importing dataset
dataset = pd.read_csv("Datasets/kc_house_data.csv")
space=dataset['sqft_living15']
price=dataset['price']

x = np.array(space).reshape(-1, 1)
y = np.array(price)


# creating training and testing data
xtrain, xtest, ytrain, ytest = train_test_split(x,y, train_size=1000, test_size=1/15, random_state=0)

# create model
model = LinearRegression()
model.fit(xtrain, ytrain)

#Predicting the prices
pred = model.predict(xtest)

# visualizing the training test results 
plt.scatter(xtrain, ytrain, color= 'gray')
plt.plot(xtrain, regressor.predict(xtrain), color = 'blue')
plt.title ("Training Data")
plt.xlabel("Space")
plt.ylabel("Price")
plt.show()

# visualizing the test results 
plt.scatter(xtest, ytest, color= 'green')
plt.plot(xtrain, regressor.predict(xtrain), color = 'red')
plt.title("Testing Data")
plt.xlabel("Space")
plt.ylabel("Price")
plt.show()


--
# correlation feature heatmap
plt.figure(figsize=(12, 7))
sns.heatmap(dataset.corr(), annot=True, cmap="YlGnBu")
plt.title("Housing Correlation Heatmap")
plt.show()





