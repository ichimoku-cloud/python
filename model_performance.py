# Tensorflow Keras (Sequential Model)
from tensorflow.keras.layers import Dense, Dropout, Activation
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam 
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# load in dataset
dataset = pd.read_csv('Datasets/Sample_Data.csv')

# Set variable and target values for model
X = crime_data.drop(['total_crime_reported_per_1_million_res'], axis=1).values 
Y = crime_data[['total_crime_reported_per_1_million_res']].values

# Seperate training and testing data
from sklearn.model_selection import train_test_split 
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=25)

# Model Parameters 
epochs = 100
learning_rate = 0.01
dropout_rate = 0.02

# Create model
def create_model(learning_rate, dropout_rate):
    model = Sequential()
    model.add(Dense(100, input_dim=X_train.shape[1], activation='relu'))
    model.add(Dropout(dropout_rate))
    model.add(Dense(50, activation='relu'))
    model.add(Dropout(dropout_rate))
    model.add(Dense(25, activation='relu'))
    model.add(Dropout(dropout_rate))
    model.add(Dense(1))
                    
                            # alternative nn design (#1)
                                            models.add(Dense(10, activation = 'relu', input_dim = input_size))
                                            models.add(Dense(units = 5, activation = 'sigmoid'))
                                            models.add(Dropout(0.2))
                                            # Adding the third hidden layer
                                            models.add(Dense(units = 5, activation = 'relu'))
                                            models.add(Dropout(0.2))

                                            models.add(Dense(units = 5, activation = 'relu'))
                                            models.add(Dense(units = 5, activation = 'relu'))
                                            # Adding the output layer
                                            models.add(Dense(units = 1))
                                            models.compile(optimizer = tf.keras.optimizers.SGD(learning_rate=0.02), loss = 'mean_squared_error')

                            # alternative nn design (#2)
                                            model.add(Dense(units=500, input_dim=46, kernel_initializer='uniform', activation='relu'))
                                            model.add(Dropout(0.5))
                                            #Hidden layer 1
                                            model.add(Dense(units=200, kernel_initializer='uniform', activation='relu'))
                                            model.add(Dropout(0.5))
                                            #Output layer
                                            model.add(Dense(units=1, kernel_initializer='uniform', activation='sigmoid'))

# optimizer
    adam = Adam(learning_rate)
     # optimizers: SGD - gradient descent with momentum
     #             Adamax - variant of Adam optimiser - at times superior to adam
# compile model
    model.compile(loss='mean_squared_error', optimizer=adam, metrics=['mae'])
                # 'binary_crossentropy' -- for classification (probablistic loss)
                # 'mean_squared_error' -- for regression (regression loss)
    return model 
    
    # Initialize the model
    model = create_model(learning_rate, dropout_rate)

# Create model from training data
model_history = model.fit(X_train, Y_train, batch_size=1, epochs=epochs, validation_split=0.2, verbose=10)




--
# Prophet (Time Series Predictions)
from prophet import Prophet
from prophet.plot import plot_plotly, plot_components_plotly
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# load in dataset
dataset = pd.read_csv('Datasets/Sample_Data.csv')

# Create model
model = Prophet()

# Create model from training data
model_history = model.fit(x.reset_index() \
                           .rename(columns={'Date': 'ds',
                                            'IAP': 'y'}))
                 # Renaming columns as Prophet model specifies ds & y columns

# create future forecast ie. 15 days into the future
future = model.make_future_dataframe(periods=15)
forecast = model.predict(future)
forecast[['ds', 'yhat']]

        # plot prophet model
        plot = model.plot(forecast, figsize=(15, 5))
        plt.ylabel('IAP Revenue')
        plt.xlabel('Date')

        # plot weekly trends
        plot2 = model.plot_components(forecast, figsize=(15, 5))
        plt.title('IAP Revenue Forecast')

        # Interactive plot
        plot_plotly(model, forecast)




--
# random forest regression model
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# Seperate training and testing data
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.25, random_state = 0)

# Create model
model = RandomForestRegressor(n_estimators = 100, random_state = 0)

# Create model from training data
model_history = model.fit(X_train, Y_train)  
predictions = regressor.predict(X_test)

# New dataframe with actual & predicted values from random forest model
df2 = pd.DataFrame({'Actual':Y_test, 'Predicted':Y_pred})

# Evaluate model performance
from sklearn import metrics
errors = abs(Y_pred - Y_test)
mape = 100 * (errors / Y_test)
accuracy = 100 - np.mean(mape)

print('Mean Absolute Error:', metrics.mean_absolute_error(Y_test, Y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(Y_test, Y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(Y_test, Y_pred)))
print('Mean Absolute Error:', round(np.mean(errors), 2), 'degrees.')
print('Accuracy:', round(accuracy, 2), '%.')





--
# Machine Learning (Ensemble)
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.linear_model import Ridge, Lasso, BayesianRidge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor

from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from scipy.stats import probplot, boxcox
from scipy.special import inv_boxcox
import pylab

# load in dataset
dataset = pd.read_csv('Datasets/Sample_Data.csv')

# Set variable and target values for model
X = crime_data.drop(['total_crime_reported_per_1_million_res'], axis=1).values 
Y = crime_data[['total_crime_reported_per_1_million_res']].values

# Seperate training and testing data
from sklearn.model_selection import train_test_split 
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=25)

# Create model
model = StandardScaler()

# Create ensemble of models
models = {
    'ridge' : Ridge(),
    'xgboost' : XGBRegressor(),
    'catboost' : CatBoostRegressor(verbose=0),
    'lightgbm' : LGBMRegressor(),
    'gradient boosting' : GradientBoostingRegressor(),
    'lasso' : Lasso(),
    'random forest' : RandomForestRegressor(),
    'bayesian ridge' : BayesianRidge(),
    'support vector': SVR(),
    'knn' : KNeighborsRegressor(n_neighbors = 4)
}

# Create model from training data
for name, model in models.items():
    model.fit(X_train, y_train)
    print(f'{name} trained')

# Evaluate model performance
results = {}
kf = KFold(n_splits= 10)

for name, model in models.items():
    result = np.mean(np.sqrt(-cross_val_score(model, X_train, y_train, scoring = 'neg_mean_squared_error', cv= kf)))
    results[name] = result

for name, result in results.items():
    print(f"{name} : {round(result, 3)}")

results_df = pd.DataFrame(results, index=range(0,1)).T.rename(columns={0: 'MSE'}).sort_values('MSE', ascending=False)
results_df.Tranpose

# Plot model performance
plt.figure(figsize = (20, 6))
sns.barplot(x= results_df.index, y = results_df['MSE'], palette = 'summer')
plt.xlabel('Model')
plt.ylabel('MSE')
plt.title('MSE of different models');

# Combine prediction of all models
final_predictions = (
    0.20 * inv_boxcox(models['catboost'].predict(X_test), lam) +
    0.20 * inv_boxcox(models['xgboost'].predict(X_test), lam) +
    0.20 * inv_boxcox(models['lightgbm'].predict(X_test), lam) + 
    0.20 * inv_boxcox(models['random forest'].predict(X_test), lam) + 
    0.20 * inv_boxcox(models['gradient boosting'].predict(X_test), lam)
)

print(f'RMSE: {np.sqrt(mean_squared_error(inv_boxcox(y_test, lam), final_predictions))}')
print(f'R-square: {r2_score(inv_boxcox(y_test, lam), final_predictions)}')


