# xgboost model
import xgboost as xgb
from xgboost import plot_importance, plot_tree 
import numpy as np
import pandas as pd 
import seaborn as sns 
import matplotlib.pyplot as plt 

# load dataset
df = pd.read_csv('Datasets/time_series.csv', index_col=[0], parse_dates=[0])   # index and parse dates allow plots to visualize indexes

# set plot scheme
color_pal = ["#F8766D", "#D39200", "#93AA00", "#00BA38", "#00C19F", "#00B9E3", "#619CFF", "#DB72FB"]
# show visual summary of data
plot = df.plot(style='.', figsize=(15,5), color=color_pal[0], title='Monthly Sales')

# summary of data
df.head(10)

# splitting the data between dates
split_date = '2011-01-01'
df_train = df.loc[df.index <= split_date].copy()
df_test = df.loc[df.index > split_date].copy()

# visualising training and testing data
        plot1 = df_test \
            .rename(columns={'Monthly Sales': 'Test Set'}) \
            .join(df_train.rename(columns={'Monthly Sales': 'Training Set'}), how='outer') \
            .plot(figsize=(15,5), title='Monthly Sales', style='.')

# create features for model
def create_features(df, label=None):
    """
    Creates time series features from datetime index
    """
    df['date'] = df.index
    df['hour'] = df['date'].dt.hour
    df['dayofweek'] = df['date'].dt.dayofweek
    df['quarter'] = df['date'].dt.quarter
    df['month'] = df['date'].dt.month
    df['year'] = df['date'].dt.year
    df['dayofyear'] = df['date'].dt.dayofyear
    df['dayofmonth'] = df['date'].dt.day
    df['weekofyear'] = df['date'].dt.weekofyear
    
    X = df[['hour','dayofweek','quarter','month','year',
           'dayofyear','dayofmonth','weekofyear']]
    if label:
        y = df[label]
        return X, y
    return X


# creating training and test data for model
X_train, y_train = create_features(df_train, label='Monthly Sales')  # creates training data, adding additional features (columns)
X_test, y_test = create_features(df_test, label='Monthly Sales')     # creates test data, adding additional features (columns)


# create xgboost model 
model = xgb.XGBRegressor(n_estimators=1000)

# run the model 
model.fit(X_train, y_train,
        eval_set=[(X_train, y_train), (X_test, y_test)],   #evaluation set between X, y -train, test
        early_stopping_rounds=50,
       verbose=False) # Change verbose to True if you want to see it train



# setting index to make it easier to plot
df_all.set_index(['Monthly Sales', 'Monthly_Prediction'])

        # feature importance bar plot (visual)
        plot2 = plot_importance(model, height=0.9)

        # monthly sales vs. model prediction (visual)
        df_test['Monthly_Prediction'] = model.predict(X_test)
        df_all = pd.concat([df_test, df_train], sort=False)
        plot3 = df_all[['Monthly Sales','Monthly_Prediction']].plot(figsize=(15, 5))

        # plot the forecast with the actuals (visual)
        f, ax = plt.subplots(1)
        f.set_figheight(5)
        f.set_figwidth(15)
        _ = df_all[['Monthly_Prediction','Monthly Sales']].plot(ax=ax,
                                                    style=['-','-'])
        ax.set_xbound(lower='01-01-2011', upper='02-01-2012')
        ax.set_ylim(0, 711000)
        plot4 = plt.suptitle('January 2011 Forecast vs Actuals')



# model performance summary
from sklearn.metrics import mean_squared_error, mean_absolute_error
# mean squared error
mean_squared_error(y_true=df_test['Monthly Sales'],
                   y_pred=df_test['Monthly_Prediction'])


# mean absolute error
def mean_absolute_percentage_error(y_true, y_pred): 
    """Calculates MAPE given y_true and y_pred"""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true))

mean_absolute_percentage_error(y_true=df_test['Monthly Sales'],
                   y_pred=df_test['Monthly_Prediction'])

# mape function
mape = mean_absolute_percentage_error(df_test['Monthly Sales'], df_test['Monthly_Prediction'])
mape


