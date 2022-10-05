
# imports
from sklearn import metrics # all metrics for model performance
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score # accuracy and classification of model performance
import matplotlib.pyplot as plt # for model visualization


# model history 
from sklearn import metrics
y_true = [...] # Your real values / test labels
y_pred = [...] # The predictions from your ML / RF model

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, Y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, Y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, Y_pred)))

                print('Mean Absolute Error (MAE):', metrics.mean_absolute_error(y_true, y_pred))
                print('Mean Squared Error (MSE):', metrics.mean_squared_error(y_true, y_pred))
                print('Root Mean Squared Error (RMSE):', metrics.mean_squared_error(y_true, y_pred, squared=False))
                print('Mean Absolute Percentage Error (MAPE):', metrics.mean_absolute_percentage_error(y_true, y_pred))
                print('Explained Variance Score:', metrics.explained_variance_score(y_true, y_pred))
                print('Max Error:', metrics.max_error(y_true, y_pred))
                print('Mean Squared Log Error:', metrics.mean_squared_log_error(y_true, y_pred))
                print('Median Absolute Error:', metrics.median_absolute_error(y_true, y_pred))
                print('R^2:', metrics.r2_score(y_true, y_pred))
                print('Mean Poisson Deviance:', metrics.mean_poisson_deviance(y_true, y_pred))
                print('Mean Gamma Deviance:', metrics.mean_gamma_deviance(y_true, y_pred))

# accuracy & classification
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
                print(confusion_matrix(y_test, prediction))
                print(classification_report(y_test,prediction))
                print(accuracy_score(y_test, prediction)*100)

scores = model.evaluate(X_test, y_test)
                print('Percentage Accuracy:')
                print(scores[1]*100)


# showing model performance examples
     
        # plotting feature importance
        plt.figure(num=None, figsize=(8, 5), dpi=80)
        feature_importances = pd.Series(model.feature_importances_, index = X.columns)
        feature_importances.nlargest(10).plot(kind='barh')
        plt.title('Fortnite (Mental State): Feature Importance')
        plt.show()
                        
                # model performance plots
                        # plotting loss function
                        loss=history.history['loss']
                        epoch = range(1, len(loss)+1)
                        plt.plot(epoch, loss, label='Training Loss')
                        plt.legend()
                        plt.show()

                        # plotting model error (mae)
                        import matplotlib.pyplot as plt 
                        plt.plot(model_history.history['mae'])
                        plt.plot(model_history.history['val_mae'])
                        plt.legend(['train', 'test'], loc='upper right')
                        plt.title('Model Error')
                        plt.ylabel('Mean Absolute Error')
                        plt.xlabel('Epoch')
                        plt.show()

                        # plotting mean squared error (mse)
                        results_df = pd.DataFrame(results, index=range(0,1)).T.rename(columns={0: 'MSE'}).sort_values('MSE', ascending=False)
                        results_df.T

                        plt.figure(figsize = (20, 6))
                        sns.barplot(x= results_df.index, y = results_df['MSE'], palette = 'summer')
                        plt.xlabel('Model')
                        plt.ylabel('MSE')
                        plt.title('MSE of different models');



# showing actual vs. predicted within your model

# create dataframe from newly predicted values
model.fit(x_train, y_train)  
prediction = regressor.predict(x_test)
        df2 = pd.DataFrame({'actual': y_test, 'predicted': prediction})
        df2.plot(kind='bar')
        plt.show()

# plot example with boxcox
        plt.figure(figsize= (10, 6))
        sns.scatterplot(x= inv_boxcox(y_test, lam), y= final_predictions, color= '#005b96')
        plt.xlabel('Actual rent')
        plt.ylabel('Predicted rent')
        plt.show()



# creating function to show history (shortcut)
def show_train_history(train_history,train,validation):
    plt.plot(train_history.history[train])
    plt.plot(train_history.history[validation])
    plt.title('Train History')
    plt.ylabel(train)
    plt.xlabel('Epoch')
    plt.legend(['train', 'validation'], loc='best')
    plt.show()

show_train_history(train_history,'acc','val_acc')
show_train_history(train_history,'loss','val_loss')
