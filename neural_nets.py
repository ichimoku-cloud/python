import pandas as pd
# Imported the layers of the neural networks
from tensorflow.keras.layers import Dense, Dropout, Activation

# Imported Model & Sequential from TF
from tensorflow.keras.models import Model, Sequential 
from tensorflow.keras.optimizers import Adam

# Load Dataset
crime_data = pd.read_csv('Datasets/crimeSTATS.csv')
crime_data.info()


# Set Feature & Targets
X = crime_data.drop(['total_crime_reported_per_1_million_res'], axis=1).values 
Y = crime_data[['total_crime_reported_per_1_million_res']].values

from sklearn.model_selection import train_test_split 
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=25)


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

    adam = Adam(lr=learning_rate)

    model.compile(loss='mean_squared_error', optimizer=adam, metrics=['mae'])
    return model 
    
# Model Parameters 
epochs = 100
learning_rate = 0.01
dropout_rate = 0.2


# Execute and Run the Model
model = create_model(learning_rate, dropout_rate)
model_history = model.fit(X_train, Y_train, batch_size=1, epochs=epochs, validation_split=0.2, verbose=10)


# Graph the Mean Absolute Error (MAE) for training and test sets
import matplotlib.pyplot as plt 

plt.plot(model_history.history['mae'])
plt.plot(model_history.history['val_mae'])
plt.legend(['train', 'test'], loc='upper right')
plt.title('Model Error')
plt.ylabel('Mean Absolute Error')
plt.xlabel('Epoch')
plt.show()

