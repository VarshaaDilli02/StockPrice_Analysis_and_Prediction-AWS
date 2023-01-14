#PREDICTION USING ML MODEL

#Importing the Libraries
import pandas as PD
import NumPy as np
%matplotlib inline
import matplotlib. pyplot as plt
import matplotlib
from sklearn. Preprocessing import MinMaxScaler
from Keras. layers import LSTM, Dense, Dropout
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib. dates as mandates
from sklearn. Preprocessing import MinMaxScaler
from sklearn import linear_model
from Keras. Models import Sequential
from Keras. Layers import Dense
import Keras. Backend as K
from Keras. Callbacks import EarlyStopping
from Keras. Optimisers import Adam
from Keras. Models import load_model
from Keras. Layers import LSTM
from Keras. utils.vis_utils import plot_model

#Loading the Stock Market dataset
df=pd.read_csv("Amazon.csv",na_values=['null'],index_col='Date',parse_dates=True,infer_datetime_format=True)
print(df.head())

#Data Pre-Processing
#Print the shape of Dataframe  and Check for Null Values
print(“Dataframe Shape: “, df. shape)
print(“Null Value Present: “, df.IsNull().values.any())

#Data Visualisations
#Plot the True Adj Close Value
df[‘Adj Close’].plot()
plt.show()

#Plot the True High Value
df[‘High’].plot()
plt.show()

#Plot the True Low Value
df[‘Low’].plot()
plt.show()

#Plot the True Volume Value
df[‘Volume’].plot()
plt.show()

#Plot the High VS Low Values
high=df['High']
low=df['Low']
plt.legend()
plt.show()

#Plot the Open VS Close Values
opening=df['Open']
closing=df['Close']
plt.legend()
plt.show()

#Setting Target Variable and Selecting the Features
output_var = PD.DataFrame(df[‘Adj Close’])
#Selecting the Features
features = [‘Open’, ‘High’, ‘Low’, ‘Volume’]

#Scaling
scaler = MinMaxScaler()
feature_transform = scaler.fit_transform(df[features])
feature_transform= pd.DataFrame(columns=features, data=feature_transform, index=df.index)
print(feature_transform.head())

#Creating a Training set and test set for Stock Market Prediction
#Splitting to Training set and Test set
timesplit= TimeSeriesSplit(n_splits=10)
for train_index, test_index in timesplit.split(feature_transform):
        X_train, X_test = feature_transform[:len(train_index)], feature_transform[len(train_index): (len(train_index)+len(test_index))]
        y_train, y_test = output_var[:len(train_index)].values.ravel(), output_var[len(train_index): (len(train_index)+len(test_index))].values.ravel()
        
#Data Processing for LSTM
#Process the data for LSTM
trainX =np.array(X_train)
testX =np.array(X_test)
X_train = trainX.reshape(X_train.shape[0], 1, X_train.shape[1])
X_test = testX.reshape(X_test.shape[0], 1, X_test.shape[1])

#Building the LSTM Model for Stock Market Prediction
lstm = Sequential()
lstm.add(LSTM(32, input_shape=(1, trainX.shape[1]), activation=’relu’, return_sequences=False))
lstm.add(Dense(1))
lstm.compile(loss=’mean_squared_error’, optimizer=’adam’)
plot_model(lstm, show_shapes=True, show_layer_names=True)

#Training the Stock Market Prediction Model
#Model Training
history=lstm.fit(X_train, y_train, epochs=100, batch_size=8, verbose=1, shuffle=False)

#LSTM Prediction
y_pred= lstm.predict(X_test)
print(y_pred)

#Comparing Predicted vs True Adj Close Value – LSTM
plt.plot(y_test, label=’True Value’)
plt.plot(y_pred, label=’LSTM Value’)
plt.title(“Prediction by LSTM”)
plt.xlabel(‘Time Scale’)
plt.ylabel(‘Scaled USD’)
plt.legend()
plt.show()










































