import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Bidirectional, LSTM, Dropout, Dense
import pickle
import joblib

if __name__ == '__main__':

    df = pd.read_csv('C:\\Naresh Workspace\\SRH Heidelberg\\Big Data and Business Analytics\\Analytics 4\\data\\nyc_taxi.csv')

    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # Setperating the timestamp columns into seperate columns
    df['Year'] = df.timestamp.dt.year
    df['Month'] = df.timestamp.dt.month
    df['Day'] = df.timestamp.dt.day
    df['Hour'] = df.timestamp.dt.hour

    # Number of readings per day
    df[(df['Day'] == 1) & (df['Month'] == 7) & (df['Year'] == 2014)]

    # (1) Adding day of the week.
    df['Weekday'] = df.timestamp.dt.weekday

    # (2) Adding the average of rides grouped by the weekday and hour
    # 7 Days, 24 Hours = 168 values
    len(df[:7344].groupby(df.Weekday.astype(str) + ' ' + df.Hour.astype(str))['value'].mean().to_dict())

    df['avg_hour_day'] = df.Weekday.astype(str) + ' ' + df.Hour.astype(str)

    df.avg_hour_day = df.avg_hour_day.replace(df[:7344].groupby(df.Weekday.astype(str) + ' ' + df.Hour.astype(str))['value'].mean().to_dict())

    # (3) Featuring the number of rides during the day and during the night.
    # We define the day time to be any hours between 6 AM and 10 PM while Night time where usually there is less 
    # demand is any time between 10:00 PM and 6:00 AM
    df['day_time'] = ((df['Hour'] >= 6) & (df['Hour'] <= 22)).astype(int)

    # Normalizing the values
    standard_scaler = preprocessing.StandardScaler()
    scaled_data = standard_scaler.fit_transform(df[['Hour', 'day_time', 'Weekday', 'avg_hour_day', 'value']])

    scaled_df = df.copy()

    scaled_df['Hour'] = scaled_data[:,0]
    scaled_df['day_time'] = scaled_data[:,1]
    scaled_df['Weekday'] = scaled_data[:,2]
    scaled_df['avg_hour_day'] = scaled_data[:,3]
    scaled_df['value'] = scaled_data[:,4]

    # Specifying how many values to predict
    time_step = 1

    # ### Splitting the dataset

    training_size = int(len(scaled_df) * 0.9)
    training, testing = scaled_df[0:training_size], scaled_df[training_size:len(df)]

    # training features: Value, Hour, day_time
    X_train = training[['value', 'Hour', 'day_time']].to_numpy()
    y_train = scaled_df[time_step:testing.index[0]]['value'].to_numpy()

    # testing data
    X_test = testing[0:-time_step][['value', 'Hour', 'day_time']].to_numpy()
    y_test = scaled_df[testing.index[0] + time_step:]['value'].to_numpy()

    # create sequences of (48-two readings per hour) data points for each training example
    def create_sequence(dataset, length):
        data_sequences = []
        for index in range(len(dataset) - length):
            data_sequences.append(dataset[index: index + length])
        return np.asarray(data_sequences)

    X_train = create_sequence(X_train, 48)
    X_test  = create_sequence(X_test, 48)
    y_train = y_train[-X_train.shape[0]:]
    y_test  = y_test[-X_test.shape[0]:]

    # ### Model Building
    # Building the model
    model1 = Sequential()
    # Adding a Bidirectional LSTM layer
    model1.add(Bidirectional(LSTM(64,return_sequences=True, dropout=0.5, input_shape=(X_train.shape[1], X_train.shape[-1]))))
    model1.add(Bidirectional(LSTM(20, dropout=0.5)))
    model1.add(Dense(1))
    model1.compile(loss='mse', optimizer='rmsprop', metrics = [tf.keras.metrics.MeanAbsoluteError()])

    # Training the model
    model1.fit(X_train, y_train, batch_size=128, epochs=50)

    # ### Save the model
    model1.save('my_model.h5')