import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline
import csv, config

from regex import D
from datetime import datetime, timedelta
from matplotlib.pylab import rcParams
rcParams['figure.figsize']=20,10

from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler(feature_range=(0,1))


def predict(isWroteTempData):

    df=pd.read_csv("./data/processed_3minutes.csv")

    if isWroteTempData[0] == False:
        f = open("./data/processed_3minutes.csv", 'a', newline='')
        writer = csv.writer(f)
        data_length = len(df)
        for i in range(0, config.TIME_PREDICTION_NEXT):
            writer.writerow([
                df["Date"][data_length + i - 1] + 60 * config.INTERVAL_TIME,
                df["Close"][data_length + i - 1]
        ])

        f.close()
        isWroteTempData[0] = True
        df=pd.read_csv("./data/processed_3minutes.csv")

    df.head()
    df["Date"]=[datetime.fromtimestamp(x) for x in df["Date"]]
    # print(df["Date"])
    df.index=df['Date']

    from keras.models import Sequential, load_model, model_from_json
    from keras.layers import LSTM,Dropout,Dense

    data=df.sort_index(ascending=True,axis=0)
    new_dataset=pd.DataFrame(index=range(0,len(df)),columns=['Date','Close'])

    for i in range(0,len(data)):
        new_dataset["Date"][i]=data['Date'][i]
        new_dataset["Close"][i]=data["Close"][i]


    new_dataset.index=new_dataset.Date
    new_dataset.drop("Date",axis=1,inplace=True)

    final_dataset=new_dataset.values

    train_data=final_dataset[0:config.AMOUNT_OF_DATA,:]
    valid_data=final_dataset[config.AMOUNT_OF_DATA:,:]

    scaler=MinMaxScaler(feature_range=(0,1))
    scaled_data=scaler.fit_transform(final_dataset)

    x_train_data,y_train_data=[],[]

    for i in range(60,len(train_data)):
        x_train_data.append(scaled_data[i-60:i,0])
        y_train_data.append(scaled_data[i,0])
        
    x_train_data,y_train_data=np.array(x_train_data),np.array(y_train_data)

    x_train_data=np.reshape(x_train_data,(x_train_data.shape[0],x_train_data.shape[1],1))

    lstm_model=Sequential()
    lstm_model.add(LSTM(units=50,return_sequences=True,input_shape=(x_train_data.shape[1],1)))
    lstm_model.add(LSTM(units=50))
    lstm_model.add(Dense(1))

    lstm_model.compile(loss='mean_squared_error',optimizer='adam')
    lstm_model.fit(x_train_data,y_train_data,epochs=1,batch_size=1,verbose=2)

    inputs_data=new_dataset[len(new_dataset)-len(valid_data)-60:].values
    inputs_data=inputs_data.reshape(-1,1)
    inputs_data=scaler.transform(inputs_data)


    X_test=[]
    for i in range(60,inputs_data.shape[0]):
        X_test.append(inputs_data[i-60:i,0])
    X_test=np.array(X_test)

    X_test=np.reshape(X_test,(X_test.shape[0],X_test.shape[1],1))
    predicted_closing_price=lstm_model.predict(X_test)
    predicted_closing_price=scaler.inverse_transform(predicted_closing_price)

    lstm_model.save("saved_lstm_model_new.h5")

    train_data=new_dataset[:config.AMOUNT_OF_DATA]
    valid_data=new_dataset[config.AMOUNT_OF_DATA:]
    valid_data['Predictions']=predicted_closing_price
    new_valid_data=pd.DataFrame(index=range(0,len(valid_data)),columns=['Date','Predictions'])
    for i in range(0,len(valid_data)):
        new_valid_data["Date"][i]=(data["Date"][i + config.AMOUNT_OF_DATA] - timedelta(hours=7)).timestamp()
        new_valid_data["Predictions"][i]=valid_data["Predictions"][i]
    
    # print(new_valid_data)
    # test_value = new_valid_data.tail(config.TIME_PREDICTION_NEXT)
    # print(test_value)
    return new_valid_data
