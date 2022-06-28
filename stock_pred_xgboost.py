import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline
import csv, config

from regex import D
from datetime import datetime, timedelta
from xgboost import XGBRegressor
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

    inputs_data=new_dataset[len(new_dataset)-len(valid_data)-60:].values
    inputs_data=inputs_data.reshape(-1,1)
    inputs_data=scaler.transform(inputs_data)

    X_test=[]
    for i in range(60,inputs_data.shape[0]):
        X_test.append(inputs_data[i-60:i,0])
    X_test=np.array(X_test)

    model = XGBRegressor(n_estimators=1000, learning_rate=0.01)
    model.fit(x_train_data,y_train_data)
    predicted_closing_price=model.predict(X_test)
    predicted_closing_price=scaler.inverse_transform(predicted_closing_price.reshape(1,-1))
    print(predicted_closing_price[0])
    predicted_closing_price = predicted_closing_price[0]
    new_valid_data=pd.DataFrame(index=range(0,len(predicted_closing_price)),columns=['Date','Predictions'])
    for i in range(0,len(valid_data)):
        new_valid_data["Date"][i]=(data["Date"][i + config.AMOUNT_OF_DATA] - timedelta(hours=7)).timestamp()
        new_valid_data["Predictions"][i]=predicted_closing_price[i]

    return new_valid_data