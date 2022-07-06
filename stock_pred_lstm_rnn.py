import pandas as pd
import numpy as np
import config

from datetime import timedelta
from matplotlib.pylab import rcParams
rcParams['figure.figsize']=20,10

from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler(feature_range=(0,1))


def predict(df, col, model):
    from keras.models import Sequential
    from keras.layers import LSTM,Dense,SimpleRNN

    data=df.sort_index(ascending=True,axis=0)
    new_dataset=pd.DataFrame(index=range(0,len(df)),columns=['Date',col])

    if col == "Rate":
        for i in range(0,len(data)):
            new_dataset["Date"][i]=data['Date'][i]
            new_dataset[col][i]=(data["Close"][i]-data["Close"][i-1])/data["Close"][i-1] * 100 if i>0 else 0
    else:
        for i in range(0,len(data)):
            new_dataset["Date"][i]=data['Date'][i]
            new_dataset[col][i]=data[col][i]

    new_dataset.index=new_dataset.Date
    new_dataset.drop("Date",axis=1,inplace=True)

    final_dataset=new_dataset.values
    train_data=final_dataset[0:(len(final_dataset) - config.TIME_PREDICTION_NEXT - 1),:]
    valid_data=final_dataset[(len(final_dataset) - config.TIME_PREDICTION_NEXT - 1):,:]

    scaler=MinMaxScaler(feature_range=(0,1))
    scaled_data=scaler.fit_transform(final_dataset)

    x_train_data,y_train_data=[],[]

    for i in range(60,len(train_data)):
        x_train_data.append(scaled_data[i-60:i,0])
        y_train_data.append(scaled_data[i,0])
        
    x_train_data,y_train_data=np.array(x_train_data),np.array(y_train_data)

    x_train_data=np.reshape(x_train_data,(x_train_data.shape[0],x_train_data.shape[1],1))

    pred_model=Sequential()

    if model == config.PREDICTION_TYPES.LSTM:
        pred_model.add(LSTM(units=50,return_sequences=True,input_shape=(x_train_data.shape[1],1)))
        pred_model.add(LSTM(units=50))
    elif model == config.PREDICTION_TYPES.RNN:
        pred_model.add(SimpleRNN(units=50,return_sequences=True,input_shape=(x_train_data.shape[1],1)))
        pred_model.add(SimpleRNN(units=50))
    pred_model.add(Dense(1))

    pred_model.compile(loss='mean_squared_error',optimizer='adam')
    pred_model.fit(x_train_data,y_train_data,epochs=1,batch_size=1,verbose=2)

    inputs_data=new_dataset[len(new_dataset)-len(valid_data)-60:].values
    inputs_data=inputs_data.reshape(-1,1)
    inputs_data=scaler.transform(inputs_data)

    X_test=[]
    for i in range(60,inputs_data.shape[0]):
        X_test.append(inputs_data[i-60:i,0])
    X_test=np.array(X_test)

    X_test=np.reshape(X_test,(X_test.shape[0],X_test.shape[1],1))
    predicted_closing_price=pred_model.predict(X_test)
    predicted_closing_price=scaler.inverse_transform(predicted_closing_price)
    valid_data=new_dataset[(len(new_dataset)-config.TIME_PREDICTION_NEXT-1):]
    valid_data['Predictions']=predicted_closing_price
    new_valid_data=pd.DataFrame(index=range(0,len(valid_data)),columns=['Date','Predictions'])
    for i in range(0,len(valid_data)):
        new_valid_data["Date"][i]=(data["Date"][len(data) - config.TIME_PREDICTION_NEXT - 1 + i] - timedelta(hours=7)).timestamp()
        new_valid_data["Predictions"][i]=valid_data["Predictions"][i]
    
    return new_valid_data
