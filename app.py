from flask import Flask, render_template, request, flash, redirect, jsonify
import config
from binance.client import Client
from binance.enums import *
from flask_cors import CORS, cross_origin
import get_data, stock_pred_lstm_rnn, stock_pred_xgboost
app = Flask(__name__)
cors = CORS(app)
app.secret_key = b'HoangKhoa'

client = Client(config.API_KEY, config.API_SECRET, tld='us')

@app.route('/')
@cross_origin()
def index():
    return render_template('index.html')

@app.route('/history')
@cross_origin()
def history():
    candlesticks = client.get_historical_klines("BTCUSDT", Client.KLINE_INTERVAL_3MINUTE, limit = 1000)

    processed_candlesticks = []

    for data in candlesticks:
        candlestick = { 
            # "open": data[1],
            # "high": data[2], 
            # "low": data[3], 
            # "close": data[4],
            "time": data[0]/1000,
            "value": data[4]
        }
        processed_candlesticks.append(candlestick)

    get_data.write_data(processed_candlesticks)
    return jsonify(processed_candlesticks)

@app.route('/predict')
@cross_origin()
def predict():
    type = request.args.get("type")
    if type == config.PREDICTION_TYPES.LSTM or type == config.PREDICTION_TYPES.RNN:
        print("Run predict with", type,"type")
        return_value = stock_pred_lstm_rnn.predict(type)
    elif type == config.PREDICTION_TYPES.XGBoost:
        print("Run predict with XGBoost type")
        return_value = stock_pred_xgboost.predict()
    return jsonify(return_value.to_json())
    
@app.route('/update-predict')
@cross_origin()
def update():
    type = request.args.get("type")
    time = request.args.get("time")
    value = request.args.get("value")

    get_data.write_more_data(time,value)
    if type == config.PREDICTION_TYPES.LSTM:
        print("RUN update predict with LSTM")
        return_value = stock_pred_lstm_rnn.predict(type)
    elif type == config.PREDICTION_TYPES.XGBoost:
        print("RUN update predict with XGBoost")
        return_value = stock_pred_xgboost.predict()
    new_value = return_value.tail(config.TIME_PREDICTION_NEXT)
    return jsonify(new_value.to_json())
    
    
