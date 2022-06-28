from asyncore import write
from pandas import Interval
import config, csv
from binance.client import Client

client = Client(config.API_KEY, config.API_SECRET)
def write_data(processed_candlesticks, isWroteTempData):
    isWroteTempData[0] = False
    csvFile = open('./data/processed_3minutes.csv', 'w', newline='')
    candleStickWriter = csv.writer(csvFile,delimiter=',')
    candleStickWriter.writerow(["Date","Close"])

    for i in range(len(processed_candlesticks)):
        candleStickWriter.writerow([processed_candlesticks[i]["time"],processed_candlesticks[i]["value"]])

    csvFile.close()
