import config, csv
from binance.client import Client

client = Client(config.API_KEY, config.API_SECRET)
def write_data(processed_candlesticks):
    csvFile = open('./data/processed_3minutes.csv', 'w', newline='')
    candleStickWriter = csv.writer(csvFile,delimiter=',')
    candleStickWriter.writerow(["Date","Close"])

    for i in range(len(processed_candlesticks)):
        candleStickWriter.writerow([processed_candlesticks[i]["time"],processed_candlesticks[i]["value"]])

    csvFile.close()

def write_more_data(time,value):
    f = open("./data/processed_3minutes.csv", 'a',newline='')
    writer = csv.writer(f)
    print("Append data: ", time, value)
    writer.writerow([time, value])
    f.close()