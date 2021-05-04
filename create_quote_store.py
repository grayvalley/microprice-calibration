import json
import pandas as pd
from bitmex_s3.src.downloader import (
    QuoteData
)

def create_date_strings(start_date, end_date):

    def month_str(month):
        if month < 10:
            return f"0{month}"
        else:
            return f"{month}"

    def day_str(day):
        if day < 10:
            return f"0{day}"
        else:
            return f"{day}"

    if end_date <= start_date:
        raise ValueError("Check dates")

    range = pd.date_range(start=start_date, end=end_date, freq='D')

    dates = []
    for date in range:
        dates.append(f"{date.year}{month_str(date.month)}{day_str(date.day)}")

    return dates


def main():

    with open('store.config.json', 'r') as json_file:
        config = json.load(json_file)

    bucket_name = config["bucket-name"]
    data_type = config["data-type"]
    store_path = config["store-path"]
    quote_data = QuoteData(bucket_name)

    #store = pd.HDFStore(f"store.{data_type}/{bucket_name}.h5")
    dates = create_date_strings(config["start-date"], config["end-date"])
    for date in dates:
        data = quote_data.get(date)
        data.to_csv(f"{store_path}/{bucket_name}-{date}")
        #store.put(f"{data_type}_{date}", data, format='table')
        print("Downloaded: ", date)


if __name__ == '__main__':
    main()