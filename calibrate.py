import pandas as pd
import glob
import json

import micro.fitting.preprocess
import micro.fitting.ml


def get_file_names(path):
    names = []
    for file in glob.glob(f"{path}/*.h5"):
        names.append(file)
    return names


def prepare_dataframe(df):
    df["time"] = pd.to_datetime(df["timestamp"], format="%Y-%m-%dD%H:%M:%S.%f000")
    df = df.rename(columns={'bidSize': 'bs', 'askSize': 'as', 'bidPrice': 'bid', 'askPrice': 'ask'})
    return df


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


def load_dataframes(symbol, store_path, store_name, start, end):

    dates = create_date_strings(start, end)

    frames = []
    for date in dates:
        try:
            df = pd.read_csv(f"{store_path}/{store_name}-{date}")
            df = df[df["symbol"] == symbol]
            df = prepare_dataframe(df)
            frames.append(df)
            print("Loaded: ", date)
        except Exception as ex:
            print(ex)
            pass

    df = pd.concat(frames)

    return df


def main():

    with open("calib.config.json") as f:
        config = json.load(f)

    symbol = config["symbol"]
    readpath = config["store-path"]
    writepath = config["write-path"]
    start = config["start-date"]
    end = config["end-date"]

    df = load_dataframes(
        config["symbol"],
        config["store-path"],
        config["store-name"],
        start, end)

    n_imb = config["n-buckets"]
    n_spread = config["n-spread"]
    dt = config["dt"]
    df, misc = micro.fitting.preprocess.discretize(df, n_imb, dt, n_spread)
    df = micro.fitting.preprocess.mirror(df, misc)
    G1, B, Q, Q2, R1, R2, K = micro.fitting.ml.estimate(df)
    Gstar, Bstar = micro.fitting.ml.calc_price_adj(G1, B, order='stationary')

    # Save model
    Gstar.to_hdf(f"{writepath}/{symbol}_model_store.h5", key=symbol)
    Gstar.to_csv(f"{writepath}/{symbol}.csv")


if __name__ == '__main__':
    main()


