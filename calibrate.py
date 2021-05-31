import pandas as pd
import numpy as np
import numba as nb
import glob
import json

import micro.fitting.preprocess
import micro.fitting.ml


def prepare_dataframe(df):
    df["time"] = pd.to_datetime(df["timestamp"], format="%Y-%m-%dD%H:%M:%S.%f")
    df.index = pd.DatetimeIndex(df["time"])
    df = df.rename(columns={'bidSize': 'bs', 'askSize':
                            'as', 'bidPrice': 'bid', 'askPrice': 'ask'})
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

    if end_date < start_date:
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

@nb.jit
def compute_microprice_forecast(microprice,
                                mid_price, time_index, millis_horizon):
    assert(len(mid_price) == len(time_index))
    reg_data = []
    i = 0
    while i < len(time_index):
        t0 = time_index[i]
        j = i + 1
        found = False
        while j < len(time_index):
            t1 = time_index[j]
            nanos = t1 - t0
            millis = nanos / 1000000
            if millis >= millis_horizon:
                mp = microprice[i]
                mid0 = mid_price[i]
                mid1 = mid_price[j]
                entry = np.zeros(7)
                entry[0] = t0
                entry[1] = t1
                entry[2] = mid0
                entry[3] = mid1
                entry[4] = mid1 - mid0
                entry[5] = mp
                entry[6] = mp - mid0
                reg_data.append(entry)
                i = j
                found = True
                break
            else:
                j += 1
         
        if not found:
            i += 1
            
    return reg_data

#%%
def main():

    with open("calib.config.json") as f:
        config = json.load(f)

    symbol = config["symbol"]
    writepath = config["write-path"]
    store_name = config["store-name"]
    start = config["start-date"]
    end = config["end-date"]

    df = load_dataframes(
        config["symbol"],
        config["store-path"],
        config["store-name"],
        start, end)
    
#%%
    tick_size = config["tick-size"]
    n_spread = config["n-spread"]
    price_decimals = config["price-decimals"]
    
    # Fitting
    training_data = micro.fitting.preprocess.create_training_data(df, tick_size, n_spread, price_decimals)
    model = micro.fitting.ml.estimate(training_data, n_spread, tick_size, price_decimals)
    Gstar, Bstar = model.calc_price_adj(order='stationary')
    
    # Save model
    Gstar.to_csv(f"{writepath}/{symbol}-{store_name}-model.csv")
    
    # Validate model
    training_data = training_data.merge(
        Gstar, how='left', on=['spread', 'imb_bucket'])
    training_data.index = pd.DatetimeIndex(training_data["time"])
    training_data['microprice'] = training_data['mid'] + training_data['Mid-Adjustment']

    training_data[['microprice', 'bid','ask']].tail(4500).plot()
    
    mid_price = training_data["mid"].values
    time_index = training_data.index.values.astype(np.int64)
    
    #%%
    import scipy.stats
    
    validation_data = compute_microprice_forecast(
         training_data.microprice.values,
        training_data.mid.values, time_index, 100)
    
    validation_data = pd.DataFrame(
        validation_data,
        columns=["t0", "t1", "mid0", "mid1", "dmid", "mp", "adj"])
    
    
    n = len(validation_data)
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(validation_data.adj, validation_data.dmid, alpha=0.5)
    plt.show()
    
if __name__ == '__main__':
    main()


