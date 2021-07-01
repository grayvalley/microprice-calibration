import pandas as pd
import numpy as np
import numba as nb
import json

import matplotlib.pyplot as plt
import statsmodels.api as sm
    
import micro.fitting.preprocess
import micro.fitting.ml

from alpha.solver import (
    ShortTermAlpha_Finite_Difference_Solver,
    ShortTermAlpha
)


def prepare_dataframe(df):
    
    df["time"] = pd.to_datetime(
        df["timestamp"], format="%Y-%m-%dD%H:%M:%S.%f")
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

    df.index = df["time"]

    return df

@nb.jit
def compute_microprice_forecast(
        microprice, spread, mid_price, imb_bucket, time_index, millis_horizon):
    """
    Backtest Micro-Price forecast against realized mid-price changes.
    """
    
    assert(len(mid_price) == len(time_index))
    
    result = []
    i = 0
    while i < len(time_index):
        t0 = time_index[i]
        j = i + 1
        found = False
        while j < len(time_index):
            t1 = time_index[j]
            nanos = t1 - t0
            millis = nanos / 1000000 # time index is in nanoseconds
            if millis >= millis_horizon:
                imb = imb_bucket[i]
                mp = microprice[i]
                mid0 = mid_price[i]
                mid1 = mid_price[j]
                s = spread[i]
                entry = np.zeros(9)
                entry[0] = t0  # reference time
                entry[1] = t1  # reference time + horizon milliseconds
                entry[2] = mid0  # mid at time t0
                entry[3] = mid1  # mid at time t1
                entry[4] = mid1 - mid0  # real change in mid-price
                entry[5] = mp  # micro-price as observed at time t0
                entry[6] = mp - mid0  # forecasted change in mid-price
                entry[7] = imb  # imbalance that motivates microprice
                entry[8] = s # spread at time t0
                result.append(entry)
                i = j
                found = True
                break
            else:
                j += 1
         
        if not found:
            i += 1
            
    return result


@nb.jit
def create_uniform_imbalance_series(imbalance, time_index, millis_horizon):
    """
    Creates time-uniform order book imbalance series. This data is used to
    fit Ornstein - Uhlenbeck process to the imbalance time series. These
    parameters are used to fit the trading model.
    """
    
    assert(len(imbalance) == len(time_index))
    
    result = []
    i = 0
    while i < len(time_index):
        t0 = time_index[i]
        i0 = imbalance[i]
        j = i + 1
        found = False
        while j < len(time_index):
            t1 = time_index[j]
            nanos = t1 - t0
            millis = nanos / 1000000  # time index is in nanoseconds
            if millis >= millis_horizon:
                i1 = imbalance[j]
                entry = np.zeros(7)
                entry[0] = t0  # reference time t0
                entry[1] = t1  # reference time t0 + horizon milliseconds
                entry[2] = i0  # imbalance at time t0
                entry[3] = i1  # imbalance at time t1
                entry[4] = 2*i0 - 1  # scaled imbalance as of time t0
                entry[5] = 2*i1 - 1  # scaled imbalance as of time t1
                entry[6] = i1 - i0   # change in imbalance from t0 to t1
                result.append(entry)
                i = j
                found = True
                break
            else:
                j += 1
         
        if not found:
            i += 1
            
    return result


def save_excel(decisions, inventory_levels, signal_levels):
 
    opt_dec = pd.DataFrame(
        data=decisions, 
        index=inventory_levels,
        columns=signal_levels)
    
    opt_dec.to_excel("solution.xlsx")
    

def calibrate_microprice(config):
    """
    Calibrates Stoikov's Micro-Price model to BitMEX market data.
    """
    
    # Load raw L1 data
    raw_l1_data = load_dataframes(
        config["symbol"],
        config["store-path"],
        config["store-name"],
        config["start-date"], 
        config["end-date"])
    
    # Create training data for micro-price
    calib_data = micro.fitting.preprocess.create_training_data(
        raw_l1_data,
        config["tick-size"],
        config["n-spread"],
        config["mid-decimals"],
        config["buckets"])
    
    # Fit micro-price model
    model = micro.fitting.ml.estimate(
        calib_data, 
        config["n-spread"],
        config["tick-size"], 
        config["mid-decimals"],
        config["buckets"])
    
    # Compute adjustments
    Gstar, Bstar = model.calc_price_adj()
    
    # Save micro-price adjustments
    save_dir = f'{config["write-path"]}'
    filename = f'{config["symbol"]}-{config["store-name"]}-model.csv'
    Gstar.to_csv(f'{save_dir}/{filename}')
    
    # Compute micro-price adjustments
    calib_data = calib_data.merge(
        Gstar, how='left',
        on=['spread', 'imb_bucket'])
    
    # Compute micro-price
    calib_data["mp"] = calib_data["mid"] + calib_data["mp_adj"]
    
    run_microprice_diagnostics(calib_data, config)
    
    return calib_data, Gstar, Bstar


def run_microprice_diagnostics(data, config):
    
    
    # "Backtest" Micro-Price forecast
    forecasts = compute_microprice_forecast(
        data["mp"].values,
        data["spread"].values,
        data["mid"].values,
        data["imb_bucket"].values, 
        data["time"].values.astype(np.int64),
        1000)
    
    forecasts = pd.DataFrame(forecasts)
    forecasts.columns = [
        't0', 't1', 'mid0', 'mid1', 'dmid', 
        'mp0', 'mp_adj', 'imb_bucket', 'spread'
        ]

    # Let's compute mean historical mid-price changes and
    # average microprice forecasts given imbalance bucket
    buckets = sorted(data["imb_bucket"].value_counts().index.values)
    mean_changes = []
    mean_forecasts = []
    for bucket in buckets:
        mean_changes.append(
            forecasts["dmid"].loc[
                (forecasts["imb_bucket"] == bucket) &
                (forecasts["spread"] == 1)].mean())
        mean_forecasts.append(
            forecasts["mp_adj"].loc[
                (forecasts["imb_bucket"] == bucket) &
                (forecasts["spread"] == 1)].mean())
    
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(buckets, mean_forecasts,
            color='blue', marker='o', lw=2, label='Model')
    ax.plot(buckets, mean_changes,
            color='red', marker='o', lw=2, label='Realized')
    plt.legend()
    plt.title('Mid-Price Change forecast (1000 milliseconds)')
    plt.suptitle(f'Symbol: {config["symbol"]}')
    ax.set_ylabel('Change in Mid-Price')
    ax.set_xticklabels(np.round(np.arange(0, 1.1, 0.1), 2))
    ax.set_xlabel('Order Book Imbalance')
    fig.savefig("graphs/calibrated.pdf", bbox_inches='tight')
    fig.savefig("graphs/calibrated.png", bbox_inches='tight')
    
    
def calibrate_nbbo_trading_model(mp_calib_data, trade_data, config):
    
    # Compute imbalance approximation to micro-price
    beta = estimate_imbalance_beta(mp_calib_data)
    print(beta)
    # Compute imbalance Ornstein-Uhlenbeck parameters
    zeta, eta = estimate_imbalance_ou_process(mp_calib_data, config)
    print(zeta)
    
    # Compute trade arrival rates
    lamda_m = 10
    lamda_p = 10
    
    model_params = config["model-params"]
    
    sta = ShortTermAlpha(zeta, 0.01, eta, beta)
    
    T = model_params["terminal-time"]
    N = model_params["finite-difference-steps"]
    
    # Create inventory grid
    q_min = model_params["min-inventory"]
    q_max = model_params["max-inventory"]
    q_grid = np.arange(q_min, q_max + 1, 1)
    
    # Create time grid
    dt = T / 5000
    t_grid = np.arange(0, T + dt, dt)
    
    # Penalty parameters
    inv_pen = 0#model_params["inventory-penalty"]
    ter_pen = 0#model_params["terminal-penalty"]
    
    # Half of typical bid-ask spread
    half_spread = 0.5 * config["tick-size"]
    
    h, lp, lm = ShortTermAlpha_Finite_Difference_Solver.solve_tob_postings(
        sta, 
        q_grid,
        t_grid, 
        half_spread,
        ter_pen,
        inv_pen,
        dt,
        lamda_p,
        lamda_m)
    
    # Create optimal posting matrix
    l_p = np.zeros((lp.shape[0], lp.shape[1]))
    l_m = np.zeros((lp.shape[0], lp.shape[1]))
    l_p[lp[:, :, 50] == True] = 1
    l_p[lp[:, :, 50] == False] = 0
    l_m[lm[:, :, 50] == True] = 2
    l_m[lm[:, :, 50] == False] = 0
    decisions = l_m + l_p

    save_excel(decisions, q_grid, sta.imbalance)


def estimate_imbalance_beta(mp_calib_data):
    """
    Estimates: MP-Adj ~ beta * (2 * imbalance - 1)
    
    Parameters:
    ----------
    mp_calib_data : pd.DataFrame
        Microprice calibration dataset.

    Returns:
    -------
    beta: slope coefficient

    """
  
    X = 2 * mp_calib_data['imb'].loc[mp_calib_data.spread==1] - 1
    y = mp_calib_data['mp_adj'].loc[mp_calib_data.spread==1]
    ols_est = sm.OLS(y, X).fit()
    
    beta = ols_est._results.params[0]
    
    return beta

def estimate_imbalance_ou_process(mp_calib_data, config):
    """
    Estimates: I_{t+1} = zeta*I_{t} + eta*dW_{t}

    Parameters
    ----------
    mp_calib_data : pd.DataFrame
        Microprice calibration dataset.
        
    config : dictionary
        configuration parameters.

    Returns
    -------
    zeta : float
        Mean-reversion speed of order book imbalance.
        
    eta : float
        Volatility of order book imbalance.

    """
    
    # imbalance values
    imb_vals = mp_calib_data["imb"].values
    
    # Create nanosecond time index
    time_idx = mp_calib_data["time"].values.astype(np.int64)
    
    # Make time deltas between observations uniform
    imb_series = create_uniform_imbalance_series(
        imb_vals, time_idx, 1000)
    
    imb_series = pd.DataFrame(
        data=imb_series)
    
    # Fit linear model: {I_{t+1} = zeta*I_{t} + e_{t}, e_{t} ~ N(0, eta)
    X = imb_series.values[:, 4]
    y = imb_series.values[:, 5]
    ols_est = sm.OLS(y, X).fit()
    
    # mean-reversion speed
    zeta = -np.log(ols_est._results.params[0])
    
    # residual volatility
    eta = imb_series[6].std()
    
    return zeta, eta
    

#%%
def main():
#%%

    # Load configuration
    with open("calib.config.json") as f:
        config = json.load(f)
        
    # Calibrate Micro-Price model
    calib_data, Gstar, Bstar = calibrate_microprice(config)
    
    # Calibrate NBBO market making model
    #calibrate_nbbo_trading_model(calib_data, None, config)
    
    
    # symbol     = config["symbol"]
    # writepath  = config["write-path"]
    # store_name = config["store-name"]
    # start      = config["start-date"]
    # end        = config["end-date"]
    # tick_size  = config["tick-size"]
    # n_spread   = config["n-spread"]
    # decimals   = config["mid-decimals"]
    # buckets    = config["buckets"]
    
    # # Load data
    # df = load_dataframes(
    #     config["symbol"],
    #     config["store-path"],
    #     config["store-name"],
    #     start, end)
    
    # # Create training data for micro-price
    # training_data = micro.fitting.preprocess.create_training_data(
    #     df, tick_size, n_spread, decimals, buckets)
    
    # # Fit micro-price model
    # model = micro.fitting.ml.estimate(
    #     training_data, n_spread, tick_size, decimals, buckets)
    
    # # Compute adjustments
    # Gstar, Bstar = model.calc_price_adj()
    
    # # Save model
    # Gstar.to_csv(f"{writepath}/{symbol}-{store_name}-model.csv")
    
    # # Save training data
    # training_data.to_csv(
    #     f"{writepath}/{symbol}-{store_name}-training-data.csv")
    
    # # Compute microprice process
    # training_data = training_data.merge(
    #     Gstar, how='left',
    #     on=['spread', 'imb_bucket'])
    
    # training_data.index = pd.DatetimeIndex(training_data["time"])
    # training_data['microprice'] = training_data['mid'] + training_data['Mid-Adjustment']
    
    # # Compare microprice forecast against real mid-price changes
    # time_index = training_data.index.values.astype(np.int64)
    # validation_data = compute_microprice_forecast(
    #       training_data.microprice.values,
    #       training_data.mid.values,
    #       training_data.imb_bucket.values,
    #       time_index,
    #       1000)
    # validation_data = pd.DataFrame(
    #     validation_data,
    #     columns=["t0", "t1", "mid0", "mid1", 
    #              "dmid", "mp", "adj", "imb"])
    
    # pred_changes = []
    # real_changes = []
    # for bucket in range(0, len(buckets)-1):
    #     pred_changes.append(validation_data['adj'].loc[
    #             validation_data['imb']==bucket].mean())
    #     real_changes.append(validation_data['dmid'].loc[
    #             validation_data['imb']==bucket].mean())
    # pred_changes = pd.DataFrame(
    #     data=pred_changes, index=np.arange(len(buckets)-1))
    # real_changes = pd.DataFrame(
    #     data=real_changes, index=np.arange(len(buckets)-1))
    
    # fig, ax = plt.subplots(figsize=(6, 4))
    # ax.plot(real_changes.index, real_changes.values, color='red')
    # ax.plot(real_changes.index, pred_changes.values, color='blue')
    # plt.show()
    
    # # fig, ax = plt.subplots(figsize=(6, 4))
    # # ax.scatter(
    # #         training_data['imb'].loc[training_data.spread==1],
    # #         training_data['Mid-Adjustment'].loc[training_data.spread==1],
    # #         color='red')
    # # plt.show()
    
    # params = {}
    
    # # Estimate linear model to approximate micro-price using imbalance only
    # X = 2*training_data['imb'].loc[training_data.spread==1] - 1
    # y = training_data['Mid-Adjustment'].loc[training_data.spread==1]
    # lm = sm.OLS(y, X).fit()
    # lm.summary()
    
    # params.update({'beta': lm._results.params[0]})
    
    # # fig, ax = plt.subplots(figsize=(6, 4))
    # # ax.scatter(X, lm.predict(X), color='blue', alpha=0.1)
    # # ax.scatter(X, y, color='red', alpha=0.1)
    # # plt.show()
    
    # # Estimate alpha-signal mean-reversion speed
    # alpha_changes = compute_imbalance_change(
    #     training_data.imb.values, time_index, 1000)
    # alpha_changes = pd.DataFrame(
    #     alpha_changes,
    #     columns=["t0", "t1", "i0", "i1", "dimb"])
    
    # X_alpha = alpha_changes["i0"]
    # y_alpha = alpha_changes["dimb"]
    # lm_alpha = sm.OLS(y_alpha, X_alpha).fit()
    # lm_alpha.summary()
    
    # params.update({'zeta': lm_alpha._results.params[0]})
    # params.update({'eta': lm_alpha.resid.std()})
    
    
    #%%
    
if __name__ == '__main__':
    main()


