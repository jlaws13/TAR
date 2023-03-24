import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib
import matplotlib.pyplot as plt

def TAR(y, threshold, lags, save_path=None):
    """
    Objective: Using OLS, estimate a threshold autoregressive model.
    :param y: Dataframe with pandas DatetimeIndex.
    :param threshold: (float) cutoff level separating the two regimes.
    :param lags: (int) number of lags to include in AR model.
    :param save_path: (str) of location to save images or None to turn off saving.
    :return:
            -results: list object containing OLS results for the regions below and above, respectively.
            -residuals: Series object containing timeseries of residuals.
            predict: Series object containing timeseries of predicted values.
    """
    y = y.dropna()
    # X = pd.DataFrame(np.zeros((len(y), lags)))
    X = pd.DataFrame()
    # print(X)
    # print('a')
    for i in range(1, lags+1):
        lag = y.shift(i)
        # print(lag)
        # X.iloc[:, i-1] = lag
        X = pd.concat([X, lag], axis=1)
        # print(X)
    # print(X)
    X = X.dropna()
    # print(X)
    X = sm.add_constant(X)
    # print(X)

    # Regression for values below the threshold
    less = X[X.iloc[:, 1] < threshold]
    # print('b')
    out1 = sm.OLS(y.loc[less.index], less).fit()
    # print('c')
    print('Regression results for values BELOW threshold:')
    print(out1.summary())

    # Regression for values above the threshold
    greater = X[X.iloc[:, 1] >= threshold]
    out2 = sm.OLS(y.loc[greater.index], greater).fit()
    print('Regression results for values ABOVE threshold:')
    print(out2.summary())

    # Residuals
    resid1 = out1.resid
    resid2 = out2.resid
    residuals = pd.concat([resid1, resid2], axis=0)

    # Results
    results = [out1, out2]

    # ACF plot of residuals
    fig1, ax1 = plt.subplots(figsize=(36, 10))
    sm.graphics.tsa.plot_acf(residuals, ax=ax1)
    ax1.set_title('ACF of Residuals')
    ax1.set_xlabel('Lags')
    ax1.set_ylabel('Autocorrelation')
    fig1.show()

    if save_path is not None:
        fig1.savefig(save_path + 'residual_acf.png')

    # Predicted values
    predict1 = out1.predict(less)
    predict2 = out2.predict(greater)
    predict = pd.concat([predict1, predict2], axis=0).sort_index()

    # create predicted values plot
    fig2, ax2 = plt.subplots(figsize=(36, 10))
    ax2.plot(y, label='Actual', linewidth=2)
    ax2.plot(predict, label='Predicted', linestyle='--', linewidth=2)
    ax2.set_title('Actual vs Predicted Values')
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Value')
    ax2.legend()
    fig2.show()

    if save_path is not None:
        fig2.savefig(save_path + 'predicted_values.png')

    return results, residuals, predict
    
    
