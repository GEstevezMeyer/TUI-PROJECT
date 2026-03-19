import pandas as pd 
import tensorflow as tf 
import yfinance as yf
from data_windowing import WindowGenerator
import numpy as np
import matplotlib
matplotlib.use('Qt5Agg') 
import matplotlib.pyplot as plt 




def add_gaussian_noise(df: pd.DataFrame,sigma = 0.02):
    for columns in df.columns: 
        noise = np.random.normal(0, df[columns].std()*sigma, size=len(df[columns]))
        df[columns] += noise
    return df

def extract_ticket_data(ticket:str,period = "max") -> pd.DataFrame:
    df = yf.Ticker(ticket).history(period = period)
    if df.empty: 
        return False
    df = df.ffill()
    df = df.drop(columns=['Dividends', 'Stock Splits'])
    df = add_gaussian_noise(df)
    df = (df-df.mean())/df.std()
  
    return df

def multivariate_input_width(df: pd.DataFrame, threshold=0.2, max_lag=20):
    n_vars = df.shape[1]
    input_width = 1

    for lag in range(1, max_lag+1):
        for col in df.columns:
            x = df[col].values
            x_mean = np.mean(x)
            cov = np.sum((x[lag:] - x_mean) * (x[:-lag] - x_mean)) / (len(x)-1)
            var = np.var(x, ddof=1)
            acf = cov / var
            if abs(acf) >= threshold:
                input_width = max(input_width, lag)
    shift = max(1, input_width // 2)
    return input_width, shift


def create_window_class(df:pd.DataFrame) -> WindowGenerator:
    input_width, shift = multivariate_input_width(df)
    wg = WindowGenerator(df,input_width,shift)
    return wg

def create_sequential_model(window: WindowGenerator,input_shape = None) -> tf.keras.Sequential:

    if input_shape is None:
        input_shape = window.training_input.shape[1:] 

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=input_shape),
        tf.keras.layers.LSTM(64, activation='tanh',return_sequences=True),   
        tf.keras.layers.LSTM(32, activation='tanh'),  
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dropout((0.5)),

        tf.keras.layers.Dense(window.label_width)             
    ])
    
    model.compile(optimizer='adam',loss='mse', metrics=['mae'])

    

    return model

def training_sequential_model(ticket:str) -> tf.keras.Sequential: 
    df = extract_ticket_data(ticket)
    df = create_window_class(df)
    model = create_sequential_model(df)
    history = model.fit(df.training_tf,validation_data =df.val_tf,epochs= 50)
    history_plot(history)

    return model



def history_plot(history) ->None: 
    plt.plot(history.history['mae'], label='train MAE')
    plt.plot(history.history['val_mae'], label='val MAE')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.legend()
    plt.show()


    plt.plot(history.history['loss'], label='train loss')
    plt.plot(history.history['val_loss'], label='val loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()




if __name__ == "__main__": 
    model = training_sequential_model_boosting("AAPL")
    model = training_sequential_model("AAPL")
