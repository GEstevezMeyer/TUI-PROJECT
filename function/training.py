import os
import pandas as pd 
import tensorflow as tf 
from tensorflow.keras.callbacks import EarlyStopping
import yfinance as yf
import numpy as np
import tomllib
from tensorflow.keras.callbacks import History
from textual.widgets import ProgressBar

from function.data_windowing import WindowGenerator


def import_parameters():
    with open("function/parameters.toml","rb") as f:
        toml_data= tomllib.load(f)
    
    return toml_data

def add_gaussian_noise(df: pd.DataFrame,sigma:float):
    for columns in df.columns: 
        noise = np.random.normal(0, df[columns].std()*sigma, size=len(df[columns]))
        df[columns] += noise
    return df

def extract_ticket_data(ticket:str,period:str,sigma:float = 0,addGaussianNoise = False) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    df = yf.Ticker(ticket).history(period = period)
    if df.empty: 
        return False
    df = df.ffill()
    df = df.drop(columns=['Dividends', 'Stock Splits'])
    if add_gaussian_noise:
        df = add_gaussian_noise(df,sigma)
    mean = df.mean()
    std = df.std()
    df = (df-mean)/std
    
  
    return df,mean,std

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


def create_window_class(df:pd.DataFrame,model_type:str = "LSTM") -> WindowGenerator:
    input_width, shift = multivariate_input_width(df)
    label_index = df.columns.get_loc("Close")
    if model_type == "LSTM":
        wg = WindowGenerator(df,input_width,shift,label_encoder=label_index)
    elif model_type == "Linear":
        wg = WindowGenerator(df=df,input_width=1,shift=1,label_width=1,label_encoder= label_index)
    return wg

def create_lstm_model(window: WindowGenerator,input_shape = None) -> tf.keras.Sequential:

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

def create_linear_model(window: WindowGenerator,input_shape = None):

    if input_shape is None:
        input_shape = window.training_input.shape[1:] 

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=input_shape),
        tf.keras.layers.Dense(1)
    ])

    model.compile(
        optimizer="adam",
        loss="mse",
        metrics=["mae"]
    )

    return model

def training_sequential_model(widget:ProgressBar, ticket:str,parameters,gaussianNoisse:bool,model_type = "Linear") -> tf.keras.Sequential: 

    widget.total = 100 

    widget.display = True

    early_stop = EarlyStopping(
        monitor='val_loss',   
        patience=12,           
        restore_best_weights=True,  
        verbose=1            
    )

    print(parameters)

    df,mean,std = extract_ticket_data(ticket,parameters["PERIOD"],sigma=parameters["SIGMA"],addGaussianNoise = gaussianNoisse)

    widget.advance(20)

    if model_type == "Linear":
        df = create_window_class(df)
        model = create_linear_model(df)
    elif model_type == "LSTM":
        df = create_window_class(df,model_type)
        model = create_lstm_model(df)

    widget.advance(20)

    history = model.fit(df.training_tf,validation_data =df.val_tf,epochs= parameters["EPOCHS"],callbacks=[early_stop])

    widget.advance(40)

    return model,history,mean,std,df

def saving_model(model:tf.keras.Sequential,ticket:str) -> None: 
    cwd = os.getcwd()
    workingSPace = os.listdir(cwd)
    if "tfKerasModels" not in workingSPace:
        os.makedirs("tfKerasModels", exist_ok=True)

    contents = os.listdir("tfKerasModels")
    if ticket not in contents:
        os.makedirs(f"tfKerasModels/{ticket}", exist_ok=True)
    model.save(f"tfKerasModels/{ticket}/{ticket}_model.keras")

def make_toml_file(model_type:str,mean:float,std:float,history:History,ticket:str,parameters:dict): 
    toml_content = f"""
    model = "{model_type}"
    epochs = {parameters["EPOCHS"]}
    sigma = {parameters["SIGMA"]}
    mae = {min(history.history["mae"])}
    val_mae =  {min(history.history["val_mae"])}
    loss = {min(history.history["loss"])}
    val_loss = {min(history.history["loss"])}
    """

    with open(f"tfKerasModels/{ticket}/config.toml", "w") as f:
        f.write(toml_content)

def main(widget:ProgressBar,gaussianNoise:bool,ticket:str,model_type:str="LSTM") ->dict:

    parameters = import_parameters()

    model,history,mean,std,wg = training_sequential_model(widget,ticket,parameters,
                                                          gaussianNoise,model_type)

    widget.advance(5)

    saving_model(model,ticket)

    widget.advance(5)

    make_toml_file(model_type,mean,std,history,ticket,parameters)

    widget.advance(10)

    return history,mean,std,model,wg



