
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from function.data_windowing import WindowGenerator
from function.model_handler import get_numbers_of_models

import tensorflow as tf 
import tomllib
import plotext as plt
from textual_plotext import PlotextPlot
from textual.widgets import DataTable
import yfinance as yf 
import numpy as np






with open("function/parameters.toml","rb") as f:
    toml_data:dict = tomllib.load(f)
    FIGSIZE = tuple(toml_data["FIGSIZE"])

def extract_action_data(ticker:str) -> np.array:
    df = yf.Ticker(ticker).actions

    return df.to_numpy()

def extract_tickers_action_data() -> dict: 
    tickers = get_numbers_of_models()
    data = {}
    for ticker in tickers: 
        data[ticker] = extract_action_data(ticker)

    return data 

def action_plot_text_widget(widget: PlotextPlot):

    data = extract_tickers_action_data()

    plt = widget.plt 
    plt.clear_data() 

    for ticker in data.keys(): 
        plt.plot(data[ticker][:,0],data[ticker][:,1], label=ticker)
    
    plt.title("Action of the Day")
    plt.xlabel("Dividends")
    plt.ylabel("Stock Splits")
    

    widget.refresh()  



def history_plot_text_widget(widget: PlotextPlot, history):
    plt = widget.plt 

    train_mae = history.history['mae']
    val_mae = history.history['val_mae']
    epochs = list(range(1, len(train_mae) + 1))

    plt.clear_data()      
    plt.plot(epochs, train_mae, label="train MAE")
    plt.plot(epochs, val_mae, label="val MAE")
    plt.title("Training History")
    plt.xlabel("Epoch")
    plt.ylabel("MAE")

    widget.refresh()   

def plot_prediction_test_general_widget(widget: PlotextPlot, wg, model, mean: float, std: float, label_encoder: int = None):
    plt = widget.plt  

    
    if label_encoder is not None:
        x, y = wg.split(wg.raw_df.to_numpy(), wg.input_width, wg.shift, wg.label_width, 2)
    else:
        x, y = wg.split(wg.raw_df.to_numpy(), wg.input_width, wg.shift, wg.label_width)

    y_pred = model.predict(x).flatten()
    y_pred = (y_pred + mean) * std

    y_real = y.flatten()
    y_real = (y_real + mean) * std

    indices = list(range(len(y_real)))

    plt.clear_data()
    plt.plot(indices, y_pred, label="Predicted")
    plt.plot(indices, y_real, label="Real")
    plt.title("Prediction vs Real")
    plt.xlabel("Index")
    plt.ylabel("Value")

    widget.refresh() 

def make_dataTable(widget:DataTable,result):
    history = result.history
    for columns in history.keys(): 
        widget.add_column(columns)

    rows = list(zip(*history.values()))

    widget.add_rows(rows)