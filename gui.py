import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2" 
import warnings
warnings.filterwarnings(
    "ignore",
    message=".*TensorFlow GPU support is not available on native Windows.*"
)

from textual.app import App
from textual.widgets import Header, Footer, Input,Tabs, Tab,Button,Select, DataTable,Static,ProgressBar,Switch
from training import main 
from textual.containers import Container
from plots import *
from model_handler import * 
from textual_plotext import PlotextPlot  
import tomllib
import tomli_w

import asyncio

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

class Terminal(App):
    def compose(self):
        yield Header()

        yield Tabs(
            Tab("Main",id = "tab_main"),
            Tab("Dashboard", id="tab_dashboard"),
            Tab("Parameters", id="tab_parameters")
        )

        yield Container(Static("TUInning"),PlotextPlot(id="stock_price_plot"),DataTable(id = "table_models"),id= "main_content")

    
        yield Container(
            Input(placeholder="Ticker", id="ticker_input"),
            Select(
            options=[
                ("Linear", "Linear"),
                ("LSTM", "LSTM"),
            ],value= "Linear",
            prompt="model_type",
            id="model_type"
        ),ProgressBar(total=None,id = "epochs_progressbar"),
            PlotextPlot(id="history_plot"),
            PlotextPlot(id="prediction_plot"),
            DataTable(id="history_dataTable"),
            id="dashboard_content"
        )

        yield Container(
            Input(placeholder="Sigma", id="sigma"),
            Input(placeholder="Epochs", id="epochs"),
            Static("Gaussian Noise"),
            Switch(id ="switch_Gaussian"),
            Button(label="Submit"),
            id="parameters_content"
        )

         
        yield Footer()

    async def on_mount(self):
        self.query_one("#history_plot",PlotextPlot).display = False
        self.query_one("#prediction_plot",PlotextPlot).display = False
        self.query_one("#history_dataTable",DataTable).display = False
        model = create_ModelTable(self.query_one("#table_models",DataTable)) 
        action_plot_text_widget(self.query_one("#stock_price_plot",PlotextPlot))
        
        self.model = model 
       

    async def on_tabs_tab_activated(self, event: Tabs.TabActivated):
        dashboard = self.query_one("#dashboard_content")
        params = self.query_one("#parameters_content")
        main = self.query_one("#main_content")


        if event.tab.id == "tab_dashboard":
            dashboard.display = True
            params.display = False
            main.display = False
            self.query_one("#epochs_progressbar",ProgressBar).display = False
        elif event.tab.id == "tab_parameters":
            dashboard.display = False
            params.display = True
            main.display = False
        else:
            dashboard.display = False
            params.display = False
            main.display = True
            self.query_one("#table_models",DataTable).clear(columns=True)
            create_ModelTable(self.query_one("#table_models",DataTable)) 


    
    async def on_input_submitted(self, event: Input.Submitted):
       if event.input.id == "ticker_input":
        
        history_widget = self.query_one("#history_plot", PlotextPlot)
        prediction_widget = self.query_one("#prediction_plot", PlotextPlot)
        data_Table = self.query_one("#history_dataTable",DataTable)
        progressBar = self.query_one("#epochs_progressbar")
        model_type = self.query_one("#model_type", Select).value
        switch_gaussian_noise = self.query_one("#switch_Gaussian").value

        history_widget.display = False
        prediction_widget.display = False
        data_Table.display = False

        data_Table.clear(columns=True)

        result, mean, std, model, wg = await asyncio.to_thread(
            main,
            progressBar,switch_gaussian_noise,event.value,model_type
        )

        progressBar.display = False
        progressBar.update(progress=0) 

        history_widget.display = True
        prediction_widget.display  = True
        data_Table.display = True

        history_plot_text_widget(history_widget, result)
        plot_prediction_test_general_widget(
            prediction_widget,
            wg,
            model,
            mean["Close"].item(),
            std["Close"].item(),
            3
        )

        make_dataTable(data_Table,result)

    
    async def on_button_pressed(self,event: Button.Pressed): 
        sigma = self.query_one("#sigma", Input).value
        epochs = self.query_one("#epochs", Input).value
        
        Safe = True 

        if sigma is None: 
            Safe = False
        if epochs is None: 
            Safe = False
        
        if not is_number(sigma) or not is_number(epochs): 
            Safe = False
        else: 
            if (float(sigma) > 1 or float(sigma) <= 0) or (int(epochs) <= 0):
                Safe = False

        
        if Safe:
            
            with open("parameters.toml", "rb") as f:
                parameters = tomllib.load(f)

            parameters["EPOCHS"] = int(epochs)
            parameters["SIGMA"] = float(sigma)

            with open("parameters.toml", "wb") as f:
                tomli_w.dump(parameters, f)

            
if __name__ == "__main__": 
    Terminal().run()