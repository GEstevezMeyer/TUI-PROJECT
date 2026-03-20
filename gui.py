import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2" 
import warnings
warnings.filterwarnings(
    "ignore",
    message=".*TensorFlow GPU support is not available on native Windows.*"
)

from textual.app import App
from textual.widgets import Header, Footer, Input
from training import main 
from plots import *
from textual_plotext import PlotextPlot  

class Terminal(App):
    def compose(self):
        yield Header()
        self.input_field = Input(placeholder="Ticket")
        yield self.input_field
        yield PlotextPlot(id="history_plot")       
        yield PlotextPlot(id="prediction_plot")  
    
        yield Footer()
    
    async def on_input_submitted(self, event: Input.Submitted):
        self.input_field.disabled = True 
        result,mean,std,model,wg = main(event.value)
        history_widget = self.query_one("#history_plot", PlotextPlot)
        prediction_widget = self.query_one("#prediction_plot", PlotextPlot)

        history_plot_text_widget(history_widget, result)
        plot_prediction_test_general_widget(prediction_widget,wg,model,mean["Close"].item(),std["Close"].item(),3)
        




if __name__ == "__main__": 
    Terminal().run()