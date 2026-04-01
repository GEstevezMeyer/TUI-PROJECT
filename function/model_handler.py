import os 
import tomllib
from textual.widgets import DataTable



def get_numbers_of_models(): 
    res = {}
    res = os.listdir("tfKerasModels")
    return res 




def create_rows(widget:DataTable,ticker:str) -> None:
    with open(f"tfKerasModels/{ticker}/config.toml","rb") as f:
        toml_data = tomllib.load(f)

    if not widget.columns:
        widget.add_columns("ticker")
        for columns in toml_data.keys(): 
            widget.add_column(columns)

    row_values = [toml_data[key] for key in toml_data.keys()]
    row_values.insert(0,ticker)
    widget.add_row(*row_values)

def create_ModelTable(widget:DataTable):
    models = get_numbers_of_models()

    for ticker in models: 
        create_rows(widget,ticker)

    return models


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False



if __name__ == "__main__": 
    print(get_numbers_of_models()) 