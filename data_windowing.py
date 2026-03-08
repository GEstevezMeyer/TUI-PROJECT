import pandas as pd 
import tensorflow as tf 
import numpy as np 


class WindowGenerator():
    def __init__(self,df:pd.DataFrame,input_width:int, shift:int,p_val_df:float = 0.2, p_test_df: float = 0.1,label_width:int = 1):

        if isinstance(df, pd.Series):
            df = df.to_frame()
        self.raw_df = df 
        self.input_width = input_width
        self.label_width = label_width
        self.shift = shift

        if (p_test_df+p_val_df) < 1:
            training_end = int((len(df)*(1-p_test_df-p_val_df)))
            validation_end = int((len(df)*(1-p_val_df)))
            self.training_df = self.raw_df.iloc[0:training_end:1].to_numpy()
            self.val_df = self.raw_df.iloc[training_end:validation_end:1].to_numpy()
            self.test_df = self.raw_df.iloc[validation_end::1].to_numpy()


        self.training_input,self.training_label = self.split(self.training_df,self.input_width,self.shift,self.label_width)
        self.val_input,self.val_label = self.split(self.val_df,self.input_width,self.shift,self.label_width)
        self.test_input,self.test_label = self.split(self.test_df,self.input_width,self.shift,self.label_width)

    
    @staticmethod
    def split(arr: np.ndarray, input_width: int, shift: int, label_width: int) -> tuple[np.ndarray, np.ndarray]:

        inputs = []
        labels = []

        for i in range(len(arr) - input_width - shift - label_width + 1):
            inputs.append(arr[i:i+input_width])
            labels.append(arr[i + input_width + shift - 1 :i + input_width + shift - 1 + label_width])

        return np.array(inputs), np.array(labels)

            


         
        
if __name__ == "__main__": 

    data = pd.DataFrame({
        "x": np.arange(10)
    })

    wg = WindowGenerator(data, input_width=3, shift=1, label_width=1)

    print("Train input shape:", wg.training_input.shape)
    print("Train label shape:", wg.training_label.shape)

    print("\nFirst input window:")
    print(wg.training_input[0])

    print("\nFirst label:")
    print(wg.training_label[0])



