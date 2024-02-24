import logging
from abc import ABC,abstractmethod

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from typing import Union
class DataStrategy(ABC):
    '''
    Abstract class for defining strategy for handling data
    '''
    @abstractmethod
    def handle_data(self,data:pd.DataFrame)->Union[pd.DataFrame,pd.Series]:
        pass

class DataPreProcessStrategy(DataStrategy):
    '''
    Strategy for preprocess data
    '''
   
    def handle_data(self,data:pd.DataFrame)->pd.DataFrame:
        try:
            data = data.drop([
                "SkinThickness",
                "DiabetesPedigreeFunction"
            ],axis=1)
            return data
        except Exception as e:
            logging.error("Error in preprocessing data: {}".format(e))
            raise e
class DataSplitStrategy(DataStrategy):
    '''
    Strategy for dividing data
    '''
    def handle_data(self, data: pd.DataFrame) ->Union[pd.DataFrame,pd.Series]:
        '''
        Divide data into train and test
        '''
        try:
            X=data.drop(["Outcome"],axis=1)
            y=data["Outcome"]
            X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
            return X_train,X_test,y_train,y_test
        except Exception as e:
            logging.error("Error in dividing data : {}".format(e))
            raise e
class DataCleaning:
    '''
    class which processes the data and dividesit into train and test
    '''
    def __init__(self,data:pd.DataFrame,strategy:DataStrategy) -> None:
        self.data=data
        self.strategy=strategy

    def handle_data(self)->Union[pd.DataFrame,pd.Series]:
        '''
        handle data
        '''
        try:
            return self.strategy.handle_data(self.data)
        except Exception as e:
            logging.error("Error in handling data : {}".format(e))
            raise e
if __name__ == "__main__" :
    data=pd.read_csv(r'C:\Users\ronny\OneDrive\Desktop\MLOPS\data\diabetes.csv')
    data_cleaning=DataCleaning(data,DataPreProcessStrategy())
    data_cleaning.handle_data()