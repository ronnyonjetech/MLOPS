import logging
import pandas as pd
from zenml import step
from src.data_cleaning import DataCleaning,DataSplitStrategy,DataPreProcessStrategy
from typing_extensions import Annotated
from typing import Tuple
@step
def clean_data(df:pd.DataFrame)->Tuple[
    Annotated[pd.DataFrame,"X_train"],
    Annotated[pd.DataFrame,"X_test"],
    Annotated[pd.DataFrame,"y_train"],
    Annotated[pd.DataFrame,"y_test"],] :
    try:
        process_strategy=DataPreProcessStrategy()
        data_cleaning=DataCleaning(df,process_strategy)
        processed_data=data_cleaning.handle_data()
        divide_strategy=DataSplitStrategy()
        data_cleaning=DataCleaning(processed_data,divide_strategy)
        X_train,X_test,y_train,y_test=data_cleaning.handle_data()
        logging.info("Data cleaning completed")
        y_train=y_train.to_frame()
        y_test=y_test.to_frame()
        print(type(y_train))
        print(type(y_test))
        print(type(X_train))
        print(type(X_test))
        return  X_train,X_test,y_train,y_test
    
    except Exception as e:
        logging.error("Error Cleaning data {}".format(e))
        raise e
