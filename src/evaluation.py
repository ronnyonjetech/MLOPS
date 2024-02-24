import logging
from abc import ABC,abstractmethod
import numpy as np
from sklearn.metrics import mean_squared_error,r2_score,mean_squared_error
class Evaluation(ABC):
    '''
    Abstract class evaluating model
    '''
    @abstractmethod
    def calculate_scores(self,y_true:np.ndarray,y_pred:np.ndarray):
        pass

class MSE(Evaluation):
    '''
    Evaluation strategy that uses MSE
    '''
    def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray):
        try:
            logging.info("Calculating Mean Squared Error")
            mse=mean_squared_error(y_true,y_pred)
            logging.info("MSE: {}".format(mse))
            return mse
        except Exception as e:
            logging.error("Error in calculating MSE: {}".format(e))
            raise e
        
class R2(Evaluation):
     def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray):
        try:
            logging.info("Calculating R2 Score")
            r2=r2_score(y_true,y_pred)
            logging.info("R2: {}".format(r2))
            return r2
        except Exception as e:
            logging.error("Error in calculating R2: {}".format(e))
            raise e
        
class RMSE(Evaluation):
    def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray):
            try:
                logging.info("Calculating RMSE Score")
                rmse=mean_squared_error(y_true,y_pred)
                logging.info("RMSE: {}".format(rmse))
                return rmse
            except Exception as e:
                logging.error("Error in calculating RMSE: {}".format(e))
                raise e