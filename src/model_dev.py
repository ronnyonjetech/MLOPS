import logging
from abc import ABC,abstractmethod
from sklearn.linear_model import LinearRegression
class Model(ABC):
    @abstractmethod
    def train(self,X_train,y_train):
        pass
class LinearRegressionModel(Model):
    def train(self, X_train, y_train,**kwargs):
        # self.model.fit(X_train, y_train)
        # logging.info("Model Training Completed")
        try:
            reg=LinearRegression(**kwargs)
            reg.fit(X_train,y_train)
            logging.info("Model Training Completed")
            return reg
        except Exception as e:
            logging.error("Model training error {}".format(e))
            raise e