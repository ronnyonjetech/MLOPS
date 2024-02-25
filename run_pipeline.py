from pipelines.training_pipeline import train_pipeline
from zenml.client import Client

if __name__=="__main__":
    print(Client().active_stack.experiment_tracker.get_tracking_uri())
    train_pipeline(data_path=r'C:\Users\ronny\OneDrive\Desktop\MLOPS\data\diabetes.csv')
                    
#mlflow ui --backend-store-uri "file:C:\Users\ronny\AppData\Roaming\zenml\local_stores\f4f6468f-5c18-4daf-824f-694b55a1679d\mlruns"