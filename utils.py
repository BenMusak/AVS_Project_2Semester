import joblib
import os

def save_model(model):
    file_name = "KNN_model.sav"
    current_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(current_dir, "models")
    file_path = os.path.join(models_dir, file_name)
    joblib.dump(model, file_path)
    return file_path



def load_model(file_path):
    return joblib.load(file_path)
