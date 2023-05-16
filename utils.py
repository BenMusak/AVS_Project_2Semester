import joblib
import os


def save_model(model, model_name):
    file_name = "{model_name}.sav".format(model_name=model_name)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(current_dir, "models")
    file_path = os.path.join(models_dir, file_name)
    joblib.dump(model, file_path)
    return file_path
