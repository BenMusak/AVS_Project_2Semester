import joblib
import os

def save_model(model_KNN, model_LDA, model_SVM):
    file_name_KNN = "KNN_model.sav"
    file_name_LDA = "LDA_model.sav"
    file_name_SVM = "SVM_model.sav"
    current_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(current_dir, "models")
    file_path_KNN = os.path.join(models_dir, file_name_KNN)
    file_path_LDA = os.path.join(models_dir, file_name_LDA)
    file_path_SVM = os.path.join(models_dir, file_name_SVM)
    joblib.dump(model_KNN, file_path_KNN)
    joblib.dump(model_LDA, file_path_LDA)
    joblib.dump(model_SVM, file_path_SVM)
    return file_path_KNN, file_path_LDA, file_name_SVM



def load_model(file_path):
    return joblib.load(file_path)
