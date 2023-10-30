import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
import mlflow
import joblib, pickle


import argparse
parser = argparse.ArgumentParser()
from preprocess import get_prepocessed_data

parser.add_argument("--model", type=int, default=1, help="1 :SVC(kernel='linear'), 2: RandomForestClassifier(n_estimators=100, random_state=42), 3: LogisticRegression(solver='liblinear'), 4: LinearDiscriminantAnalysis(), 5: GaussianNB(), 6: DecisionTreeClassifier(random_state=42), 7: KNeighborsClassifier(n_neighbors=2)")
parser.add_argument("--register", type=str, default="False")



experiment_name = "heart-disease-classifier"
args = parser.parse_args()


if __name__ == "__main__":
    
    models = [SVC(kernel='linear'), RandomForestClassifier(n_estimators=100, random_state=42),
              LogisticRegression(solver='liblinear'),LinearDiscriminantAnalysis(),GaussianNB(),
              DecisionTreeClassifier(random_state=42),KNeighborsClassifier(n_neighbors=2)]
    
    selected_model = models[args.model-1]
    selected_model_name = type(selected_model).__name__ 

    x_train,y_train,x_test,y_test= get_prepocessed_data()
    

    mlflow.set_experiment(selected_model_name)
    # mlflow.sklearn.autolog()

    with mlflow.start_run() as mlflow_run:
        selected_model.fit(x_train,y_train)
        out_train = selected_model.predict(x_train)
        accuracy_train = accuracy_score(y_train, out_train)

        out_test = selected_model.predict(x_test)
        accuracy_test = accuracy_score(y_test, out_test)

        mlflow.log_param("Train Accuracy", "{:.2f}".format(accuracy_train))
        mlflow.log_param("Test Accuracy", "{:.2f}".format(accuracy_test))

        mlflow.sklearn.log_model(selected_model, "model")

        mlflow_run_id = mlflow_run.info.run_id

    if(eval(args.register)):
  
        logged_model_path = f"runs:/{mlflow_run_id}/model"
        
        with mlflow.start_run(run_id=mlflow_run_id) as run:
            result = mlflow.register_model(
                logged_model_path,
                selected_model_name
            )
    

    