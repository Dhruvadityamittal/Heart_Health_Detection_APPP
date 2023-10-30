import mlflow
from preprocess import get_prepocessed_data

model_name = "KNeighborsClassifier"
model_version = 1
loaded_model = mlflow.sklearn.load_model(model_uri = f"models/{model_name}/{model_version}")


x_train,y_train,x_test,y_test= get_prepocessed_data()

print(loaded_model.predict(x_test))