import mlflow
from mlflow import MlflowClient
import argparse
from preprocess import get_prepocessed_data
from sklearn.metrics import accuracy_score

parser = argparse.ArgumentParser()

parser.add_argument("--stage", type=str, default="Production",
                     help = "Stage, Production or Archive")
parser.add_argument("--version", type=int, default=1,
                     help = "Stage, Production or Archive")
parser.add_argument("--model_name", type=str, default="RandomForestClassifier",
                     help = "Type Model name")

args = parser.parse_args()
print(args.model_name)

client = mlflow.tracking.MlflowClient()

client.transition_model_version_stage(
    name=args.model_name,
    version=args.version,
    stage=args.stage
)

deployed_model = mlflow.sklearn.load_model(f"models:/{args.model_name}/{args.stage}")

x_train,y_train,x_test,y_test= get_prepocessed_data()

out_train = deployed_model.predict(x_train)
accuracy_train = accuracy_score(y_train, out_train)

out_test = deployed_model.predict(x_test)
accuracy_test = accuracy_score(y_test, out_test)


print(f"accuracy_train = {accuracy_train}, accuracy_test = {accuracy_test}")