import pandas as pd
from sklearn.model_selection import train_test_split
def get_prepocessed_data():
    data = pd.read_csv("heart.csv")
    # data["Sex"].unique()

    data['Sex'] = data['Sex'].apply({'M':1, 'F':0}.get)
    data['ChestPainType'] = data['ChestPainType'].apply({'ATA':0, 'NAP':1, 'ASY' : 2, "TA":3 }.get)
    data['RestingECG'] = data['RestingECG'].apply({'Normal':0, 'ST':1, 'LVH' : 2 }.get)
    data['ExerciseAngina'] = data['ExerciseAngina'].apply({'N':0, 'Y':1}.get)
    # data['ST_Slope'] = data['ST_Slope'].apply({'Up':0, 'Down':1}.get)

    # data["ChestPainType"].unique()

    # sum(pd.is.na(data['Sex']))
    # for col in data.columns:
    #     # print(col,data[col].isna().sum())
    #     print(r"{} -> {}".format(col,data[col].isna().sum()))

    data = data.drop(columns=['ST_Slope'])

    train, test = train_test_split(data, test_size=0.2, random_state=50)
    x_train, y_train = train.iloc[:,:-1],  train.iloc[:,-1]
    x_test, y_test   = test.iloc[:,:-1],  test.iloc[:,-1]

    return x_train,y_train,x_test,y_test