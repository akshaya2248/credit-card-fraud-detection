import pickle
import pandas as pd
from flask import Flask, request, render_template, redirect, url_for
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
import datetime as dt


importedmodel = pickle.load(open("model.pkl", "rb"))
categorical_columns = [1, 2]
numeric_columns = [0, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

preprocessor2 = ColumnTransformer(
    [
        ("num", StandardScaler(), numeric_columns),
        ("cat", OneHotEncoder(sparse_output=False, drop="first"), categorical_columns),
    ]
)


train_data = pd.read_csv("data.csv")

testlist = [
    # fraud
    {
        "cc_num": [4613314721966],
        "category": ["grocery_pos"],
        "state": ["NC"],
        "amt": [281.06],
        "zip": [28611],
        "lat": [35.9946],
        "long": [-81.7266],
        "merch_lat": [36.430124],
        "merch_long": [-81.17948299999999],
        "age": [35],
        "hour": [1],
        "day": [2],
        "month": [1],
    },
    {
        "cc_num": [340187018810220],
        "category": ["gas_transport"],
        "state": ["TX"],
        "amt": [11.52],
        "zip": [78208],
        "lat": [29.44],
        "long": [-98.459],
        "merch_lat": [29.819364],
        "merch_long": [-99.142791],
        "age": [63],
        "hour": [1],
        "day": [2],
        "month": [1],
    },
    {
        "cc_num": [4613314721966],
        "category": ["grocery_pos"],
        "state": ["NC"],
        "amt": [281.06],
        "zip": [28611],
        "lat": [35.9946],
        "long": [-81.7266],
        "merch_lat": [36.430124],
        "merch_long": [-81.17948299999999],
        "age": [35],
        "hour": [1],
        "day": [2],
        "month": [1],
    },
    # nofraud
    {
        "cc_num": [630423337322],
        "category": ["grocery_pos"],
        "state": ["WA"],
        "amt": [107.23],
        "zip": [99160],
        "lat": [48.8878],
        "long": [-118.2105],
        "merch_lat": [49.159046999999994],
        "merch_long": [-118.186462],
        "age": [45],
        "hour": [0],
        "day": [1],
        "month": [1],
    },
]

train_data["month"] = pd.to_datetime(train_data["trans_date_trans_time"]).dt.month
train_data["day"] = pd.to_datetime(train_data["trans_date_trans_time"]).dt.dayofweek
train_data["hour"] = pd.to_datetime(train_data["trans_date_trans_time"]).dt.hour
train_data["age"] = dt.date.today().year - pd.to_datetime(train_data["dob"]).dt.year
train = train_data[
    [
        "cc_num",
        "category",
        "state",
        "amt",
        "zip",
        "lat",
        "long",
        "merch_lat",
        "merch_long",
        "age",
        "hour",
        "day",
        "month",
        "is_fraud",
    ]
]
X = train.iloc[:, :-1].values
y = train.iloc[:, -1].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)


preprocessor2.fit_transform(X_train)

pipe3 = Pipeline(steps=[("step1", preprocessor2), ("step2", importedmodel)])


app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/submit", methods=["POST", "GET"])
def submit():
    if request.method == "POST":
        category = request.form["category"]
        amount = float(request.form["amount"])
        state = request.form["state"]
        zip = float(request.form["amount"])
        latitude = float(request.form["latitude"])
        longitude = float(request.form["longitude"])
        ccnum = float(request.form["ccnum"])
        mlatitude = float(request.form["mlatitude"])
        mlongitude = float(request.form["mlongitude"])
        age = float(request.form["age"])
        hour = float(request.form["hour"])
        day = float(request.form["day"])
        month = float(request.form["month"])

        input_df = pd.DataFrame(
            {
                "cc_num": [ccnum],
                "category": [category],
                "state": [state],
                "amt": [amount],
                "zip": [zip],
                "lat": [latitude],
                "long": [longitude],
                "merch_lat": [mlatitude],
                "merch_long": [mlongitude],
                "age": [age],
                "hour": [hour],
                "day": [day],
                "month": [month],
            }
        )
        print(input_df)
        print(type(input_df))
        print(input_df.values)

        result = pipe3.predict(input_df.values)
        r = ""
        if result == 0 or result <= 0.5:
            r = "Transaction is not a fraud Transaction !"
        else:
            r = "Transaction is a fraud Transaction !"
        return render_template("result.html", labels=r)