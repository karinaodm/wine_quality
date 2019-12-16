from fastapi import FastAPI
from joblib import load
from pydantic import  BaseModel

import metrics

# load model
clf = load('/model/knn.joblib')

def get_prediction(p):
    x = [[p.fixed_acidity, p.volatile_acidity, p.residual_sugar, p.chlorides,
         p.free_sulfur_dioxide, p.total_sulfur_dioxide, p.density, p.pH, p.sulphates, p.alcohol]]

    try:
        y = str(clf.predict(x)[0])
    except Exception as e:
        y = str(e)
    return {'prediction': y}


# initiate API
app = FastAPI()


# define model for post request.
class ModelParams(BaseModel):
    fixed_acidity: float
    volatile_acidity: float
    citric_acid: float
    residual_sugar: float
    chlorides: float
    free_sulfur_dioxide: float
    total_sulfur_dioxide: float
    density: float
    pH: float
    sulphates: float
    alcohol: float


@app.post("/predict")
def predict(params: ModelParams):

    pred = get_prediction(params)

    return pred
