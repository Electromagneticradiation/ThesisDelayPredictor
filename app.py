from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

import joblib
import json

from pydantic import BaseModel

import pandas
import numpy
import shap

from fastapi import HTTPException

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],         # substitute website origin before deploying
    allow_methods=['*'],
    allow_headers=['*'],
)

model = joblib.load('model.pkl')
with open('metadata.json', 'r') as file :
    metadata = json.load(file)
features = metadata['features']
labels = metadata['labels']

class ScholarData(BaseModel) :
    attendance: float
    progress: float
    published: int
    extensions: int
    delay: float
    score: float

@app.post('/predict')
def predict(ScholarData : ScholarData) :
    try :
        data = pandas.DataFrame([ScholarData.model_dump()])
        data = data[features]   # reorder and filter

        prob = model.predict_proba(data)[0]
        index = numpy.argmax(prob)
        prediction = labels[str(index)]
        confidence = float(numpy.max(prob))

        explainer = shap.Explainer(model)
        explanation = explainer(data).values[0][:, 0]
        reason = dict(zip(features, explanation))

        return {
             'prediction' : prediction,
             'confidence' : confidence,
             'reason' : reason
        }
    
    except Exception as e :
        raise HTTPException(status_code=500, detail=str(e))
