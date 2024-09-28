import numpy as np
from flask import Flask, request
from PIL import Image
import os
import sys

ROOT_FOLDER = os.path.dirname(os.getcwd())
sys.path.append(ROOT_FOLDER)

from app.model_proxy import ModelProxy
from app.models import get_custom_densenet201
from app.data_preprocessing_scripts import preprocessing


app = Flask(__name__)

ROOT_URL = '/flowerclf/api'
model_proxy = ModelProxy(model=get_custom_densenet201(), data_preprocess=preprocessing)


@app.post(ROOT_URL + '/classification')
def classification():
    img = request.files['image']
    img = np.array(Image.open(img))
    prob, lab = model_proxy(img)
    json_response = {'probability': list(prob.astype(dtype=float)), 'labels': list(lab)}
    return json_response
