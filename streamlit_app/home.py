import streamlit as st
from PIL import Image
import numpy as np

import html_templates

import sys
import os

ROOT_FOLDER = os.path.dirname(os.getcwd())
sys.path.append(ROOT_FOLDER)

from app.model_proxy import ModelProxy
from app.models import get_custom_densenet201
from app.data_preprocessing_scripts import preprocessing


model_proxy = ModelProxy(model=get_custom_densenet201(),
                         data_preprocess=preprocessing)

st.set_page_config(page_title="FoundFlower",
                   page_icon=":flower:")

st.write(
    '''
    <h1>Found flower</h1> <br>
    Drop below you image
    ''',
    unsafe_allow_html=True)

uploaded = st.file_uploader(label='Drop here you file', type=['png', 'jpg', 'jpeg'])

if st.button("Process"):
    if uploaded is not None:
        img = np.array(Image.open(uploaded))

        st.image(img)

        probability, labels = model_proxy(img)

        output = [html_templates.main_line.format(prob=probability[0], label=labels[0])]
        other = [html_templates.sub_line.format(prob=prob, label=lab) for prob, lab in zip(probability[1::], labels[1::])]
        output += other
        st.write("<br>".join(output),
                 unsafe_allow_html=True)

