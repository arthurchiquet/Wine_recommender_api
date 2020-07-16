import streamlit as st
import numpy as np
import pandas as pd
import altair as alt
from Wine_recommender_api.data import get_data, get_params
from Wine_recommender_api.utils import lowercase, compute_distance
from Wine_recommender_api.input_prep import clean_input, transform
from Wine_recommender_api.browser import bottle_mode, flavors_mode
from PIL import Image

@st.cache(allow_output_mutation=True)
def read_data():
    return get_data()

@st.cache(allow_output_mutation=True)
def read_params(mode):
    return get_params(mode)

if __name__ == "__main__":

    mode = st.sidebar.selectbox('Browsing mode:', ('By bottle name','By aromas description'))
    image = Image.open('pictures/logo.png')
    st.sidebar.image(image, use_column_width=True)
    if mode == 'By bottle name':
        user_input = st.sidebar.text_input('Type your desired wine: ')
        bottle_mode(user_input, read_data(), read_params('half'))
    else :
        user_input = st.sidebar.text_input('Write a short description of your wine: ')
        flavors_mode(user_input, read_data(), read_params('all'))
    help = st.sidebar.checkbox('help')
    if help:
        st.sidebar.markdown("<p style='text-align: center; font-style: italic; color: Gray;'>BY BOTTLE NAME : 1. Select your favorite wine. 2. Then, select the number of recommended bottles. 3. Then, select the country of the recommended wines or let the filed blank for a global research. 4. The system will return the closest wines according to their aromatic description </p>", unsafe_allow_html=True)
        st.sidebar.markdown("<p style='text-align: center; font-style: italic; color: Gray;'>BY AROMAS DESCRIPTION : 1. Write a short description of your favorite aromas. 2. Then, Select the number of recommended bottles. 3. Then, select the country of the recommended wines or let the filed blank for a global research. 4. The system will return the wines with the most similar aromas according to your description </p>", unsafe_allow_html=True)
