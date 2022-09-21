import streamlit as st
from model import run_app
from PIL import Image

## streamlit app
st.set_option("deprecation.showfileUploaderEncoding", False)

st.title("Dog or Human Classifier")
st.text(
    "Upload an Image for image classification as dog or human.\nI will also predict the dog breed in the image.\nIf you are human I will tell you which breed you look like"
)

uploaded_file = st.file_uploader(
    "Upload your image here ...", type=["jpg", "jpeg", "png"]
)
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    run_app(image)
    st.image(uploaded_file, use_column_width=True)
