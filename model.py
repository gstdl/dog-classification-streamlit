import torch
import json
import cv2
from torchvision import models, transforms
from PIL import Image, ImageOps
import numpy as np
import streamlit as st

## face detector
face_cascade = cv2.CascadeClassifier("models/haarcascade_frontalface_alt.xml")


def face_detector(img):
    img = np.asarray(img)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    faces = face_cascade.detectMultiScale(gray)
    return len(faces) > 0


## preprocessing for pytorch models
def transform_img(img):
    preprocess = transforms.Compose(
        [
            transforms.Resize([224, 224]),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    return preprocess(img).unsqueeze(0)


## dog detector
VGG16 = models.vgg16(pretrained=True)
VGG16.eval()


def dog_detector(img):
    pred_proba = VGG16(img).detach().numpy()
    pred = np.argmax(pred_proba)
    pred = 151 <= pred <= 268
    return pred


## breed
model_transfer = torch.load(
    "models/model_transfer.pth", map_location=torch.device("cpu")
)
model_transfer.eval()
with open("models/classes.json", "r") as f:
    class_names = json.load(f)


def predict_breed_transfer(img):
    pred_proba = model_transfer(img)
    _, pred = torch.topk(pred_proba, dim=1, k=1)
    pred = str(pred.detach().numpy()[0][0])
    pred = class_names[pred]
    return pred


## final predictor
def run_app(img):
    human = face_detector(img)
    img = transform_img(img)
    dog = dog_detector(img)
    if dog + human > 0:
        dog_breed = predict_breed_transfer(img)
        if dog:
            st.header("hello, dog!")
        else:
            st.header("hello, human!")
        st.header(f"You look like a {dog_breed}")
    else:
        st.header("um, what are you? Are you an alien!")
