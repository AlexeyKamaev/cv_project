import streamlit as st
import PIL
import cv2
import numpy as np
import requests
import urllib

from ultralytics import YOLO
from io import BytesIO
from requests.models import MissingSchema
from PIL import Image

st.set_option('deprecation.showfileUploaderEncoding', False)

# Функция для загрузки изображения по URL и предварительной обработки
st.markdown("""
  <p style='text-align: center; font-size:36px'>
    Обнаружение ветрогенераторов с использованием YOLO
    
    
    
  </p>
""", unsafe_allow_html=True)

st.page_link("main.py", label="Home", icon="🏠")

st.markdown("""
    <style>
        section[data-testid="stSidebar"][aria-expanded="true"]{
            display: none;
        }
    </style>
    """, unsafe_allow_html=True)


def create_opencv_image_from_bytesio(img_bytesio, cv2_img_flag=1):
    img_bytesio.seek(0)
    img_array = np.asarray(bytearray(img_bytesio.read()), dtype=np.uint8)
    return cv2.imdecode(img_array, cv2_img_flag)


st.markdown("""
  <p 
  </p>
""", unsafe_allow_html=True)

model_path = "models/best.pt"

st.set_option('deprecation.showfileUploaderEncoding', False)


uploaded_files = st.file_uploader(
    "Загрузите изображение", type=["jpg", "png", "jpeg"], accept_multiple_files=True
)

url = st.text_input("Введите URL изображения")

if uploaded_files:
    for uploaded_file in uploaded_files:
        image = Image.open(uploaded_file)
        image = image.convert('RGB')
        image_bytesio = BytesIO()
        image.save(image_bytesio, format='JPEG')
        cv2_image = create_opencv_image_from_bytesio(image_bytesio)

        try:
            model = YOLO(model_path)
        except Exception as ex:
            st.error(f"Не удалось загрузить модель. Проверьте указанный путь: {model_path}")
            st.error(ex)
            continue

        st.image(image, caption="Загруженное изображение", use_column_width=True)

        if len(uploaded_files) == 1 and st.button("Обнаружить"):
            result = model.predict(cv2_image)
            st.image(result[0].plot()[:, :, ::-1], caption="Результаты детекции", use_column_width=True)

elif url:
    try:
        model = YOLO(model_path)
    except Exception as ex:
        st.error(f"Не удалось загрузить модель. Проверьте указанный путь: {model_path}")
        st.error(ex)

    response = requests.get(url)
    if response.status_code == 200:
        image = Image.open(BytesIO(response.content))
        image = image.convert('RGB')
        image_bytesio = BytesIO()
        image.save(image_bytesio, format='JPEG')
        cv2_image = create_opencv_image_from_bytesio(image_bytesio)

        st.image(image, caption="Загруженное изображение", use_column_width=True)

        if st.button("Обнаружить"):
            result = model.predict(cv2_image)
            st.image(result[0].plot()[:, :, ::-1], caption="Результаты детекции", use_column_width=True)
    else:
        st.error("Ошибка при получении изображения", response.status_code)