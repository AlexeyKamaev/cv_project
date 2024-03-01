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


def create_opencv_image_from_bytesio(img_bytesio, cv2_img_flag=1):
    img_bytesio.seek(0)
    img_array = np.asarray(bytearray(img_bytesio.read()), dtype=np.uint8)
    return cv2.imdecode(img_array, cv2_img_flag)

st.set_option('deprecation.showfileUploaderEncoding', False)
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


# Dataset information
st.markdown("""
  <p style='font-size:20px'>
    <strong>Информация о датасете</strong><br>
    Number of train images: 2643<br>
    Number of test images: 130<br>
    Number of valid images: 247
  </p>
""", unsafe_allow_html=True)

# Model training information
st.markdown("""
  <p style='font-size:20px'>
    <strong>Информация о модели</strong><br>
    Модель YOLOv8n обучалась на 60 эпохах
  </p>
""", unsafe_allow_html=True)



# Load and display graphs
graph_path = "media_2/results.png"  # Replace with the actual path
confusion_matrix_path = "media_2/confusion_matrix.png"  # Replace with the actual path

# Load graphs
graph_img = mpimg.imread(graph_path)
confusion_matrix_img = mpimg.imread(confusion_matrix_path)

# Display graphs
st.markdown("""
  <p style='font-size:20px'>
    <strong>Графики</strong>
  </p>
""", unsafe_allow_html=True)
st.image(graph_img, caption="Графики", use_column_width=True)

st.markdown("""
  <p style='font-size:20px'>
    <strong>Confusion Matrix</strong>
  </p>
""", unsafe_allow_html=True)
st.image(confusion_matrix_img, caption="Confusion Matrix", use_column_width=True)







