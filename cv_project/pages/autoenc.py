import streamlit as st
import matplotlib.pyplot as plt
import numpy as np

from torchvision.transforms import ToTensor
from torchvision import transforms as T
import torch.nn.functional as F
import torch

from PIL import Image
import cv2



import datetime

from models.autoencoder import autoencoder

model = autoencoder

PATH = 'models\weights.pt'

model.load_state_dict(torch.load(PATH))


device = 'cpu'
model.to(device)

st.markdown("""
    <style>
        section[data-testid="stSidebar"][aria-expanded="true"]{
            display: none;
        }
    </style>
    """, unsafe_allow_html=True)





def get_prediction(image: str):

    preprocessing = T.Compose(
    [
        T.ToPILImage(),
        T.Resize((400, 400)),
        T.ToTensor()
    ]
)

    numpydata = np.asarray(image)

    x,y  = numpydata.shape[0], numpydata.shape[1]
    try:
        imag = cv2.cvtColor(numpydata, cv2.COLOR_BGR2GRAY)
    except:
        imag = T.ToTensor()(image)


    img = preprocessing(imag)


    model.to('cpu')
    model.eval()

    back = T.Compose(
    [
        T.ToPILImage(),
        T.Resize((x, y)),
    ]
)
    
    with torch.no_grad():
        pred = model(img.unsqueeze(0)).squeeze().detach()
        pred = back(pred)

    return pred


st.markdown("""
    <style>
        section[data-testid="stSidebar"][aria-expanded="true"]{
            display: none;
        }
    </style>
    """, unsafe_allow_html=True)

st.title('Чистка документов')

uploaded_file = st.file_uploader("Выберите изображение", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

if uploaded_file is not None:

    for image in uploaded_file:
        now = datetime.datetime.now()

        image = Image.open(image)

        st.image(image, caption="Загруженное изображение", use_column_width=None)

        prediction = get_prediction(image)
        # st.success(f"Предсказание: {prediction}")
        finish = datetime.datetime.now()
        elapsed_time = finish - now
        st.image(prediction, caption="Результат работы нейросети", use_column_width=None)
        # st.write(f' Затраченное время на обработку:  Ч : М : С : МС ----> {elapsed_time}.')
        st.write('\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n')
st.page_link("main.py", label="Home", icon="🏠")

st.title('Описание модели')

st.header("Модель училась на 20 эпохах.", divider='rainbow')
joke = st.slider("Сколько по вашему её нужно учить?", 0, 100, (20))

if joke != 20:
    st.write('Мы непременно учтём ваше мнение')
    st.caption(':blue[***Спасибо!!!!***]')

st.divider()  # 👈 Draws a horizontal rule

st.header("Объем train-выборки", divider='blue')
st.write("Объем train-выборки состовлял 12:red[69] изображения")

st.divider()  # 👈 Another horizontal rule

st.header("RMSE на тестовой выборке составил 0.014", divider='orange')
# import base64
# gif = "media\RMSE.gif"

# file_ = open(gif, "rb")
# contents = file_.read()
# data_url = base64.b64encode(contents).decode("utf-8")
# file_.close()

# st.markdown(
#     f'<img src="data:image/gif;base64,{data_url}" alt="cat gif">',
#     unsafe_allow_html=True,
# )
st.divider()  # 👈 Another horizontal rule


st.header('Код модели', divider='rainbow')

code = '''class ConvEncoder(nn.Module):

    def __init__(self):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=2, padding=1),
            nn.LazyBatchNorm2d(),
            nn.SELU()
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 128, kernel_size=3, padding=1),
            nn.LazyBatchNorm2d(),
            nn.SELU()
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=4, padding=2),
            nn.LazyBatchNorm2d(),
            nn.ReLU()
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(256, 64, kernel_size=4, padding=2),
            nn.LazyBatchNorm2d(),
            nn.Sigmoid()
        )

        self.layer1_t = nn.Sequential(
            nn.ConvTranspose2d(64, 256, kernel_size=4, padding=2),
            nn.LazyBatchNorm2d(),
            nn.SELU()
        )
        self.layer2_t = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, padding=2),
            nn.LazyBatchNorm2d(),
            nn.ReLU()
        )
        self.layer3_t = nn.Sequential(
            nn.ConvTranspose2d(128, 16, kernel_size=3, padding=1),
            nn.LazyBatchNorm2d(),
            nn.Sigmoid()
        )
        self.layer4_t = nn.Sequential(
            nn.ConvTranspose2d(16, 1, kernel_size=2, padding=1),
            nn.LazyBatchNorm2d(),
            nn.Sigmoid()
        )

    def encode(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x


    def decode(self, x):
        x = self.layer1_t(x)
        x = self.layer2_t(x)
        x = self.layer3_t(x)
        x = self.layer4_t(x)
        return x

    def forward(self, x):
        latent = self.encode(x)
        out = self.decode(latent)
        return out'''


st.code(code, language='python')

