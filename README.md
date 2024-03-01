## cv_project
# Streamlit app with computer vision 💡
Elbrus Bootcamp | Phase-2 | Team Project

## Team🧑🏻‍💻
1. [Алексей Камаев](https://github.com/AlexeyKamaev)
2. [Марина Кочетова](https://github.com/Valera4096)

## Task 📌
Create a service for object detection with YOLOv8 and image denoising using a custom AutoEncoder class.

## Contents 📝
1. Wind Turbines Object Detection using YOLOv8 💨 [Dataset](https://www.kaggle.com/datasets/kylegraupe/wind-turbine-image-dataset-for-computer-vision)
2. Document denoising using an autoencoder 📑 [Dataset](https://drive.google.com/file/d/1LsHSn8dM8BTZ7EoWU6-n1I1BvR0p5tIx/view)

## Deployment 🎈
The service is implemented on [Streamlit](https://windandpapers.streamlit.app/)

## Libraries 📖
```python
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision.datasets import ImageFolder
from torchvision import datasets
from torchvision import transforms as T
from torchvision.io import read_image
from torch.utils.data import Dataset, random_split
import torchutils as tu
from typing import Tuple
from tqdm import tqdm
from torchvision import io
import pydensecrf.densecrf as dcrf
import pydensecrf.utils as dcrf_utils
import streamlit as st
from skimage import io
import torchvision.models as models
from PIL import Image, ImageDraw
from ultralytics import YOLO
```
## Guide 📜 
####  How to run locally?

1. To create a Python virtual environment for running the code, enter:

    ``python3 -m venv my-env``

2. Activate the new environment:

    * Windows: ```my-env\Scripts\activate.bat```
    * macOS and Linux: ```source my-env/bin/activate```

3. Install all dependencies from the *requirements.txt* file:

    ``pip install -r requirements.txt``
