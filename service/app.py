import streamlit as st
import spacy
import torch
import numpy as np
from PIL import Image
from models import Generator, TextTransformer

spacy.prefer_gpu()
nlp = spacy.load("en_core_web_md")

# Подготовка генератора
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
generator = Generator(z_dim=100).to(device)
generator.load_state_dict(torch.load(r"../GAN/models/generator_15.pth", map_location=device))
generator.eval()

def preprocess_text(text):
    z_dim = 100
    random_data = torch.randn(1, z_dim)
    text_transformer = TextTransformer(nlp_model=nlp)
    text_vector = text_transformer.transform(text)
    z = torch.cat((random_data, text_vector), dim=1)
    return z

# Преобразование тензора в изображение
def postprocess_output(output): 
    image = (output.cpu().detach().numpy().squeeze(0) * 255).astype("uint8")
    image = np.transpose(image, (1, 2, 0))
    return image

st.title("Генерация логотипа по текстовому описанию")

description = st.text_input("Введите описание логотипа:")

if st.button("Сгенерировать логотип"):
    if description:
        try:
            input_tensor = preprocess_text(description)

            with torch.no_grad():
                output = generator(input_tensor.to(device))

            image = postprocess_output(output)

            pil_image = Image.fromarray(image)
            st.image(pil_image, caption="Сгенерированный логотип", use_container_width=True)
        except Exception as e:
            st.error(f"Ошибка: {e}")
    else:
        st.warning("Пожалуйста, введите описание.")