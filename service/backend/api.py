import spacy
import torch
from models import Generator
from models import TextTransformer
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np

app = FastAPI()

spacy.prefer_gpu()
nlp = spacy.load("en_core_web_md")

# Подготовка генератора
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
generator = Generator(z_dim = 100).to(device)
generator.load_state_dict(torch.load(r"..\GAN\models\generator_15.pth", map_location=device))
generator.eval()


class TextDescription(BaseModel):
    description: str

@app.post("/generate/")
async def generate_logo(text: TextDescription):
    try:
        input_tensor = preprocess_text(text.description)

        with torch.no_grad():
            output = generator(input_tensor.to(device))

        image = postprocess_output(output)
        return {"image": image.tolist()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Функция для преобразования вывода модели в изображение
def postprocess_output(output):
    image = (output.detach().numpy().squeeze(0) * 255).astype("uint8")
    image = np.transpose(image, (1, 2, 0))
    return image

# Функция для предобработки текста во вход генератора
def preprocess_text(text):
    z_dim = 100
    random_data = torch.randn(1, z_dim)
    text_transformer = TextTransformer(nlp_model = nlp)
    text_vector = text_transformer.transform(text)
    z = torch.cat((random_data, text_vector), dim=1)
    return z