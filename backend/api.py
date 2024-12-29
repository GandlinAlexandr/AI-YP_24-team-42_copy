import os  
import spacy
import torch
from models import Generator
from models import TextTransformer
from fastapi import FastAPI, HTTPException, Body
from typing import List, Annotated
from pydantic import BaseModel
import numpy as np
import logging
from logging.handlers import RotatingFileHandler
import time

# Настройка логирования
log_directory = "backend/logs"  # Измените путь к логам
if not os.path.exists(log_directory):
    os.makedirs(log_directory)

log_file_path = os.path.join(log_directory, "app.log")

# Создание обработчика для ротации логов
handler = RotatingFileHandler(log_file_path, maxBytes=10 * 1024 * 1024, backupCount=5)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[handler]
)
logger = logging.getLogger()

app = FastAPI()
logger.info("Запуск приложения FastAPI")

spacy.prefer_gpu()
nlp = spacy.load("en_core_web_md")

# Подготовка генератора
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
generator = Generator(z_dim = 100).to(device)
generator.load_state_dict(torch.load(r"generator_15.pth", map_location=device))
generator.eval()


class TextDescription(BaseModel):
    description: str


class LogoImage(BaseModel):
    image: List[List[List[int]]]  # Предполагаем, что изображение возвращается в формате 3D списка (H x W x C)


@app.post("/generate/", response_model=LogoImage)
async def generate_logo(
    text: Annotated[TextDescription, Body()]
) -> LogoImage:
    logger.info(f"Получен запрос на генерацию логотипа с текстом: {text.description}")
    start_time = time.time()  # Начало отсчета времени
    try:
        input_tensor = preprocess_text(text.description)

        with torch.no_grad():
            output = generator(input_tensor.to(device))

        image = postprocess_output(output)
        elapsed_time = time.time() - start_time  # Время генерации логотипа
        logger.info(f"Логотип успешно сгенерирован за {elapsed_time:.2f} секунд")
        return {"image": image.tolist()}
    except Exception as e:
        logger.error(f"Ошибка при обработке запроса: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# Функция для преобразования вывода модели в изображение
def postprocess_output(output: torch.Tensor) -> np.ndarray:
    image = (output.cpu().detach().numpy().squeeze(0) * 255).astype("uint8")
    image = np.transpose(image, (1, 2, 0))
    return image


# Функция для предобработки текста во вход генератора
def preprocess_text(text: str) -> torch.Tensor:
    z_dim = 100
    random_data = torch.randn(1, z_dim)
    text_transformer = TextTransformer(nlp_model=nlp)
    text_vector = text_transformer.transform(text)
    z = torch.cat((random_data, text_vector), dim=1)
    return z
