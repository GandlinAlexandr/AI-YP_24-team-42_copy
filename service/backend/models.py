import spacy
import torch
import torch.nn as nn
import numpy as np

spacy.prefer_gpu()
nlp = spacy.load("en_core_web_md")

class TextTransformer:
    def __init__(self, nlp_model, vect_size=300):
        self.nlp_model = nlp_model
        self.vect_size = vect_size

    def transform(self, text):
        doc = self.nlp_model(text)
        lemmatized_tokens = [
                                token.lemma_ for token in doc if not token.is_punct and not token.is_space
                            ]  # Лемматизация
        vectors = [
                    token.vector for token in doc if not token.is_punct and not token.is_space
                    ]  # Векторизация

        if vectors:
            mean_vector = np.mean(vectors, axis=0)[:self.vect_size]
            return torch.tensor(mean_vector, dtype=torch.float32).unsqueeze(0)
        else:
            raise ValueError("Текст не содержит значимых токенов.")
        
# ГЕНЕРАТОР
class Generator(nn.Module):
    def __init__(self, z_dim, vect_size=300):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.ConvTranspose2d(
                z_dim + vect_size, 128, kernel_size=7, stride=1, padding=0, bias=False
            ),  # тут получается не совсем монотонное расширение,
            nn.BatchNorm2d(128),  # поэтому можно увеличить z_dim
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh(),
        )

    def forward(self, z):
        z = z.view(z.size(0), z.size(1), 1, 1)
        img = self.model(z)
        return img