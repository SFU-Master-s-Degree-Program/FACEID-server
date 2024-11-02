# utils.py
import numpy as np
from PIL import Image
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
import io
import pickle

# Инициализация моделей MTCNN и FaceNet с параметром weights_only=True для безопасности
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
mtcnn = MTCNN(image_size=160, margin=0, min_face_size=20, device=device)
resnet = InceptionResnetV1(pretrained='vggface2', classify=False).eval().to(device)



def extract_embedding(image_bytes):
    # Открываем изображение
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')

    # Обнаруживаем лицо
    face = mtcnn(image)
    if face is None:
        return None

    # Извлекаем эмбеддинг
    with torch.no_grad():
        embedding = resnet(face.unsqueeze(0).to(device))
    return embedding.cpu().numpy()


def serialize_embedding(embedding):
    return pickle.dumps(embedding)


def deserialize_embedding(embedding_bytes):
    return pickle.loads(embedding_bytes)