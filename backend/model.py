import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from PIL import Image
import os
import joblib

class PlaceRecognizer:
    def __init__(self, dataset_path='dataset'):
        self.dataset_path = dataset_path
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Трансформации для обучения
        self.train_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        # Трансформации для валидации/теста
        self.val_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        

        self.model = models.resnet50(pretrained=True)

        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, len(os.listdir(dataset_path)))
        self.model = self.model.to(self.device)
        
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9)
        
        self.features_model = nn.Sequential(*list(self.model.children())[:-1])
        self.features = []
        self.image_paths = []
        self.labels = []
        self.label_names = []
    
    
    def recognize_place(self, query_img_path):
        """Распознавание места и поиск похожих изображений"""
        try:
            # Загружаем и преобразуем изображение
            img = Image.open(query_img_path).convert('RGB')
            img_tensor = self.val_transform(img).unsqueeze(0).to(self.device)
            
            # Извлекаем признаки
            with torch.no_grad():
                self.features_model.eval()
                query_features = self.features_model(img_tensor)
                query_features = query_features.cpu().numpy().flatten()
            
            # Поиск ближайших соседей
            distances, indices = self.nn.kneighbors([query_features])
            
            # Получение результатов
            label_idx = self.labels[indices[0][0]]
            predicted_label = self.label_names[label_idx]
            similar_images = [self.image_paths[i] for i in indices[0]]
            similar_distances = distances[0]
            
            return predicted_label, list(zip(similar_images, similar_distances))
        except Exception as e:
            print(f"Ошибка при распознавании: {e}")
            return None, []
        
    def save_model(self, save_dir='model'):
        """Сохранение всей модели и данных"""
        os.makedirs(save_dir, exist_ok=True)
        
        # Сохраняем PyTorch модель
        torch.save({
            'model_state': self.model.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
        }, os.path.join(save_dir, 'model_weights.pth'))
        
        # Сохраняем остальные данные с помощью joblib
        data_to_save = {
            'features': self.features,
            'image_paths': self.image_paths,
            'labels': self.labels,
            'label_names': self.label_names,
            'nn': self.nn
        }
        joblib.dump(data_to_save, os.path.join(save_dir, 'recognizer_data.joblib'))
        
        print(f"Модель сохранена в папке {save_dir}")


    def initialize_model(self):
        """Инициализация или переинициализация модели"""
        # Загружаем предобученную ResNet50
        self.model = models.resnet50(pretrained=True)
        
        # Заменяем последний слой для нашего числа классов
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, len(os.listdir(self.dataset_path)))
        self.model = self.model.to(self.device)
        
        # Инициализируем оптимизатор и критерий
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9)
        
        # Модель для извлечения признаков (без последнего слоя)
        self.features_model = nn.Sequential(*list(self.model.children())[:-1])
        
        # Инициализируем структуры для хранения данных
        self.features = []
        self.image_paths = []
        self.labels = []
        self.label_names = []
        self.nn = None
    
    
    def load_model(self, save_dir='model'):
        """Загрузка модели и данных"""
        # Загружаем веса модели
        checkpoint = torch.load(os.path.join(save_dir, 'model_weights.pth'))
        self.initialize_model()  # Переинициализируем модель
        self.model.load_state_dict(checkpoint['model_state'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        
        # Загружаем остальные данные
        data_loaded = joblib.load(os.path.join(save_dir, 'recognizer_data.joblib'))
        self.features = data_loaded['features']
        self.image_paths = data_loaded['image_paths']
        self.labels = data_loaded['labels']
        self.label_names = data_loaded['label_names']
        self.nn = data_loaded['nn']
        
        # Переводим модель в eval режим
        self.model.eval()
        self.features_model = nn.Sequential(*list(self.model.children())[:-1]).to(self.device)
        
        print(f"Модель загружена из папки {save_dir}")