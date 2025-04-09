import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import random
from typing import List, Tuple, Dict
import cv2
from monai import Compose, LoadImage, AddChannel, ScaleIntensity, Resize, RandGaussianNoise, RandGaussianSmooth, RandAdjustContrast, RandRotate, RandFlip, EnsureType, ToTensor
from monai.data import CacheDataset
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from skimage.feature import graycomatrix, graycoprops
import pywt
from monai.networks.nets import DenseNet121
from monai.utils import set_determinism


# Définition des constantes
NUM_CLASSES = 3  # Normal, Bénin, Malin
BATCH_SIZE = 16
LEARNING_RATE = 0.001
NUM_EPOCHS = 100
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGE_SIZE = (256, 256)

# Pour la reproductibilité
set_determinism(seed=42)

# ============ 1. PRÉTRAITEMENT AVEC MONAI ============
def get_transforms(mode="train"):
    """Obtenir les transformations MONAI pour le prétraitement des images"""
    if mode == "train":
        return Compose([
            LoadImage(image_only=True),  # Charge l'image (fonctionne avec DICOM et autres formats)
            AddChannel(),                # Ajoute une dimension de canal (MONAI attend N,C,H,W)
            ScaleIntensity(),            # Normalisation des intensités
            Resize(spatial_size=IMAGE_SIZE),  # Redimensionnement
            RandGaussianNoise(prob=0.5, mean=0.0, std=0.1),  # Bruit gaussien aléatoire
            RandGaussianSmooth(prob=0.5, sigma_x=(0.5, 1.5)),  # Lissage gaussien aléatoire
            RandAdjustContrast(prob=0.5, gamma=(0.9, 1.1)),  # Ajustement de contraste aléatoire
            RandRotate(range_x=np.pi/12, prob=0.5, keep_size=True),  # Rotation aléatoire
            RandFlip(spatial_axis=0, prob=0.5),  # Retournement horizontal aléatoire
            EnsureType(),                # Assure le type PyTorch
            ToTensor()                   # Conversion en tensor PyTorch
        ])
    else:  # mode == "test" ou "val"
        return Compose([
            LoadImage(image_only=True),
            AddChannel(),
            ScaleIntensity(),
            Resize(spatial_size=IMAGE_SIZE),
            EnsureType(),
            ToTensor()
        ])

# ============ 2. EXTRACTION DE CARACTÉRISTIQUES ============
class FeatureExtractor:
    """Classe pour l'extraction des caractéristiques des images CT pulmonaires"""
    
    def extract_histogram_features(self, image, bins=32):
        """Extraction des caractéristiques d'histogramme"""
        # Convertir le tensor en numpy si nécessaire
        if torch.is_tensor(image):
            image_np = image.cpu().numpy()
        else:
            image_np = image
            
        # S'assurer que l'image est 2D
        if len(image_np.shape) > 2:
            if image_np.shape[0] == 1:  # Si c'est [1, H, W]
                image_np = image_np.squeeze(0)
            else:  # Si c'est [C, H, W] avec C > 1
                image_np = np.mean(image_np, axis=0)  # Moyenne sur les canaux
        
        # Normaliser entre 0 et 255 pour calcHist
        image_np = ((image_np - np.min(image_np)) / (np.max(image_np) - np.min(image_np) + 1e-7) * 255).astype(np.uint8)
        
        hist = cv2.calcHist([image_np], [0], None, [bins], [0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        return hist
    
    def extract_texture_features(self, image, distances=[1, 3], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4]):
        """Extraction des caractéristiques de texture avec GLCM"""
        # Convertir le tensor en numpy si nécessaire
        if torch.is_tensor(image):
            image_np = image.cpu().numpy()
        else:
            image_np = image
            
        # S'assurer que l'image est 2D
        if len(image_np.shape) > 2:
            if image_np.shape[0] == 1:  # Si c'est [1, H, W]
                image_np = image_np.squeeze(0)
            else:  # Si c'est [C, H, W] avec C > 1
                image_np = np.mean(image_np, axis=0)  # Moyenne sur les canaux
        
        # Conversion en entiers pour GLCM
        img_uint8 = ((image_np - np.min(image_np)) / (np.max(image_np) - np.min(image_np) + 1e-7) * 255).astype(np.uint8)
        
        # Calcul de la matrice de co-occurrence
        glcm = graycomatrix(img_uint8, distances, angles, 256, symmetric=True, normed=True)
        
        # Calcul des propriétés
        contrast = graycoprops(glcm, 'contrast').flatten()
        dissimilarity = graycoprops(glcm, 'dissimilarity').flatten()
        homogeneity = graycoprops(glcm, 'homogeneity').flatten()
        energy = graycoprops(glcm, 'energy').flatten()
        correlation = graycoprops(glcm, 'correlation').flatten()
        ASM = graycoprops(glcm, 'ASM').flatten()
        
        # Concaténation des caractéristiques
        texture_features = np.concatenate([contrast, dissimilarity, homogeneity, energy, correlation, ASM])
        return texture_features
    
    def extract_wavelet_features(self, image, wavelet='db1', level=3):
        """Extraction des caractéristiques d'ondelettes"""
        # Convertir le tensor en numpy si nécessaire
        if torch.is_tensor(image):
            image_np = image.cpu().numpy()
        else:
            image_np = image
            
        # S'assurer que l'image est 2D
        if len(image_np.shape) > 2:
            if image_np.shape[0] == 1:  # Si c'est [1, H, W]
                image_np = image_np.squeeze(0)
            else:  # Si c'est [C, H, W] avec C > 1
                image_np = np.mean(image_np, axis=0)  # Moyenne sur les canaux
        
        # Normalisation de l'image
        img_norm = (image_np - np.min(image_np)) / (np.max(image_np) - np.min(image_np) + 1e-7)
        
        # Transformation en ondelettes
        coeffs = pywt.wavedec2(img_norm, wavelet, level=level)
        
        # Extraction des statistiques des coefficients
        wavelet_features = []
        for i, coeff_tuple in enumerate(coeffs):
            if i == 0:  # Approximation
                features = self._get_stats(coeff_tuple)
                wavelet_features.extend(features)
            else:  # Détails
                for detail in coeff_tuple:
                    features = self._get_stats(detail)
                    wavelet_features.extend(features)
        
        return np.array(wavelet_features)
    
    def _get_stats(self, array):
        """Calcule les statistiques d'un tableau"""
        return [
            np.mean(array),
            np.std(array),
            np.min(array),
            np.max(array),
            np.median(array),
            np.percentile(array, 25),
            np.percentile(array, 75)
        ]
    
    def extract_all_features(self, image):
        """Extraction de toutes les caractéristiques"""
        hist_features = self.extract_histogram_features(image)
        texture_features = self.extract_texture_features(image)
        wavelet_features = self.extract_wavelet_features(image)
        
        # Concaténation de toutes les caractéristiques
        all_features = np.concatenate([hist_features, texture_features, wavelet_features])
        return all_features

# ============ 3. RÉDUCTION DE DIMENSION AVEC LDA ============
class DimensionalityReducer:
    """Classe pour la réduction de dimension avec LDA"""
    
    def __init__(self, n_components=None):
        self.lda = LinearDiscriminantAnalysis(n_components=n_components)
        self.is_fitted = False
        
    def fit(self, features, labels):
        """Entraînement du LDA"""
        self.lda.fit(features, labels)
        self.is_fitted = True
        
    def transform(self, features):
        """Transformation des caractéristiques avec LDA"""
        if not self.is_fitted:
            raise ValueError("Le LDA doit être entraîné avant la transformation")
        return self.lda.transform(features)

# ============ 4. CLASSIFICATION AVEC ODNN OPTIMISÉ PAR MGSA ============
class ODNN(nn.Module):
    """Réseau de neurones optimisé pour la classification des images CT pulmonaires"""
    
    def __init__(self, input_size, hidden_layers, num_classes):
        super(ODNN, self).__init__()
        self.layers = nn.ModuleList()
        
        # Couche d'entrée
        self.layers.append(nn.Linear(input_size, hidden_layers[0]))
        
        # Couches cachées
        for i in range(len(hidden_layers)-1):
            self.layers.append(nn.Linear(hidden_layers[i], hidden_layers[i+1]))
        
        # Couche de sortie
        self.layers.append(nn.Linear(hidden_layers[-1], num_classes))
        
        # Activation et dropout
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.batch_norm_layers = nn.ModuleList([
            nn.BatchNorm1d(hidden_layers[i]) for i in range(len(hidden_layers))
        ])
        
    def forward(self, x):
        for i, layer in enumerate(self.layers[:-1]):
            x = layer(x)
            x = self.batch_norm_layers[i](x)
            x = self.relu(x)
            x = self.dropout(x)
        x = self.layers[-1](x)
        return x

class MGSA:
    """Modified Gravitational Search Algorithm pour l'optimisation de l'ODNN"""
    
    def __init__(self, population_size=30, max_iter=100, G0=100, alpha=20):
        self.population_size = population_size
        self.max_iter = max_iter
        self.G0 = G0  # Constante gravitationnelle initiale
        self.alpha = alpha  # Constante de décroissance
        self.best_solution = None
        self.best_fitness = float('inf')
        
    def optimize(self, input_size, num_classes, evaluate_fn, min_layers=1, max_layers=5, 
                min_neurons=8, max_neurons=128):
        """Optimise la structure du réseau avec MGSA"""
        # Initialisation de la population
        population = []
        fitness_values = []
        
        for i in range(self.population_size):
            # Génération aléatoire d'une structure de réseau
            num_layers = random.randint(min_layers, max_layers)
            hidden_layers = [random.randint(min_neurons, max_neurons) for _ in range(num_layers)]
            population.append(hidden_layers)
            
            # Évaluation du réseau
            fitness = evaluate_fn(input_size, hidden_layers, num_classes)
            fitness_values.append(fitness)
            
            # Mise à jour de la meilleure solution
            if fitness < self.best_fitness:
                self.best_fitness = fitness
                self.best_solution = hidden_layers.copy()
        
        # Algorithme MGSA
        for t in range(self.max_iter):
            # Calcul de la constante gravitationnelle actuelle
            G = self.G0 * np.exp(-self.alpha * t / self.max_iter)
            
            # Calcul des masses
            mass_sum = sum(1 / (f + 1e-10) for f in fitness_values)
            masses = [(1 / (f + 1e-10)) / mass_sum for f in fitness_values]
            
            # Calcul des forces et accélérations
            accelerations = [[] for _ in range(self.population_size)]
            
            for i in range(self.population_size):
                for j in range(self.population_size):
                    if i != j:
                        for d in range(len(population[i])):
                            # Si l'agent i a moins de dimensions que l'agent j, nous étendons
                            if d >= len(population[i]):
                                population[i].append(random.randint(min_neurons, max_neurons))
                            
                            # Si l'agent j a moins de dimensions que l'agent i, nous l'étendons aussi
                            if d >= len(population[j]):
                                population[j].append(random.randint(min_neurons, max_neurons))
                                
                            # Calcul de la force
                            r = abs(population[i][d] - population[j][d]) + 1e-10
                            force = G * (masses[i] * masses[j]) / r
                            
                            if d >= len(accelerations[i]):
                                accelerations[i].append(0)
                            
                            # Calcul de l'accélération
                            accelerations[i][d] += force * (population[j][d] - population[i][d]) / r
            
            # Mise à jour des positions
            new_population = []
            
            for i in range(self.population_size):
                new_hidden_layers = []
                
                for d in range(len(population[i])):
                    # Calcul de la nouvelle valeur de neurones
                    if d < len(accelerations[i]):
                        new_val = int(population[i][d] + accelerations[i][d])
                    else:
                        new_val = population[i][d]
                        
                    # Limites
                    new_val = max(min_neurons, min(max_neurons, new_val))
                    new_hidden_layers.append(new_val)
                
                # Tronquer ou ajouter des dimensions si nécessaire
                if len(new_hidden_layers) < min_layers:
                    for _ in range(min_layers - len(new_hidden_layers)):
                        new_hidden_layers.append(random.randint(min_neurons, max_neurons))
                elif len(new_hidden_layers) > max_layers:
                    new_hidden_layers = new_hidden_layers[:max_layers]
                
                new_population.append(new_hidden_layers)
            
            # Évaluation de la nouvelle population
            population = new_population
            fitness_values = []
            
            for i in range(self.population_size):
                fitness = evaluate_fn(input_size, population[i], num_classes)
                fitness_values.append(fitness)
                
                # Mise à jour de la meilleure solution
                if fitness < self.best_fitness:
                    self.best_fitness = fitness
                    self.best_solution = population[i].copy()
        
        return self.best_solution, self.best_fitness

# ============ 5. DATASET ============
class LungCTDatasetMONAI(CacheDataset):
    """Dataset pour les images CT de poumons avec MONAI"""
    
    def __init__(self, data_dir, transform=None, cache_rate=1.0, train=True):
        self.data_dir = data_dir
        self.train = train
        
        # Classes: 0=Normal, 1=Bénin, 2=Malin
        self.classes = ['normal', 'benign', 'malignant']
        
        # Préparation des données
        data = self._prepare_data()
        
        # Si aucune transformation n'est fournie, utilisez les transformations par défaut
        if transform is None:
            transform = get_transforms("train" if train else "test")
        
        super().__init__(data=data, transform=transform, cache_rate=cache_rate)
        
        # Initialisation des outils d'extraction et de réduction
        self.feature_extractor = FeatureExtractor()
        self.dimensionality_reducer = None
        
        # Cache pour les caractéristiques extraites
        self.features_cache = {}
    
    def _prepare_data(self):
        """Prépare les données pour MONAI Dataset"""
        data = []
        
        for class_id, class_name in enumerate(self.classes):
            class_dir = os.path.join(self.data_dir, class_name)
            if os.path.exists(class_dir):
                files = os.listdir(class_dir)
                for file in files:
                    if file.endswith(('.jpg', '.png', '.jpeg', '.dcm')):
                        data.append({
                            "img": os.path.join(class_dir, file),
                            "label": class_id
                        })
        
        return data
    
    def extract_features(self, idx):
        """Extrait les caractéristiques d'une image"""
        data_item = super().__getitem__(idx)
        img_tensor = data_item["img"]
        label = data_item["label"]
        img_path = self.data[idx]["img"]
        
        # Vérifier si les caractéristiques sont déjà extraites
        if img_path in self.features_cache:
            features = self.features_cache[img_path]
        else:
            # Extraction des caractéristiques
            features = self.feature_extractor.extract_all_features(img_tensor)
            
            # Réduction de dimension si le réducteur est configuré
            if self.dimensionality_reducer is not None and self.dimensionality_reducer.is_fitted:
                features = self.dimensionality_reducer.transform(features.reshape(1, -1)).flatten()
            
            # Mise en cache des caractéristiques
            self.features_cache[img_path] = features
        
        return features, label, img_tensor
    
    def __getitem__(self, idx):
        """Retourne une image et son étiquette"""
        features, label, img_tensor = self.extract_features(idx)
        return {
            "img": img_tensor,  # L'image originale pour l'utilisation avec des CNN
            "features": torch.FloatTensor(features),  # Les caractéristiques extraites pour l'ODNN
            "label": torch.tensor(label, dtype=torch.long)  # L'étiquette
        }
    
    def setup_dimensionality_reduction(self, n_components=None):
        """Configure et entraîne le réducteur de dimensionnalité"""
        if not self.train:
            raise ValueError("La réduction de dimension ne peut être configurée qu'en mode d'entraînement")
        
        # Créer le réducteur
        self.dimensionality_reducer = DimensionalityReducer(n_components=n_components)
        
        # Extraire toutes les caractéristiques et étiquettes
        all_features = []
        all_labels = []
        
        for idx in range(len(self)):
            features, label, _ = self.extract_features(idx)
            all_features.append(features)
            all_labels.append(label)
        
        # Entraîner le LDA
        self.dimensionality_reducer.fit(np.array(all_features), np.array(all_labels))
        
        # Transformer toutes les caractéristiques dans le cache
        for img_path in list(self.features_cache.keys()):
            features = self.features_cache[img_path]
            reduced_features = self.dimensionality_reducer.transform(features.reshape(1, -1)).flatten()
            self.features_cache[img_path] = reduced_features
    
    def get_feature_dimension(self):
        """Retourne la dimension des caractéristiques après réduction"""
        if not self.features_cache:
            # Extraire les caractéristiques du premier élément
            features, _, _ = self.extract_features(0)
            return len(features)
        else:
            # Utiliser la première entrée du cache
            first_key = list(self.features_cache.keys())[0]
            return len(self.features_cache[first_key])

# ============ 6. MODEL HYBRIDE (CNN + ODNN) ============
class HybridModel(nn.Module):
    """
    Modèle hybride qui combine un CNN (DenseNet121) pour l'extraction 
    automatique de caractéristiques et un ODNN pour la classification.
    """
    
    def __init__(self, feature_size, hidden_layers, num_classes, pretrained=True):
        super(HybridModel, self).__init__()
        
        # CNN pour l'extraction de caractéristiques (DenseNet121 de MONAI)
        self.feature_extractor = DenseNet121(
            spatial_dims=2,  # 2D pour les images CT
            in_channels=1,   # 1 canal pour les images en niveaux de gris
            out_channels=num_classes,  # Égal au nombre de classes pour la compatibilité
            pretrained=pretrained
        )
        
        # Remplacer la couche de classification du DenseNet par une identité
        self.feature_extractor.class_layers.out = nn.Identity()
        
        # ODNN pour la classification utilisant les caractéristiques manuelles et CNN
        # DenseNet121 produit 1024 caractéristiques
        cnn_feature_size = 1024
        combined_feature_size = feature_size + cnn_feature_size
        
        self.odnn = ODNN(combined_feature_size, hidden_layers, num_classes)
        
    def forward(self, img, features):
        # Extraction de caractéristiques CNN
        cnn_features = self.feature_extractor(img)
        
        # Combinaison des caractéristiques manuelles et CNN
        combined_features = torch.cat([cnn_features, features], dim=1)
        
        # Classification avec ODNN
        output = self.odnn(combined_features)
        
        return output

# ============ 7. MAIN ============
def train_and_evaluate(data_dir, output_dir):
    """Fonction principale pour l'entraînement et l'évaluation du modèle"""
    
    # Création des datasets avec MONAI
    train_transform = get_transforms(mode="train")
    test_transform = get_transforms(mode="test")
    
    train_dataset = LungCTDatasetMONAI(
        data_dir=os.path.join(data_dir, 'train'),
        transform=train_transform,
        cache_rate=1.0,  # Mettre en cache toutes les données
        train=True
    )
    
    test_dataset = LungCTDatasetMONAI(
        data_dir=os.path.join(data_dir, 'test'),
        transform=test_transform,
        cache_rate=1.0,
        train=False
    )
    
    # Réduction de dimension avec LDA
    train_dataset.setup_dimensionality_reduction(n_components=min(NUM_CLASSES-1, 100))  # LDA limite à n_classes-1
    
    # Créer les dataloaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    
    # Fonction d'évaluation pour MGSA
    def evaluate_network(input_size, hidden_layers, num_classes):
        model = ODNN(input_size, hidden_layers, num_classes).to(DEVICE)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
        
        # Mini-entraînement pour évaluation
        model.train()
        total_loss = 0
        for batch_idx, batch_data in enumerate(train_loader):
            features = batch_data["features"].to(DEVICE)
            labels = batch_data["label"].to(DEVICE)
            
            optimizer.zero_grad()
            output = model(features)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
            # Limiter à quelques batchs pour l'évaluation
            if batch_idx >= 3:
                break
                
        return total_loss
    
    # Optimisation de la structure avec MGSA
    feature_size = train_dataset.get_feature_dimension()
    print(f"Dimension des caractéristiques extraites: {feature_size}")
    
    mgsa = MGSA(population_size=10, max_iter=10)  # Valeurs réduites pour la démonstration
    best_structure, best_fitness = mgsa.optimize(
        feature_size, NUM_CLASSES, evaluate_network, 
        min_layers=1, max_layers=3, min_neurons=16, max_neurons=64
    )
    
    print(f"Meilleure structure trouvée: {best_structure}")
    print(f"Meilleur fitness: {best_fitness}")
    
    # Création du modèle hybride final avec la structure optimisée
    model = HybridModel(
        feature_size=feature_size,
        hidden_layers=best_structure,
        num_classes=NUM_CLASSES,
        pretrained=True
    ).to(DEVICE)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)
    
    # Variables pour suivre les performances
    best_acc = 0.0
    train_losses = []
    test_accs = []
    
    # Boucle d'entraînement
    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss = 0.0
        
        for batch_idx, batch_data in enumerate(train_loader):
            images = batch_data["img"].to(DEVICE)
            features = batch_data["features"].to(DEVICE)
            labels = batch_data["label"].to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(images, features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
        epoch_loss = running_loss / len(train_loader)
        train_losses.append(epoch_loss)
        
        # Évaluation
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch_data in test_loader:
                images = batch_data["img"].to(DEVICE)
                features = batch_data["features"].to(DEVICE)
                labels = batch_data["label"].to(DEVICE)
                
                outputs = model(images, features)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        test_acc = 100 * correct / total
        test_accs.append(test_acc)
        
        # Mise à jour du scheduler
        scheduler.step(epoch_loss)
        
        print(f'Epoch {epoch+1}/{NUM_EPOCHS}, Loss: {epoch_loss:.4f}, Test Acc: {test_acc:.2f}%')
        
        # Sauvegarde du meilleur modèle
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), os.path.join(output_dir, 'best_model.pth'))
    
    # Génération des graphiques
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses)
    plt.title('Perte d\'entraînement')
    plt.xlabel('Epochs')
    plt.ylabel('Perte')
    
    plt.subplot(1, 2, 2)
    plt.plot(test_accs)
    plt.title('Précision de test')
    plt.xlabel('Epochs')
    plt.ylabel('Précision (%)')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_results.png'))
    
    # Évaluation finale sur l'ensemble de test
    model.eval()
    all_preds = []
    all_targets = []
    all_probs = []
    
    with torch.no_grad():
        for batch_data in test_loader:
            images = batch_data["img"].to(DEVICE)
            features = batch_data["features"].to(DEVICE)
            labels = batch_data["label"].to(DEVICE)
            
            outputs = model(images, features)
            probabilities = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs.data, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(labels.cpu().numpy())
            all_probs.append(probabilities.cpu().numpy())
    
    # Fusion des probabilités
    all_probs = np.v