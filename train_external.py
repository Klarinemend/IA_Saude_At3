import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import pandas as pd
from PIL import Image
from sklearn.model_selection import GroupShuffleSplit

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 1e-4

DIAGNOSTIC_MAP = {'BCC': 0, 'SCC': 1, 'ACK': 2, 'SEK': 3, 'NEV': 4, 'MEL': 5}

# Transformação específica para o Modelo Binário usando Grayscale (Tons de Cinza)
binary_train_transforms = transforms.Compose([
    transforms.RandomRotation(degrees=15),
    transforms.RandomResizedCrop(224, scale=(0.7, 0.9)), 
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    
    # Esta linha converte para cinza mas mantém os 3 canais R, G e B idênticos.
    # Assim, a ResNet pré-treinada não reclama do formato de entrada.
    transforms.Grayscale(num_output_channels=3), 
    
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Para a validação binária, aplique também o Grayscale para manter a consistência
binary_val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def get_external_group_id(filename_or_path):
    """Extrai de forma limpa o ID do grupo externo do Roboflow (ex: normal_-121)."""
    base = os.path.basename(filename_or_path)
    if "-_jpg" in base:
        return base.split("-_jpg")[0]
    return base


class SkinDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = self.df.loc[idx, 'img_path']
        label = self.df.loc[idx, 'label']
        
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception:
            image = Image.new("RGB", (224, 224), color=0)
            
        if self.transform:
            image = self.transform(image)
            
        return image, torch.tensor(label, dtype=torch.long)


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
    return running_loss / total, correct / total


def evaluate(model, dataloader, criterion, device):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
    return running_loss / total, correct / total


def train_model(model, train_loader, val_loader, save_name):
    model = model.to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    best_acc = 0.0
    print(f"\n--- Treinando Modelo Externo: {save_name} ---")
    for epoch in range(EPOCHS):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
        val_loss, val_acc = evaluate(model, val_loader, criterion, DEVICE)
        
        print(f"Época {epoch+1}/{EPOCHS} | Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")
        
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), save_name)
            print(f"=> Salvo com acurácia de validação: {val_acc:.4f}")


def main():
    metadata_path = "metadata.csv"
    images_dir = "images/"
    # APONTE PARA A PASTA DO DATASET EXTERNO DO ROBOFLOW/KAGGLE
    sem_external_dir = "data_sem_external/" 
    
    df_pad = pd.read_csv(metadata_path)
    df_pad['img_path'] = df_pad['img_id'].apply(lambda x: os.path.join(images_dir, f"{x}" if x.endswith('.png') else f"{x}.png"))
    df_lesao = df_pad[['img_path', 'patient_id', 'diagnostic']].copy()
    
    # 1. Mapeamento de grupos do PAD-UFES
    df_lesao['group_id'] = df_lesao['patient_id']
    
    # 2. Carrega imagens externas do Roboflow
    ext_images = [os.path.join(sem_external_dir, f) for f in os.listdir(sem_external_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    df_sem_ext = pd.DataFrame({'img_path': ext_images})
    df_sem_ext['diagnostic'] = 'SEM'
    
    # Mapeia o group_id usando o ID extraído do Roboflow (evita data leakage)
    df_sem_ext['group_id'] = df_sem_ext['img_path'].apply(get_external_group_id)
    
    # --- MODELO 1: BINÁRIO (Lesão vs Sem Lesão Externo) ---
    df_lesao_bin = df_lesao.copy()
    df_lesao_bin['label'] = 1
    
    df_sem_bin = df_sem_ext.copy()
    df_sem_bin['label'] = 0
    
    df_binary = pd.concat([df_lesao_bin, df_sem_bin]).reset_index(drop=True)
    
    # Separação por Grupo para evitar vazamento
    gss_bin = GroupShuffleSplit(n_splits=1, train_size=0.8, random_state=42)
    train_idx_bin, val_idx_bin = next(gss_bin.split(df_binary, groups=df_binary['group_id']))
    
    train_loader_bin = DataLoader(SkinDataset(df_binary.iloc[train_idx_bin], train_transforms), batch_size=BATCH_SIZE, shuffle=True)
    val_loader_bin = DataLoader(SkinDataset(df_binary.iloc[val_idx_bin], val_transforms), batch_size=BATCH_SIZE, shuffle=False)
    
    model_bin = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model_bin.fc = nn.Linear(model_bin.fc.in_features, 2)
    train_model(model_bin, train_loader_bin, val_loader_bin, "model_binary_external.pth")
    
    # --- MODELO 2: MULTICLASSE ---
    # (O Modelo 2 é idêntico ao anterior, mas vamos salvar como external para manter a independência de arquivos)
    df_multiclass = df_lesao.copy()
    df_multiclass['label'] = df_multiclass['diagnostic'].map(DIAGNOSTIC_MAP)
    df_multiclass = df_multiclass.dropna(subset=['label'])
    
    gss_multi = GroupShuffleSplit(n_splits=1, train_size=0.8, random_state=42)
    train_idx_multi, val_idx_multi = next(gss_multi.split(df_multiclass, groups=df_multiclass['patient_id']))
    
    train_loader_multi = DataLoader(SkinDataset(df_multiclass.iloc[train_idx_multi], train_transforms), batch_size=BATCH_SIZE, shuffle=True)
    val_loader_multi = DataLoader(SkinDataset(df_multiclass.iloc[val_idx_multi], val_transforms), batch_size=BATCH_SIZE, shuffle=False)
    
    model_multi = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model_multi.fc = nn.Linear(model_multi.fc.in_features, 6)
    train_model(model_multi, train_loader_multi, val_loader_multi, "model_multiclass_external.pth")


if __name__ == "__main__":
    main()