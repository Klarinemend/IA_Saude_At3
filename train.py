import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt
import os
from collections import Counter # Já vem no Python, não precisa instalar!

# --- CONFIGURAÇÕES ---
DATA_DIR = 'dataset_maximo'  # Pasta com as imagens reais
BATCH_SIZE = 32
EPOCHS = 30 # Aumentei para 30 para aproveitar o dataset maior
LR = 0.0001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ApplyTransform(torch.utils.data.Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform
    def __getitem__(self, index):
        x, y = self.subset[index]
        if self.transform: x = self.transform(x)
        return x, y
    def __len__(self):
        return len(self.subset)

def train():
    if not os.path.exists(DATA_DIR):
        print(f"ERRO: A pasta '{DATA_DIR}' não existe!")
        return

    # 1. TRANSFORMAÇÕES
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(180),
        transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    val_tf = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # 2. CARREGAMENTO
    full_ds = datasets.ImageFolder(DATA_DIR)
    class_names = full_ds.classes
    num_data = len(full_ds)
    
    indices = torch.randperm(num_data).tolist()
    split = int(0.8 * num_data)
    
    train_ds = ApplyTransform(Subset(full_ds, indices[:split]), transform=train_tf)
    val_ds = ApplyTransform(Subset(full_ds, indices[split:]), transform=val_tf)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    # --- LÓGICA DE PESOS PARA BALANCEAMENTO ---
    # Contamos quantas imagens tem em cada classe no dataset de treino
    labels_treino = [full_ds.targets[i] for i in indices[:split]]
    contagem = Counter(labels_treino)
    
    # Criamos o peso: quanto menos imagens, maior o peso do erro
    # Peso = Total / (Qtd_Classes * Qtd_da_Classe)
    pesos = []
    for i in range(len(class_names)):
        qtd = contagem[i]
        peso = len(labels_treino) / (len(class_names) * qtd)
        pesos.append(peso)
    
    pesos_tensor = torch.FloatTensor(pesos).to(DEVICE)
    print(f"Classes: {class_names}")
    print(f"Pesos aplicados: {pesos}")

    # 3. MODELO
    model = models.resnet18(weights='IMAGENET1K_V1')
    model.fc = nn.Linear(model.fc.in_features, len(class_names))
    model = model.to(DEVICE)

    # Aplicamos os pesos aqui no CrossEntropyLoss
    criterion = nn.CrossEntropyLoss(weight=pesos_tensor)
    
    # Adicionei weight_decay=1e-4 para ajudar contra o overfitting
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)

    history = {'t_loss': [], 'v_loss': [], 't_acc': [], 'v_acc': []}

    print("\nIniciando Treinamento com Balanceamento...")
    for epoch in range(EPOCHS):
        model.train()
        train_loss, train_corr = 0.0, 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            train_corr += torch.sum(preds == labels.data)

        model.eval()
        val_loss, val_corr = 0.0, 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                val_corr += torch.sum(preds == labels.data)

        history['t_loss'].append(train_loss / len(train_ds))
        history['v_loss'].append(val_loss / len(val_ds))
        history['t_acc'].append(train_corr.double().item() / len(train_ds))
        history['v_acc'].append(val_corr.double().item() / len(val_ds))

        print(f"Epoca {epoch+1}/{EPOCHS} | Train Acc: {history['t_acc'][-1]:.3f} | Val Acc: {history['v_acc'][-1]:.3f}")

    torch.save({'state': model.state_dict(), 'classes': class_names}, 'modelo_skin.pth')

    # Gerar Gráficos
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history['t_loss'], label='Treino')
    plt.plot(history['v_loss'], label='Validação')
    plt.title('Loss (Erro) - Com Pesos')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(history['t_acc'], label='Treino')
    plt.plot(history['v_acc'], label='Validação')
    plt.title('Acurácia - Com Pesos')
    plt.legend()
    plt.savefig('graficos_resultado.png')
    print("\nTreino finalizado com sucesso!")

if __name__ == '__main__':
    train()
