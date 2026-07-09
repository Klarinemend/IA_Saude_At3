import torch
import torch.nn as nn
from torchvision import transforms, models
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
from PIL import Image
import os
import sys
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GroupShuffleSplit

# Configurações do Dispositivo
DEVICE = torch.device("cpu")
if torch.cuda.is_available():
    try:
        test_tensor = torch.zeros(1).to("cuda")
        test_tensor.add_(1)
        DEVICE = torch.device("cuda")
    except Exception:
        pass

MODEL_PATH = "model_7class.pth"
METADATA_PATH = "metadata.csv"
IMAGES_DIR = "images"
SEM_DIR = "images_SEM"
SEED = 42

DIAGNOSTIC_MAP = {"BCC": 0, "SCC": 1, "ACK": 2, "SEK": 3, "NEV": 4, "MEL": 5, "SEM": 6}
CLASSES = ['BCC', 'SCC', 'ACK', 'SEK', 'NEV', 'MEL', 'SEM']

# Dataset e Transforms idênticos ao treino
class SkinDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_path = row["img_path"]
        image = Image.open(image_path).convert("RGB")
        label = int(row["label"])
        if self.transform:
            image = self.transform(image)
        return image, label

def load_pad_dataframe(metadata_path, images_dir):
    df_pad = pd.read_csv(metadata_path)
    df_pad = df_pad[df_pad["diagnostic"].isin(DIAGNOSTIC_MAP)].copy()
    df_pad["img_path"] = df_pad["img_id"].apply(lambda x: os.path.join(images_dir, x))
    df_pad = df_pad[df_pad["img_path"].apply(os.path.exists)].copy()
    return df_pad[["img_path", "patient_id", "diagnostic"]]

def get_external_group_id(path):
    return os.path.splitext(os.path.basename(path))[0]

def load_sem_dataframe(sem_dir):
    df_sem = pd.DataFrame(columns=["img_path", "group_id", "diagnostic"])
    if os.path.exists(sem_dir):
        files = [os.path.join(sem_dir, f) for f in os.listdir(sem_dir)
                 if f.lower().endswith((".png", ".jpg", ".jpeg", ".tif", ".tiff"))]
        df_sem = pd.DataFrame({
            "img_path": files,
            "diagnostic": "SEM"
        })
        df_sem["group_id"] = df_sem["img_path"].apply(get_external_group_id)
    return df_sem

def split_by_group(df, group_col, seed):
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    train_idx, val_idx = next(gss.split(df, groups=df[group_col]))
    return df.iloc[train_idx].copy(), df.iloc[val_idx].copy()

def make_resnet18(num_classes):
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

def main():
    if not os.path.exists(MODEL_PATH):
        print(f"Erro: Arquivo do modelo '{MODEL_PATH}' não encontrado. Treine o modelo primeiro.")
        sys.exit(1)

    print("Carregando dados de validação...")
    df_lesion = load_pad_dataframe(METADATA_PATH, IMAGES_DIR)
    df_lesion["group_id"] = df_lesion["patient_id"]
    df_lesion["label"] = df_lesion["diagnostic"].map(DIAGNOSTIC_MAP).astype(int)

    df_sem = load_sem_dataframe(SEM_DIR)
    df_sem["label"] = 6

    df_all = pd.concat([df_lesion, df_sem], ignore_index=True)
    _, val_df = split_by_group(df_all, "group_id", SEED)

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    val_loader = DataLoader(
        SkinDataset(val_df, val_transform),
        batch_size=32,
        shuffle=False,
        num_workers=0
    )

    print("Carregando modelo...")
    model = make_resnet18(7)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

    y_true = []
    y_pred = []

    print("Calculando predições...")
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(DEVICE)
            outputs = model(images)
            preds = outputs.argmax(dim=1).cpu().numpy()
            y_true.extend(labels.numpy())
            y_pred.extend(preds)

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Gera a matriz de confusão
    cm = confusion_matrix(y_true, y_pred, labels=range(7))

    # 1. Imprime a Matriz de Confusão em Markdown
    print("\n--- MATRIZ DE CONFUSÃO EM FORMATO MARKDOWN (COPIAR PARA RELATORIO.MD) ---")
    header = " | ".join(["Real \\ Pred"] + CLASSES)
    print(header)
    print("-|-" + "-|-".join(["---"] * 7))
    for i, row in enumerate(cm):
        row_str = " | ".join([CLASSES[i]] + [str(x) for x in row])
        print(row_str)

    # 2. Tenta plotar graficamente usando matplotlib e seaborn
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns

        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=CLASSES, yticklabels=CLASSES)
        plt.title('Matriz de Confusão (Conjunto de Validação)')
        plt.ylabel('Classe Real')
        plt.xlabel('Classe Predita')
        plt.tight_layout()
        
        output_image = "confusion_matrix.png"
        plt.savefig(output_image, dpi=300)
        print(f"\n=> Gráfico salvo com sucesso em '{output_image}'!")
    except ImportError:
        print("\nAviso: matplotlib ou seaborn não estão instalados.")
        print("Para gerar o gráfico visual, instale-os com: pip install matplotlib seaborn")

if __name__ == "__main__":
    main()
