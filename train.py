import argparse
import os
import random
import zipfile

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DIAGNOSTIC_MAP = {"BCC": 0, "SCC": 1, "ACK": 2, "SEK": 3, "NEV": 4, "MEL": 5, "SEM": 6}
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".tif", ".tiff")


def seed_everything(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_transforms():
    lesion_train = transforms.Compose(
        [
            transforms.RandomResizedCrop(224, scale=(0.75, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.12, contrast=0.12, saturation=0.08),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    lesion_val = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return lesion_train, lesion_val


def maybe_extract_zip(zip_path, target_dir):
    if os.path.isdir(target_dir):
        return
    if not os.path.exists(zip_path):
        return
    print(f"Extraindo {zip_path}...")
    with zipfile.ZipFile(zip_path, "r") as archive:
        archive.extractall(".")


def get_external_group_id(filename_or_path):
    base = os.path.basename(filename_or_path)
    if "-_jpg" in base:
        return base.split("-_jpg")[0]
    return os.path.splitext(base)[0]


def list_images(folder):
    return [
        os.path.join(folder, name)
        for name in os.listdir(folder)
        if name.lower().endswith(IMAGE_EXTENSIONS)
    ]


class SkinDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = self.df.loc[idx, "img_path"]
        label = int(self.df.loc[idx, "label"])

        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.long)


def train_one_epoch(model, dataloader, criterion, optimizer):
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    for images, labels in dataloader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        correct += outputs.argmax(1).eq(labels).sum().item()
        total += labels.size(0)

    return running_loss / total, correct / total


def evaluate(model, dataloader, criterion):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            correct += outputs.argmax(1).eq(labels).sum().item()
            total += labels.size(0)

    return running_loss / total, correct / total


def make_resnet18(num_classes, pretrained=True, freeze_backbone=False):
    weights = models.ResNet18_Weights.DEFAULT if pretrained else None
    model = models.resnet18(weights=weights)
    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    for param in model.fc.parameters():
        param.requires_grad = True
    return model


def train_model(model, train_loader, val_loader, save_name, epochs, learning_rate, class_weights=None):
    model = model.to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(trainable_params, lr=learning_rate, weight_decay=1e-4)

    best_acc = 0.0
    print(f"\n--- Treinando {save_name} em {DEVICE} ---")

    for epoch in range(epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc = evaluate(model, val_loader, criterion)

        print(
            f"Epoca {epoch + 1}/{epochs} | "
            f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}"
        )

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), save_name)
            print(f"=> Melhor modelo salvo: {save_name} | val_acc={val_acc:.4f}")

    return best_acc


def calibrate_threshold(model, val_loader):
    model.eval()
    all_probs = []
    all_labels = []

    print("\nAvaliando conjunto de validação para calibração de limiar...")
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(DEVICE)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            all_probs.append(probs.cpu())
            all_labels.append(labels)

    all_probs = torch.cat(all_probs, dim=0)
    all_labels = torch.cat(all_labels, dim=0).numpy()

    best_threshold = 0.50
    best_score = 0.0

    # Testa limiares de 0.40 a 0.98 com passo 0.01
    thresholds = [i / 100.0 for i in range(40, 99)]

    for t in thresholds:
        preds = []
        for prob in all_probs:
            max_prob, pred_idx = prob.max(0)
            pred_class_idx = pred_idx.item()
            conf = max_prob.item()

            # Se for classificado como lesão (0-5) mas a confiança for menor que t,
            # reclassifica como SEM (6)
            if pred_class_idx != 6 and conf < t:
                preds.append(6)
            else:
                preds.append(pred_class_idx)

        score = balanced_accuracy_score(all_labels, preds)
        if score > best_score:
            best_score = score
            best_threshold = t

    print(f"\n=======================================================")
    print(f"=> CALIBRAÇÃO DE LIMIAR CONCLUÍDA")
    print(f"=> Limiar ótimo recomendado (threshold): {best_threshold:.2f}")
    print(f"=> Acurácia Balanceada obtida na validação: {best_score * 100:.2f}%")
    print(f"=======================================================\n")
    return best_threshold


def load_pad_dataframe(metadata_path, images_dir):
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Arquivo nao encontrado: {metadata_path}")
    if not os.path.isdir(images_dir):
        raise FileNotFoundError(
            f"Pasta '{images_dir}' nao encontrada. Coloque nela as imagens do PAD-UFES-20."
        )

    df_pad = pd.read_csv(metadata_path)
    required_columns = {"img_id", "patient_id", "diagnostic"}
    missing_columns = required_columns.difference(df_pad.columns)
    if missing_columns:
        raise ValueError(f"Colunas ausentes no metadata: {sorted(missing_columns)}")

    df_pad["img_path"] = df_pad["img_id"].apply(lambda name: os.path.join(images_dir, str(name)))
    df_pad = df_pad[df_pad["diagnostic"].isin(DIAGNOSTIC_MAP)].copy()
    df_pad = df_pad[df_pad["img_path"].apply(os.path.exists)].copy()

    if df_pad.empty:
        raise ValueError("Nenhuma imagem do PAD-UFES-20 foi encontrada com base no metadata.csv.")

    return df_pad[["img_path", "patient_id", "diagnostic"]]


def load_sem_dataframe(sem_dir, sem_zip):
    maybe_extract_zip(sem_zip, sem_dir)
    if not os.path.isdir(sem_dir):
        raise FileNotFoundError(
            f"Pasta '{sem_dir}' nao encontrada. Extraia as imagens SEM ou mantenha {sem_zip} na raiz."
        )

    sem_images = list_images(sem_dir)
    if not sem_images:
        raise ValueError(f"Nenhuma imagem SEM encontrada em '{sem_dir}'.")

    df_sem = pd.DataFrame({"img_path": sem_images})
    df_sem["diagnostic"] = "SEM"
    df_sem["group_id"] = df_sem["img_path"].apply(get_external_group_id)
    return df_sem


def split_by_group(df, group_column, seed):
    splitter = GroupShuffleSplit(n_splits=1, train_size=0.8, random_state=seed)
    train_idx, val_idx = next(splitter.split(df, groups=df[group_column]))
    return df.iloc[train_idx].copy(), df.iloc[val_idx].copy()


def main():
    parser = argparse.ArgumentParser(description="Treina modelos para PAD-UFES-20 + classe SEM.")
    parser.add_argument("--metadata", default="metadata.csv")
    parser.add_argument("--images-dir", default="images")
    parser.add_argument("--sem-dir", default="images_SEM")
    parser.add_argument("--sem-zip", default="")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--freeze-backbone", action="store_true", help="Congela os pesos do backbone ResNet18 para acelerar o treino em CPU")
    parser.add_argument("--device", default=None, help="Dispositivo para treino (cpu ou cuda). Se None, tenta usar cuda.")
    args = parser.parse_args()

    global DEVICE
    if args.device:
        DEVICE = torch.device(args.device)
    elif torch.cuda.is_available():
        try:
            # Testa se o CUDA está realmente operacional executando um kernel simples (soma)
            test_tensor = torch.zeros(1).to("cuda")
            test_tensor.add_(1)
            DEVICE = torch.device("cuda")
        except Exception as e:
            print("\nAviso: CUDA está disponível mas não é compatível com o hardware/PyTorch instalados (ex: RTX 5070 sm_120).")
            print("Forçando treinamento em CPU. Dica: Use '--freeze-backbone' para treinar rapidamente.")
            DEVICE = torch.device("cpu")
    else:
        DEVICE = torch.device("cpu")

    seed_everything(args.seed)
    lesion_train, lesion_val = build_transforms()

    df_lesion = load_pad_dataframe(args.metadata, args.images_dir)
    df_lesion["group_id"] = df_lesion["patient_id"]
    df_lesion["label"] = df_lesion["diagnostic"].map(DIAGNOSTIC_MAP).astype(int)

    df_sem = load_sem_dataframe(args.sem_dir, args.sem_zip)
    df_sem["label"] = 6

    print(f"Imagens PAD-UFES encontradas: {len(df_lesion)}")
    print(f"Imagens SEM encontradas: {len(df_sem)}")

    # Unifica todos os dados em um único dataframe de 7 classes
    df_all = pd.concat([df_lesion, df_sem], ignore_index=True)

    # Separação de treino/val baseada em grupo (patient_id para lesões e nome base para SEM)
    train_df, val_df = split_by_group(df_all, "group_id", args.seed)

    # Calcula frequência inversa das classes para balancear perda (CrossEntropyLoss)
    counts = [0] * 7
    for label, count in train_df["label"].value_counts().items():
        counts[int(label)] = count
    counts = [max(c, 1) for c in counts]
    class_weights = [1.0 / c for c in counts]
    sum_weights = sum(class_weights)
    class_weights = [w / sum_weights * 7.0 for w in class_weights]
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(DEVICE)
    print(f"Pesos de classe calculados para CrossEntropyLoss: {[round(w, 4) for w in class_weights]}")

    train_loader = DataLoader(
        SkinDataset(train_df, lesion_train),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        SkinDataset(val_df, lesion_val),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    model = make_resnet18(num_classes=7, pretrained=not args.no_pretrained, freeze_backbone=args.freeze_backbone)
    train_model(
        model,
        train_loader,
        val_loader,
        "model_7class.pth",
        args.epochs,
        args.learning_rate,
        class_weights=class_weights_tensor,
    )

    # Carrega o melhor modelo salvo e calibra o limiar na validação
    model.load_state_dict(torch.load("model_7class.pth", map_location=DEVICE))
    calibrate_threshold(model, val_loader)


if __name__ == "__main__":
    main()
