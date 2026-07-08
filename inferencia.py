import os
import sys

import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms


DEVICE = torch.device("cpu")
if torch.cuda.is_available():
    try:
        # Testa se CUDA esta realmente operacional executando um kernel simples
        test_tensor = torch.zeros(1).to("cuda")
        test_tensor.add_(1)
        DEVICE = torch.device("cuda")
    except Exception:
        pass
BINARY_MODEL_PATH = "model_binary_external.pth"  # Mantido para retrocompatibilidade caso necessário
MODEL_PATH = "model_7class.pth"

REV_DIAGNOSTIC_MAP = {
    0: "BCC",
    1: "SCC",
    2: "ACK",
    3: "SEK",
    4: "NEV",
    5: "MEL",
    6: "SEM",
}

INFERENCE_TRANSFORM = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


def make_resnet18(num_classes):
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def load_state_dict(path):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Modelo '{path}' nao encontrado. Treine a solucao antes de executar a inferencia."
        )
    return torch.load(path, map_location=DEVICE)


def load_model():
    model = make_resnet18(7)
    model.load_state_dict(load_state_dict(MODEL_PATH))
    model.to(DEVICE)
    model.eval()
    return model


def predict_image(image_path, model):
    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as exc:
        print(f"Aviso: erro ao ler '{image_path}': {exc}. Predicao definida como SEM.", file=sys.stderr)
        return "SEM"

    with torch.no_grad():
        tensor = INFERENCE_TRANSFORM(image).unsqueeze(0).to(DEVICE)
        outputs = model(tensor)
        pred = outputs.argmax(1).item()
        return REV_DIAGNOSTIC_MAP.get(pred, "SEM")


def main():
    input_csv = "entrada.csv"
    output_csv = "resultado.csv"

    if not os.path.exists(input_csv):
        print(f"Erro: arquivo '{input_csv}' nao encontrado.", file=sys.stderr)
        sys.exit(1)

    df_input = pd.read_csv(input_csv)
    required_columns = {"id", "path"}
    if not required_columns.issubset(df_input.columns):
        print("Erro: entrada.csv deve conter as colunas 'id' e 'path'.", file=sys.stderr)
        sys.exit(1)

    try:
        model = load_model()
    except Exception as exc:
        print(f"Erro ao carregar modelo: {exc}", file=sys.stderr)
        sys.exit(1)

    results = []
    for _, row in df_input.iterrows():
        prediction = predict_image(str(row["path"]), model)
        results.append({"id": row["id"], "predicao": prediction})

    pd.DataFrame(results).to_csv(output_csv, index=False)
    print(f"Inferencia concluida. Resultado salvo em '{output_csv}'.")


if __name__ == "__main__":
    main()
