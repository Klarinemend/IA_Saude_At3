import os
import sys
import torch
import torch.nn as nn
from torchvision import transforms, models
import pandas as pd
from PIL import Image

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

REV_DIAGNOSTIC_MAP = {
    0: 'BCC', 1: 'SCC', 2: 'ACK', 3: 'SEK', 4: 'NEV', 5: 'MEL'
}

infer_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def load_models():
    model_bin = models.resnet18()
    model_bin.fc = nn.Linear(model_bin.fc.in_features, 2)
    if os.path.exists("model_binary_external.pth"):
        model_bin.load_state_dict(torch.load("model_binary_external.pth", map_location=DEVICE))
    else:
        print("Aviso: 'model_binary_external.pth' não encontrado.", file=sys.stderr)
    model_bin = model_bin.to(DEVICE)
    model_bin.eval()
    
    model_multi = models.resnet18()
    model_multi.fc = nn.Linear(model_multi.fc.in_features, 6)
    if os.path.exists("model_multiclass_external.pth"):
        model_multi.load_state_dict(torch.load("model_multiclass_external.pth", map_location=DEVICE))
    else:
        print("Aviso: 'model_multiclass_external.pth' não encontrado.", file=sys.stderr)
    model_multi = model_multi.to(DEVICE)
    model_multi.eval()
    
    return model_bin, model_multi


def predict_image(image_path, model_bin, model_multi):
    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"Erro ao ler imagem {image_path}: {e}", file=sys.stderr)
        return "SEM"
        
    image_tensor = infer_transforms(image).unsqueeze(0).to(DEVICE)
    
    with torch.no_grad():
        outputs_bin = model_bin(image_tensor)
        _, pred_bin = outputs_bin.max(1)
        
        if pred_bin.item() == 0:
            return "SEM"
        else:
            outputs_multi = model_multi(image_tensor)
            _, pred_multi = outputs_multi.max(1)
            return REV_DIAGNOSTIC_MAP.get(pred_multi.item(), "SEM")


def main():
    input_csv = "entrada.csv"
    output_csv = "resultado.csv"
    
    if not os.path.exists(input_csv):
        print(f"Erro: {input_csv} não encontrado.", file=sys.stderr)
        sys.exit(1)
        
    df_input = pd.read_csv(input_csv)
    
    if 'id' not in df_input.columns or 'path' not in df_input.columns:
        print("Erro: Entrada deve conter as colunas 'id' e 'path'.", file=sys.stderr)
        sys.exit(1)
        
    model_bin, model_multi = load_models()
    
    results = []
    for _, row in df_input.iterrows():
        img_id = row['id']
        img_path = row['path']
        
        pred_label = predict_image(img_path, model_bin, model_multi)
        results.append({'id': img_id, 'predicao': pred_label})
        
    df_output = pd.DataFrame(results)
    df_output.to_csv(output_csv, index=False)
    print(f"Inferência terminada. Resultados salvos em '{output_csv}'.")


if __name__ == "__main__":
    main()