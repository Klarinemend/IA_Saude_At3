import torch
import torch.nn as nn
import pandas as pd
from PIL import Image
from torchvision import models, transforms

def run():
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Carregar Modelo Binário (Filtro)
    ckpt_bin = torch.load('modelo_binario.pth', map_location=DEVICE)
    model_bin = models.resnet18()
    model_bin.fc = nn.Linear(model_bin.fc.in_features, 2)
    model_bin.load_state_dict(ckpt_bin['state'])
    model_bin.to(DEVICE).eval()
    bin_classes = ckpt_bin['classes'] # Ex: ['lesao', 'sem_lesao']

    # 2. Carregar Modelo Multiclasse (Doenças)
    ckpt_multi = torch.load('modelo_multiclasse.pth', map_location=DEVICE)
    model_multi = models.resnet18()
    model_multi.fc = nn.Linear(model_multi.fc.in_features, 6)
    model_multi.load_state_dict(ckpt_multi['state'])
    model_multi.to(DEVICE).eval()
    multi_classes = ckpt_multi['classes']

    preprocess = transforms.Compose([
        transforms.Resize(256), transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    df_in = pd.read_csv('entrada.csv')
    results = []

    for _, row in df_in.iterrows():
        img = Image.open(row['path']).convert('RGB')
        tensor = preprocess(img).unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            # ESTÁGIO 1: É lesão?
            out_bin = model_bin(tensor)
            _, pred_bin = torch.max(out_bin, 1)
            classe_bin = bin_classes[pred_bin.item()]

            if classe_bin == 'sem_lesao':
                final_pred = "SEM"
            else:
                # ESTÁGIO 2: Qual lesão?
                out_multi = model_multi(tensor)
                _, pred_multi = torch.max(out_multi, 1)
                final_pred = multi_classes[pred_multi.item()].upper()
            
            results.append({'id': row['id'], 'predicao': final_pred})

    pd.DataFrame(results).to_csv('resultado.csv', index=False)
    print("Inferencia concluída com Pipeline de 2 estágios!")

if __name__ == '__main__':
    run()
