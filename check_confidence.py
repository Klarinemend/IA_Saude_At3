import torch
import pandas as pd
from PIL import Image
from torchvision import transforms, models
import torch.nn as nn
import os

# Configuração do dispositivo
DEVICE = torch.device("cpu")
if torch.cuda.is_available():
    try:
        test_tensor = torch.zeros(1).to("cuda")
        test_tensor.add_(1)
        DEVICE = torch.device("cuda")
    except Exception:
        pass

print(f"Executando no dispositivo: {DEVICE}")

# Carrega o modelo de 7 classes
def make_resnet18(num_classes):
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

MODEL_PATH = "model_7class.pth"
model = make_resnet18(7)
if os.path.exists(MODEL_PATH):
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

# Transformação
INFERENCE_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

df = pd.read_csv("entrada.csv")
classes = ['BCC', 'SCC', 'ACK', 'SEK', 'NEV', 'MEL', 'SEM']

print("\nID   | Predição Final | Confiança | Detalhamento das Probabilidades por Classe")
print("-" * 110)

for _, row in df.iterrows():
    img_id = row['id']
    img_path = row['path']
    if not os.path.exists(img_path):
        print(f"{img_id:4s} | Erro: arquivo não encontrado em '{img_path}'")
        continue
    
    img = Image.open(img_path).convert("RGB")
    tensor = INFERENCE_TRANSFORM(img).unsqueeze(0).to(DEVICE)
    
    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.softmax(outputs, dim=1)[0]
        
    max_prob, pred_idx = probs.max(0)
    pred_class = classes[pred_idx.item()]
    conf = max_prob.item()
    
    probs_list = [f"{classes[i]}: {p.item()*100:.1f}%" for i, p in enumerate(probs)]
    print(f"{img_id:4s} | {pred_class:14s} | {conf*100:8.2f}% | {', '.join(probs_list)}")
