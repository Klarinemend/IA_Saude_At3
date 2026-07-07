import pandas as pd
import os
import shutil
import zipfile
import random
from PIL import Image

# --- CONFIGURAÇÕES CORRIGIDAS ---
CSV_PATH = 'metadata.csv' 
# Adicionei 'data/' antes do nome das pastas conforme seu print
PART_FOLDERS = ['data/imgs_part_1', 'data/imgs_part_2', 'data/imgs_part_3'] 
TEXTURE_ZIP = 'Skin Texture.zip' 
OUTPUT_DIR = 'dataset_maximo'
MAX_SEM_IMAGES = 700 

def organizar_maximo():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # --- PARTE 1: PAD-UFES ---
    if os.path.exists(CSV_PATH):
        df = pd.read_csv(CSV_PATH)
        classes = df['diagnostic'].unique()
        print(f"Organizando TODAS as imagens do PAD-UFES...")

        for classe in classes:
            classe_upper = classe.upper()
            os.makedirs(os.path.join(OUTPUT_DIR, classe_upper), exist_ok=True)
            df_classe = df[df['diagnostic'] == classe]
            
            count = 0
            for _, row in df_classe.iterrows():
                img_nome = row['img_id']
                foi_encontrada = False
                for part in PART_FOLDERS:
                    origem = os.path.join(part, img_nome)
                    if os.path.exists(origem):
                        shutil.copy(origem, os.path.join(OUTPUT_DIR, classe_upper, img_nome))
                        count += 1
                        foi_encontrada = True
                        break
            print(f"Classe {classe_upper}: {count} imagens copiadas.")
    else:
        print(f"ERRO: Arquivo '{CSV_PATH}' não encontrado na raiz!")

    # --- PARTE 2: CLASSE SEM ---
    print(f"\n--- Processando Classe SEM ---")
    sem_dir = os.path.join(OUTPUT_DIR, 'SEM')
    os.makedirs(sem_dir, exist_ok=True)
    
    temp_extract = 'temp_skin'
    if os.path.exists(temp_extract): shutil.rmtree(temp_extract)
    
    if os.path.exists(TEXTURE_ZIP):
        with zipfile.ZipFile(TEXTURE_ZIP, 'r') as zip_ref:
            zip_ref.extractall(temp_extract)
        
        todos_arquivos_sem = []
        for root, _, files in os.walk(temp_extract):
            for f in files:
                if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')):
                    todos_arquivos_sem.append(os.path.join(root, f))
        
        if len(todos_arquivos_sem) > 0:
            qtd = min(MAX_SEM_IMAGES, len(todos_arquivos_sem))
            amostras_sem = random.sample(todos_arquivos_sem, qtd)
            
            for i, path_origem in enumerate(amostras_sem):
                try:
                    img = Image.open(path_origem).convert('RGB')
                    img.save(os.path.join(sem_dir, f"sem_{i}.png"))
                except:
                    pass
            print(f"Sucesso! {len(os.listdir(sem_dir))} imagens de pele saudável preparadas.")
    
    if os.path.exists(temp_extract): shutil.rmtree(temp_extract)
    print(f"\nDataset final pronto em: {OUTPUT_DIR}")

if __name__ == '__main__':
    organizar_maximo()
