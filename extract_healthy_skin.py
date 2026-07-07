import os
import numpy as np
from PIL import Image

def extract_corners(image_path, output_dir, img_id, crop_size=224, min_size=800):
    try:
        img = Image.open(image_path)
    except Exception as e:
        print(f"Erro ao abrir {image_path}: {e}")
        return 0

    w, h = img.size

    # Regra de segurança estrita: ignora imagens menores que 800x800 pixels
    if w < min_size or h < min_size:
        return 0

    # Coordenadas dos 4 cantos extremos para evitar a lesão centralizada
    corners = {
        "top_left": (0, 0, crop_size, crop_size),
        "top_right": (w - crop_size, 0, w, crop_size),
        "bottom_left": (0, h - crop_size, crop_size, h),
        "bottom_right": (w - crop_size, h - crop_size, w, h)
    }

    saved_count = 0
    for name, box in corners.items():
        cropped = img.crop(box)
        img_np = np.array(cropped)
        
        # Filtro 1: Descarta áreas excessivamente escuras (falhas de lente ou vinheta)
        if np.mean(img_np) < 35:
            continue
            
        # Filtro 2: Descarta áreas sólidas homogêneas (como réguas ou papéis adesivos brancos nas bordas)
        if np.std(img_np) < 8:
            continue

        # Nome preserva o ID da imagem para podermos associar ao paciente depois
        out_name = f"healthy_{img_id}_{name}.png"
        out_path = os.path.join(output_dir, out_name)
        
        cropped.save(out_path)
        saved_count += 1

    return saved_count


def main():
    images_dir = "images/"      # Diretório com as imagens originais do PAD-UFES
    output_dir = "data_sem/"    # Pasta de destino da classe SEM (Sem Lesão)
    
    os.makedirs(output_dir, exist_ok=True)
    
    all_images = [f for f in os.listdir(images_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    print(f"Total de imagens originais: {len(all_images)}")
    
    total_extracted = 0
    for idx, filename in enumerate(all_images):
        img_id = os.path.splitext(filename)[0]
        full_path = os.path.join(images_dir, filename)
        
        extracted = extract_corners(full_path, output_dir, img_id)
        total_extracted += extracted
        
        if (idx + 1) % 500 == 0 or (idx + 1) == len(all_images):
            print(f"Processando: {idx + 1}/{len(all_images)} | Extraídos {total_extracted} patches de pele saudável.")

    print(f"\nExtração concluída!")
    print(f"Confira os resultados na pasta: '{output_dir}'")


if __name__ == "__main__":
    main()