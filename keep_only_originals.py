import os
import sys

def main():
    # --- CONFIRME O NOME DA SUA PASTA DE IMAGENS EXTERNAS ---
    folder = "data_sem_external"

    if not os.path.exists(folder):
        print(f"Erro: A pasta '{folder}' não foi encontrada na raiz do projeto.", file=sys.stderr)
        sys.exit(1)

    # Lista todos os arquivos de imagem na pasta
    files = [f for f in os.listdir(folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    print(f"Total de arquivos encontrados originalmente: {len(files)}")

    # Agrupa as imagens pelo ID base (ex: "normal_-121")
    groups = {}
    for f in files:
        if "-_jpg" in f:
            base_id = f.split("-_jpg")[0]
        else:
            base_id = os.path.splitext(f)[0]
            
        if base_id not in groups:
            groups[base_id] = []
        groups[base_id].append(f)

    print(f"Total de peles saudáveis únicas detectadas: {len(groups)}")

    # Mantém apenas o primeiro arquivo de cada grupo e deleta o resto
    deleted_count = 0
    kept_count = 0

    print("\nIniciando a limpeza dos arquivos rotacionados...")
    for base_id, file_list in groups.items():
        # Ordena a lista de arquivos para garantir consistência
        file_list.sort()
        
        # O primeiro arquivo ordenado é a imagem original limpa
        original_img = file_list[0]
        kept_count += 1
        
        # Deleta as rotações (do índice 1 em diante)
        for rotated_img in file_list[1:]:
            full_path = os.path.join(folder, rotated_img)
            try:
                os.remove(full_path)
                deleted_count += 1
            except Exception as e:
                print(f"Erro ao deletar {rotated_img}: {e}", file=sys.stderr)

    print(f"\nLimpeza concluída com sucesso!")
    print(f"Imagens originais sem rotação MANTIDAS: {kept_count}")
    print(f"Imagens rotacionadas/duplicadas DELETADAS: {deleted_count}")


if __name__ == "__main__":
    main()