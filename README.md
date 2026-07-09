# Classificação de Lesões de Pele (PAD-UFES-20 + Classe SEM)

Este repositório contém uma solução baseada em uma Rede Neural Convolucional (ResNet18) para classificar imagens de lesões de pele em 7 categorias: BCC, SCC, ACK, SEK, NEV, MEL, e uma 7ª classe customizada **`SEM`** (para imagens sem lesão, fundos ou fotos inválidas).

A solução utiliza **apenas imagens** como entrada do modelo. Os metadados clínicos do PAD-UFES-20 são usados somente para a separação do conjunto de treinamento e validação por paciente (evitando vazamento de dados).

---

## Estrutura do Projeto

```text
metadata.csv
images/
images_SEM/
train.py
inferencia.py
requirements.txt
README.md
relatorio.md
```

* `images/`: Contém as imagens clínicas originais do PAD-UFES-20.
* `images_SEM/`: Contém imagens representativas de pele saudável, rostos, objetos e fundos diversos para compor a 7ª classe.

---

## Instalação das Dependências

Para instalar as dependências necessárias do projeto:

```powershell
pip install -r requirements.txt
```

---

## Treinamento do Modelo

O treinamento é feito a partir de uma única rede convolucional unificada com 7 classes, aplicando ponderação de perda para balanceamento de classes e transfer learning:

```powershell
python train.py --epochs 35 --batch-size 32
```

O treinamento gera o arquivo de pesos do modelo:
* `model_7class.pth`

---

## Execução da Inferência

Para executar a inferência de novas imagens:

1. Crie um arquivo `entrada.csv` na raiz do projeto com as colunas `id` e `path`:
   ```csv
   id,path
   img1,C:\caminho\para\imagem1.png
   img2,C:\caminho\para\imagem2.png
   ```

2. Execute o script de inferência:
   ```powershell
   python inferencia.py
   ```

3. O resultado final das predições será salvo na raiz do projeto no arquivo `resultado.csv`:
   ```csv
   id,predicao
   img1,SEK
   img2,SEM
   ```

---

## Contribuição dos Integrantes

* **Integrante 1 (Gustavo Sarti):** Planejamento da arquitetura de 7 classes, coleta e curadoria de imagens para a classe SEM, desenvolvimento dos códigos, calibração do threshold e testes locais com imagens.
* **Integrante 2 (Klarine Silva):** Planejamento da arquitetura de 7 classes, curadoria de imagens para a classe SEM, desenvolvimento dos códigos e elaboração do relatório técnico.
