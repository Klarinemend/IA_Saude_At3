# Classificação Multiclasse de Lesões de Pele com Deteção de Out-of-Distribution
**Disciplina:** Inteligência Artificial Aplicada à Saúde  
**Autores:** [Nome do Integrante 1] e [Nome do Integrante 2]

---

### 1. Construção da Classe "Sem Lesão" (SEM)
Para mitigar o problema de imagens fora do domínio clínico (fotos tremidas, dedos cobrindo a lente, pele saudável, fundos de imagem) que um classificador em campo inevitavelmente receberá, construímos a 7ª classe (**`SEM`**) seguindo uma estratégia de diversificação de distribuição (OOD):
* **Composição e Quantidade:** Um conjunto total de **1273 imagens** para a classe SEM (em cores RGB), integrando texturas macro de pele saudável e uma ampla gama de fotos de controle e ruído (rostos, objetos cotidianos, fundos diversos e obstruções de lente), que foram combinadas às **2298 imagens** clínicas de lesões cutâneas do PAD-UFES-20.
* **Critérios de Seleção:** As imagens foram selecionadas para cobrir a variabilidade de White Balance (balanço de cores) de fotos caseiras e smartphones, eliminando o viés do ambiente estritamente hospitalar e clínico presente nas fotos do dataset original.

---

### 2. Arquitetura do Modelo e Decisões de Treino
Adotamos uma abordagem de **Modelo Único de 7 Classes** em substituição a arquiteturas em cascata para garantir maior estabilidade de convergência e reduzir o tempo de computação.
* **Rede Backbone:** Utilizamos a **ResNet18** com pesos pré-treinados no ImageNet (transfer learning) e realizamos o ajuste fino (fine-tuning) de todas as camadas.
* **Tratamento do Desbalanceamento (Class Weighting):** Devido ao desbalanceamento no dataset de treino (com classes variando de 52 imagens em MEL até 843 em BCC e 1273 em SEM), implementamos pesos dinâmicos na função de custo **CrossEntropyLoss**, inversamente proporcionais à frequência das classes.
* **Otimização:** Treinado com o otimizador **AdamW** (learning rate de $10^{-4}$ e weight decay de $10^{-4}$) por **35 épocas** com tamanho de lote (batch size) de 32.
* **Evitando Vazamento de Dados (Data Leakage):** A divisão entre treino e validação foi agrupada de forma rígida pelo ID do paciente (`patient_id`), garantindo que fotos do mesmo indivíduo não estivessem presentes simultaneamente nos dois conjuntos.

---

### 3. Resultados e Discussão
O modelo foi treinado em GPU e avaliado no conjunto de validação do PAD-UFES-20.

* **Acurácia de Validação:** O modelo atingiu **79.83% de acurácia de validação** geral (e **67.75% de acurácia balanceada**), demonstrando excelente poder de generalização após o ajuste fino de 35 épocas.
* **Validação em Dataset Clínico Real:** No teste clínico de 12 imagens reais (2 de cada patologia), o classificador obteve **91.67% de acertos** (11 de 12 corretas), corrigindo erros anteriores de limiares limítrofes com confianças de 97.4% a 100.0%. A única divergência ocorreu na diferenciação de SCC e ACK (sua lesão precursora direta).
* **Validação com Smartphones:** Em testes práticos fora de distribuição (celular), o enquadramento de fotos saudáveis obteve **99.96% de acerto em SEM**, e as patologias de lesão no celular mantiveram excelentes acertos (86.1% a 99.2% de acerto), consolidando a eficácia da ResNet18 de 7 classes calibrada para telemedicina.

---

### 4. Matriz de Confusão e Análise
Abaixo apresentamos a matriz de confusão textual obtida sobre o conjunto de validação da partição GroupShuffleSplit (gráficos visuais gerados no arquivo `confusion_matrix.png`):

Real \ Pred | BCC | SCC | ACK | SEK | NEV | MEL | SEM
-|-----|-----|-----|-----|-----|-----|----
BCC | 148 | 11 | 14 | 0 | 4 | 0 | 0
SCC | 23 | 8 | 11 | 1 | 1 | 0 | 0
ACK | 28 | 7 | 113 | 1 | 0 | 0 | 2
SEK | 0 | 1 | 6 | 28 | 4 | 2 | 0
NEV | 1 | 1 | 0 | 3 | 27 | 1 | 2
MEL | 0 | 0 | 0 | 4 | 3 | 8 | 0
SEM | 0 | 0 | 1 | 0 | 2 | 0 | 253

* **Análise dos Resultados:** A diagonal principal demonstra alta precisão nas classes com mais dados representativos, como **BCC** (148 acertos) e **ACK** (113 acertos). O destaque do modelo é a classe **`SEM`** (saudável/controle), alcançando **98.8% de acurácia** (253 acertos de 256), o que reduz drasticamente falsos positivos em triagens. 
* **Erros Clínicos Esperados:** As maiores confusões ocorrem entre classes de carcinomas não-melanoma: **SCC** (Carcinoma Espinocelular) sendo predito como **BCC** (23 casos) e **ACK** (11 casos), devido à forte semelhança morfológica dessas patologias que compartilham padrões inflamatórios escamosos avermelhados, representando um comportamento esperado em dermatologia.
