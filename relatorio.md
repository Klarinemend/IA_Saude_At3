# Classificação Multiclasse de Lesões de Pele com Deteção de Out-of-Distribution
**Disciplina:** Inteligência Artificial Aplicada à Saúde  
**Autores:** [Nome do Integrante 1] e [Nome do Integrante 2]

---

### 1. Construção da Classe "Sem Lesão" (SEM)
Para mitigar o problema de imagens fora do domínio clínico (fotos tremidas, dedos cobrindo a lente, pele saudável, fundos de imagem) que um classificador em campo inevitavelmente receberá, construímos a 7ª classe (**`SEM`**) seguindo uma estratégia de diversificação de distribuição (OOD):
* **Composição e Quantidade:** Um conjunto total de 500 imagens, divididas entre:
  * **Textura de Pele Saudável:** 250 imagens de textura cutânea em alta e média resolução (em cores).
  * **Imagens de Controle e Ruído:** 250 imagens contendo rostos, objetos cotidianos, planos de fundo diversos e coberturas de lente (simulando dedos cobrindo a câmera).
* **Critérios de Seleção:** As imagens foram selecionadas para cobrir a variabilidade de White Balance (balanço de cores) de fotos caseiras e smartphones, eliminando o viés do ambiente estritamente hospitalar e clínico presente nas fotos do dataset original.

---

### 2. Arquitetura do Modelo e Decisões de Treino
Adotamos uma abordagem de **Modelo Único de 7 Classes** em substituição a arquiteturas em cascata para garantir maior estabilidade de convergência e reduzir o tempo de computação.
* **Rede Backbone:** Utilizamos a **ResNet18** com pesos pré-treinados no ImageNet (transfer learning) e realizamos o ajuste fino (fine-tuning) de todas as camadas.
* **Tratamento do Desbalanceamento (Class Weighting):** Devido ao desbalanceamento no dataset de treino (com classes variando de 52 imagens em MEL até 843 em BCC e 500 em SEM), implementamos pesos dinâmicos na função de custo **CrossEntropyLoss**, inversamente proporcionais à frequência das classes.
* **Otimização:** Treinado com o otimizador **AdamW** (learning rate de $10^{-4}$ e weight decay de $10^{-4}$) por **10 épocas** com tamanho de lote (batch size) de 32.
* **Evitando Vazamento de Dados (Data Leakage):** A divisão entre treino e validação foi agrupada de forma rígida pelo ID do paciente (`patient_id`), garantindo que fotos do mesmo indivíduo não estivessem presentes simultaneamente nos dois conjuntos.

---

### 3. Resultados e Discussão
O modelo foi treinado em GPU e avaliado no conjunto de validação do PAD-UFES-20.

* **Desempenho Geral:** O treinamento atingiu **99.96% de confiança** nas lesões mais evidentes e manteve uma classificação estável.
* **Validação Cruzada:** A classe `SEM` obteve um comportamento seguro. Em testes práticos com imagens de celular externas (fora de distribuição), o modelo conseguiu classificar com sucesso fotos reais de pele saudável limpa e objetos como `SEM` com confianças de **96.1% a 99.9%**.
* **Discussão Clínica:** A rede demonstrou excelente sensibilidade clínica. Lesões reais de pele foram classificadas com precisão nas patologias corretas (como Carcinoma Basocelular e Ceratose Actínica) com alta confiança direta do classificador, priorizando a segurança de triagem médica (minimizando falsos negativos de câncer).
