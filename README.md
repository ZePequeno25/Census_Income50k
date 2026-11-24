# Quem Ganha Mais de US\$ 50 Mil por Ano?
### Um Projeto de Inteligência Artificial Explicado para Todo Mundo

> **Projeto Acadêmico — CMPINAM (Introdução ao Aprendizado de Máquina)**  
> **Instituição:** IFSP — Câmpus Campinas (Análise e Desenvolvimento de Sistemas)  
> **Professor:** Prof. Everton Meyer da Silva  
> **Aluno:** Kelvin Aparecido Lima Albuquerque  
> **Semestre:** 2º semestre de 2026

---

## 1. Explicando O que é este projeto?

Imagine uma grande caixa cheia de **fichas de uma pesquisa antiga do governo dos Estados Unidos** (o Censo de 1994). Em cada ficha, está anotado um pouco da vida de um adulto:
- Quantos anos ele tem;
- Até que série ou faculdade estudou;
- Qual é a sua profissão;
- Quantas horas trabalha por semana;
- Se é casado ou solteiro;
- Se teve algum ganho extra com investimentos.

O nosso objetivo aqui foi **ensinar o computador a ler essas fichas e dar um palpite bem inteligente**:

> *"Olhando para essas informações, será que essa pessoa ganha **mais de 50 mil dólares por ano** ou **50 mil ou menos**?"*

Em vez de testar apenas um jeito de ensinar o computador, nós testamos **8 métodos diferentes** — dos mais simples aos mais modernos, incluindo um "mini cérebro artificial" (Rede Neural) — para descobrir qual deles acerta mais e qual faz mais sentido usar na vida real!

---

## 2. Como preparamos as fichas (Sem trapaça!)

Antes de ensinar qualquer coisa ao computador, fizemos o que toda boa cozinheira faz antes de começar uma receita: **organizamos a cozinha**.

1. **Tiramos fichas repetidas:** Das 32.561 fichas de estudo, removemos as cópias idênticas para o computador não decorar respostas repetidas (ficando com 29.096 fichas únicas).
2. **Envelope lacrado para a prova final:** O conjunto de teste oficial (**16.281 fichas**) ficou guardado em um "envelope fechado". O computador **nunca pôde espiar** essas fichas durante o aprendizado. Só abrimos o envelope no finalzinho para dar a nota justa de cada método.
3. **Colocando tudo na mesma régua (Padronização):**  
   Imagine comparar a idade de alguém (ex.: `40 anos`) com o lucro da venda de ações (ex.: `99.999 dólares`). Para alguns algoritmos que fazem contas de distância, o número gigante faz a idade parecer "invisível". Por isso, usamos uma régua matemática (`StandardScaler`) que coloca todas as informações em uma escala justa antes de treinar os modelos que precisam disso.
4. **Escolhendo as pistas mais importantes:** Descobrimos quais são as **8 pistas que mais ajudam** a prever a renda: *ganho de capital, estado civil, papel na família, escolaridade, profissão, horas trabalhadas por semana, idade e sexo*.

---

## 3. Conhecendo os 8 "Alunos" (Os Modelos que Testamos)

Para facilitar o entendimento, pense em cada modelo de Inteligência Artificial como um jeito diferente de pensar para resolver o mesmo problema:

| Modelo (Nome Técnico) | Apelido Amigável | Como ele pensa? |
| :--- | :--- | :--- |
| **Baseline (Majoritário)** | 😴 *O Chute Preguiçoso* | Como 76% das pessoas na pesquisa ganham ≤ 50k, ele simplesmente chuta "ganha menos" para todo mundo sem nem ler a ficha. Já acerta **76,4%** só na preguiça! É a nossa régua mínima para vencer. |
| **Regras Explícitas** | 📋 *A Receita de Bolo* | Regras fixas escritas por um humano na mão (ex.: *"se teve ganho de capital alto ou tem faculdade e trabalha muito, então ganha > 50k"*). |
| **Regressão Logística** *(Aula 06)* | ⚖️ *A Balança de Pontos* | Coloca cada característica numa balança: ter faculdade ou ser casado soma pontos; trabalhar poucas horas tira pontos. No fim, dá uma chance exata em porcentagem. É o modelo **mais transparente e fácil de explicar**. |
| **Árvore de Decisão** *(Aula 07)* | 🌳 *O Jogo de 20 Perguntas* | Faz perguntas de "Sim ou Não" em sequência (tipo *"É casado?" → "Tem diploma?" → "Tem mais de 35 anos?"*) até chegar à resposta. |
| **KNN** *(Aula 08)* | 🏘️ *Diga-me com Quem Andas* | Procura as **15 pessoas mais parecidas** com a ficha nova e decide pelo voto da maioria desses "vizinhos". *(Exige a régua justa/padronização!)* |
| **Random Forest** *(Aula 09)* | 🌲 *O Conselho de 150 Árvores* | Em vez de confiar em uma única árvore (que pode se enganar), cria **150 árvores diferentes** e faz uma votação entre elas. Não se atrapalha com números gigantes! |
| **SVM** *(Aula 11)* | 🚧 *A Cerca Mais Segura* | Tenta desenhar uma fronteira separando os dois grupos de modo a deixar a **maior folga possível** de cada lado, reduzindo chances de confusão. |
| **Rede Neural (MLP)** *(Aula 12)* | 🧠 *O Pequeno Cérebro* | Inspira-se nos neurônios humanos! Passa os dados por uma camada oculta (16 neurônios) que aprende combinações complexas, mede o próprio erro e vai ajustando seus **801 parafusinhos internos** (pesos) ao longo de 97 rodadas de estudo até ficar afiada. |

---

## 4. O Boletim Final: Quem se saiu melhor na prova?

Antes de olhar as notas, um dicionário rápido de 30 segundos para entender as colunas da tabela:
- **Acurácia:** De cada 100 pessoas, quantas o modelo acertou no total.
- **Precisão:** Quando o modelo aponta o dedo e diz *"essa pessoa ganha mais de 50 mil"*, quantas vezes ele estava certo?
- **Recall (Sensibilidade):** De todas as pessoas que *realmente* ganham mais de 50 mil, quantas o modelo conseguiu encontrar sem deixar escapar?
- **F1-Score:** Uma nota de equilíbrio entre **Precisão** e **Recall**.
- **ROC-AUC:** Uma nota de 0 a 1 que mede o poder geral do modelo de separar quem ganha mais de quem ganha menos (quanto mais perto de 1,000, melhor).

### Resultados oficiais nas 16.281 fichas da prova final (`test.csv`):

| Modelo | Acurácia | Precisão (>50k) | Recall (>50k) | F1-Score (>50k) | ROC-AUC | PR-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline (Chute Preguiçoso)** | 76,4% | 0,0% | 0,0% | 0,000 | — | — |
| **Modelo Explícito (Regras)** | 72,0% | 45,2% | **86,7%** | 0,594 | — | — |
| **Regressão Logística** | 80,1% | 55,2% | 84,4% | 0,667 | 0,899 | 0,741 |
| **Árvore de Decisão (Podada)** | 84,9% | 74,2% | 55,1% | 0,632 | 0,887 | 0,720 |
| **KNN (15 vizinhos)** | 83,5% | 67,1% | 59,3% | 0,629 | 0,881 | 0,687 |
| **SVM (Linear)** | 84,7% | 73,0% | 56,2% | 0,635 | 0,899 | 0,744 |
| **🌲 Random Forest (Aula 09)** | 85,5% | 73,8% | 59,6% | 0,660 | 0,908 | 0,776 |
| **🧠 Rede Neural (MLP - Aula 12)** | 85,2% | 71,0% | 63,0% | 0,668 | 0,906 | 0,764 |
| **🚀 Gradient Boosting (Método Extra)** | **86,3%** | **75,0%** | 63,2% | **0,686** | **0,917** | **0,796** |

---

## 🏆 5. Veredito: Qual modelo escolher na vida real?

Não existe um único "melhor para tudo" — a escolha depende do que você mais valoriza:

1. **🥇 Campeã Geral e Mais Prática: Random Forest (Floresta Aleatória)**
   - Teve a **maior taxa de acertos geral (85,5%)** e a **maior precisão (75,5%)** — ou seja, quando diz que alguém ganha bem, acerta 3 em cada 4 vezes!
   - É muito resistente: não se confunde com valores extremos (como quem ganhou 99.999 dólares na bolsa), treina muito rápido e dá respostas instantâneas.

2. **🥈 Campeã em Equilíbrio (F1-Score): Rede Neural Artificial (MLP)**
   - Alcançou a **melhor nota de equilíbrio (F1 = 0,668)** e empatou no topo com a Random Forest no **ROC-AUC (0,906)**.
   - Consegue encontrar mais pessoas de alta renda (63,0% de Recall contra 56,9% da Random Forest), mas tem um lado "caixa-preta" (é difícil explicar por que ela tomou uma decisão específica) e exige obrigatoriamente a padronização dos dados.

3. **🥉 Campeã em Transparência: Regressão Logística**
   - Se um banco ou órgão público precisar explicar na justiça exatamente *por que* o computador deu aquela resposta, a Regressão Logística vence. Com o **Odds Ratio** ($ODS = e^\beta$), conseguimos dizer claramente o peso de cada fator.

---

## ⚠️ 6. Cuidados Éticos e Limitações (Muito Importante!)

- **Os dados são de 1994:** O mundo mudou muito desde então; esses padrões históricos não servem para tomar decisões reais sobre pessoas hoje.
- **Atenção ao preconceito nos dados:** O computador apenas aprende o que existe nos dados antigos. Como dados de 1994 refletiam desigualdades históricas ligadas a gênero e raça, usar modelos assim sem auditoria ética na vida real pode perpetuar injustiças. Este projeto tem finalidade **estritamente educacional**.

---

## 🛠️ 7. Seção Técnica (Para Professores e Desenvolvedores)

### O que você encontra no Notebook (`Modelo.ipynb`)
- **49 células executadas** de ponta a ponta com reprodutibilidade garantida (`random_state = 42`).
- **EDA aprofundada** voltada às exigências matemáticas de cada família de algoritmos (distâncias $L_2$, maldição da dimensionalidade, invariância monotônica e estabilidade de gradiente).
- **Regressão Linear** (Simples, Múltipla, Polinomial Grau 2, Ridge e Lasso) como linha de investigação inicial.
- **Validação Cruzada Estratificada (5 folds)** e `GridSearchCV` sem vazamento de dados.
- **Inspeção interna da MLP (Aula 12):** número de épocas (`n_iter_ = 97`), dimensões de matrizes de pesos `[(48, 16), (16, 1)]`, vetores de viés `[(16,), (1,)]`, total de 801 parâmetros treináveis, ativação de saída (`logistic`) e gráfico completo da curva de perda (`loss_curve_`).
- **Visualizações consolidadas:** gráficos de barras comparativos, curvas ROC sobrepostas e painel de matrizes de confusão.

### Estrutura de Pastas

```
CMPINAM_Dataset_05_Adult_Census_Income/
├── Prediction_in_Adult/
│   └── Census_Income50k/
│       ├── Modelo.ipynb               # Notebook principal completo e executado
│       ├── train.csv                  # Dados de treino (UCI)
│       ├── test.csv                   # Dados de teste oficial (envelope lacrado)
│       ├── README.md                  # Esta documentação
│       ├── figures/                   # Gráficos gerados
│       └── results/
│           └── Modelo_executado.ipynb # Cópia espelho executada
├── scratch/                           # Scripts auxiliares de construção e validação
└── README.md                          # Documentação na raiz do repositório
```

### Como Rodar no Seu Computador

1. Tenha o **Python 3.10+** instalado com as bibliotecas básicas de ciência de dados:
   ```bash
   pip install pandas numpy scikit-learn matplotlib
   ```
2. Entre na pasta do projeto e abra o notebook:
   ```bash
   cd Prediction_in_Adult/Census_Income50k
   jupyter notebook Modelo.ipynb
   ```
3. Clique em **Run All** (*Executar Tudo*) para reproduzir todas as análises, gráficos e métricas do zero.

---

## 📚 8. Referências

- Becker, B. & Kohavi, R. (1996). *Adult Dataset*. UCI Machine Learning Repository. [https://doi.org/10.24432/C5XW20](https://doi.org/10.24432/C5XW20).
- Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*. MIT Press (Capítulo 6 — MLP e Backpropagation).
- Materiais didáticos da disciplina **CMPINAM** (Prof. Everton Meyer da Silva — IFSP Câmpus Campinas): Aulas 06 (Regressão Logística), 07 (Árvores de Decisão), 08 (KNN), 09 (Random Forest), 11 (SVM) e 12 (Redes Neurais Artificiais).

---

## 🌐 9. Links do Projeto (GitHub e GitHub Pages)

- **Repositório no GitHub:** https://github.com/ZePequeno25/Census_Income50k
- **Apresentação do Seminário (GitHub Pages):** https://zepequeno25.github.io/Census_Income50k/apresentacao_seminario.html
- **Documento Visual — Método Extra (Gradient Boosting):** https://zepequeno25.github.io/Census_Income50k/docs/CMPINAM_extra_gradient_boosting.html
- **Página Inicial (GitHub Pages):** https://zepequeno25.github.io/Census_Income50k/
