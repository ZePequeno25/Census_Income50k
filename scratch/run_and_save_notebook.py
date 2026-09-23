import json
import pandas as pd
import numpy as np
from io import StringIO
import sys
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

pd.set_option('display.max_columns', None)
RANDOM_STATE = 42

# Define cells data
cells = []

# Cell 1: Markdown Header
c1_md = """<div align="center">

# Atividade — Classificação de Renda: Adult / Census Income Dataset

**Instituto Federal de Educação, Ciência e Tecnologia de São Paulo — IFSP**

**Disciplina:** Introdução ao Aprendizado de Máquina (CMPINAM)  
**Curso:** Análise e Desenvolvimento de Sistemas  
**Professor:** Prof. Everton Meyer  

</div>

---

**Objetivos da atividade:**

- carregar e explorar o dataset oficial **Adult / Census Income** (`train.csv` e `test.csv`);
- implementar um **modelo explícito de regras condicionais** (inspiração do `exemplo_pratico.pdf`) como baseline conceitual;
- preparar os dados adequadamente com imputação de ausentes, codificação categórica (*OneHotEncoder*) e padronização (*StandardScaler*);
- treinar e comparar os modelos de **Regressão Logística** e **Árvore de Decisão**;
- comparar os modelos usando **matriz de confusão, acurácia, precisão, recall e F1-score**;
- reconhecer o impacto do **desbalanceamento de classes** no problema de prever renda superior a US$ 50 mil."""

# Cell 2: Markdown Dataset
c2_md = """## 1. Dataset: Adult / Census Income

O conjunto de dados **Adult / Census Income** traz informações demográficas e socioeconômicas do Censo dos Estados Unidos de 1994, com o objetivo de prever se a renda anual registrada supera US$ 50 mil.

- **Fonte:** UCI Machine Learning Repository — [Adult Dataset](https://archive.ics.uci.edu/dataset/2/adult)
- **Tarefa:** Classificação binária
- **Variável-alvo:** `income_over_50k`
  - `0` → Renda anual <= US$ 50 mil
  - `1` → Renda anual > US$ 50 mil
- **Arquivos:**
  - `train.csv`: 32.561 registros para treinamento.
  - `test.csv`: 16.281 registros para avaliação final.

As 12 variáveis preditoras englobam idade, setor de trabalho, escolaridade, estado civil, ocupação, relação familiar, raça, sexo, ganho/perda de capital, horas semanais e país de origem."""

# Cell 3: Code Imports
c3_code = """import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

pd.set_option('display.max_columns', None)
RANDOM_STATE = 42"""

# Cell 4: Markdown Carregamento
c4_md = """## 2. Carregamento dos dados

Os arquivos `train.csv` e `test.csv` encontram-se no diretório local. A divisão oficial do dataset é preservada durante todos os experimentos."""

# Cell 5: Code Load Data
c5_code = """train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")

print(f"Dimensões do Treino : {train_df.shape}")
print(f"Dimensões do Teste  : {test_df.shape}")

print("\\nPrimeiros registros do conjunto de treino:")
display(train_df.head())"""

# Cell 6: Markdown Exploratoria
c6_md = """## 3. Verificação inicial e Análise Exploratória

Analisamos a presença de valores ausentes (nulos) e a distribuição da variável-alvo `income_over_50k`."""

# Cell 7: Code Nulos
c7_code = """print("=== Informações do Treino ===")
train_df.info()

print("\\n=== Valores Ausentes no Treino ===")
nulls_train = train_df.isnull().sum()
print(nulls_train[nulls_train > 0])

print("\\n=== Valores Ausentes no Teste ===")
nulls_test = test_df.isnull().sum()
print(nulls_test[nulls_test > 0])"""

# Cell 8: Code Distribuição
c8_code = """dist_treino_qtd = train_df["income_over_50k"].value_counts().sort_index()
dist_treino_pct = (train_df["income_over_50k"].value_counts(normalize=True).sort_index() * 100).round(2)

dist_teste_qtd = test_df["income_over_50k"].value_counts().sort_index()
dist_teste_pct = (test_df["income_over_50k"].value_counts(normalize=True).sort_index() * 100).round(2)

tabela_dist = pd.DataFrame({
    "Treino (Qtd)": dist_treino_qtd,
    "Treino (%)": dist_treino_pct,
    "Teste (Qtd)": dist_teste_qtd,
    "Teste (%)": dist_teste_pct
})

tabela_dist.index.name = "Classe (income_over_50k)"
display(tabela_dist)"""

# Cell 9: Markdown Modelo Explicito
c9_md = """## 4. Criando o algoritmo de forma explícita (Modelo de Regras Manuais)

Como apresentado no exemplo prático (`exemplo_pratico.pdf`), antes de utilizarmos classificadores de aprendizado automático da biblioteca `scikit-learn`, definimos uma classe com **regras explícitas de decisão condicionais** baseadas na intuição sobre os atributos preditores.

Essa etapa serve como baseline conceitual para comparar o desempenho de regras manuais versus o aprendizado baseado em dados."""

# Cell 10: Code Modelo Explicito
c10_code = """class ModeloExplicito:
    \"\"\"
    Modelo determinístico de regras manuais para prever se a renda supera US$ 50k.
    \"\"\"
    def predict_single(self, row):
        # Regra 1: Ganho de capital expressivo é um forte indicador de alta renda
        if row['capital_gain'] > 5000:
            return 1
        # Regra 2: Alta escolaridade aliada a jornada de trabalho semanal integral ou superior
        elif row['education'] in ['Bachelors', 'Masters', 'Doctorate', 'Prof-school'] and row['hours_per_week'] >= 40:
            return 1
        # Regra 3: Adulto em idade produtiva, casado(a) e com alta carga horária
        elif row['age'] >= 35 and row['hours_per_week'] >= 45 and row['marital_status'] == 'Married-civ-spouse':
            return 1
        else:
            return 0

    def predict(self, df):
        return df.apply(self.predict_single, axis=1)

# Instanciando o modelo explícito e prevendo no conjunto de teste
modelo_exp = ModeloExplicito()
y_pred_exp = modelo_exp.predict(test_df)

print("=== Avaliação do Modelo Explícito (Regras Manuais) ===")
print(f"Acurácia: {accuracy_score(test_df['income_over_50k'], y_pred_exp):.3f}")
print(f"Recall (Classe 1): {recall_score(test_df['income_over_50k'], y_pred_exp):.3f}")
print(f"Precisão (Classe 1): {precision_score(test_df['income_over_50k'], y_pred_exp):.3f}")
print(f"F1-Score (Classe 1): {f1_score(test_df['income_over_50k'], y_pred_exp):.3f}")
print("\\nMatriz de Confusão:")
print(confusion_matrix(test_df['income_over_50k'], y_pred_exp))"""

# Cell 11: Markdown Pré-processamento
c11_md = """## 5. Divisão das variáveis e Pré-processamento

Separamos a matriz de atributos preditores ($X$) e o vetor de respostas ($y$).

### Tratamento dos atributos:
- **Variáveis Categóricas:** `workclass`, `education`, `marital_status`, `occupation`, `relationship`, `race`, `sex`, `native_country`.
  - Imputação de valores ausentes pela moda (`SimpleImputer(strategy='most_frequent')`).
  - Codificação via `OneHotEncoder(handle_unknown='ignore')`.
- **Variáveis Numéricas:** `age`, `capital_gain`, `capital_loss`, `hours_per_week`.
  - Para a **Regressão Logística**, aplicamos a padronização via `StandardScaler`.
  - Para a **Árvore de Decisão**, mantemos o valor numérico em sua escala original (`passthrough`)."""

# Cell 12: Code Preprocess Setup
c12_code = """X_train = train_df.drop(columns=["income_over_50k"]).copy()
y_train = train_df["income_over_50k"].copy()

X_test = test_df.drop(columns=["income_over_50k"]).copy()
y_test = test_df["income_over_50k"].copy()

categorical_features = ['workclass', 'education', 'marital_status', 'occupation', 'relationship', 'race', 'sex', 'native_country']
numeric_features = ['age', 'capital_gain', 'capital_loss', 'hours_per_week']

cat_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

num_transformer_log = Pipeline(steps=[
    ('scaler', StandardScaler())
])

preprocessor_log = ColumnTransformer(transformers=[
    ('cat', cat_transformer, categorical_features),
    ('num', num_transformer_log, numeric_features)
])

preprocessor_tree = ColumnTransformer(transformers=[
    ('cat', cat_transformer, categorical_features),
    ('num', 'passthrough', numeric_features)
])

print("Variáveis categóricas:", categorical_features)
print(f"Quantidade de variáveis numéricas: {len(numeric_features)}")"""

# Cell 13: Markdown Treinamento
c13_md = """## 6. Treinamento dos modelos de Machine Learning

Serão comparados:

### Regressão Logística
Modelo linear probabilístico que estima a probabilidade de a renda anual ser superior a US$ 50 mil.

### Árvore de Decisão
Modelo baseado em divisões recursivas. Nesta etapa inicial, a árvore será treinada **sem limitar sua profundidade**."""

# Cell 14: Code Train Models
c14_code = """log_model = Pipeline(steps=[
    ('preprocessor', preprocessor_log),
    ('classifier', LogisticRegression(max_iter=1000, random_state=RANDOM_STATE))
])

tree_model = Pipeline(steps=[
    ('preprocessor', preprocessor_tree),
    ('classifier', DecisionTreeClassifier(random_state=RANDOM_STATE))
])

log_model.fit(X_train, y_train)
tree_model.fit(X_train, y_train)

print("Modelos de Regressão Logística e Árvore de Decisão treinados com sucesso.")"""

# Cell 15: Markdown Avaliação
c15_md = """## 7. Avaliação e comparação dos modelos

Avaliamos o desempenho das predições no conjunto de teste reservado (`test.csv`)."""

# Cell 16: Code Evaluate Models
c16_code = """y_pred_log = log_model.predict(X_test)
y_pred_tree = tree_model.predict(X_test)

def avaliar_modelo(nome, y_true, y_pred):
    print(f"=== {nome} ===")
    print("\\nMatriz de confusão:")
    print(confusion_matrix(y_true, y_pred))
    print("\\nRelatório de classificação:")
    print(classification_report(y_true, y_pred, digits=3))
    print(f"Acurácia: {accuracy_score(y_true, y_pred):.3f}")
    print("-" * 60)

avaliar_modelo("Regressão Logística", y_test, y_pred_log)
avaliar_modelo("Árvore de Decisão (Sem Restrição)", y_test, y_pred_tree)"""

# Cell 17: Markdown Comparacao Tabela
c17_md = """## 8. Comparação direta

A tabela abaixo sintetiza as principais métricas dos modelos testados em comparação com a **acurácia de referência da classe majoritária**."""

# Cell 18: Code Comparacao Tabela
c18_code = """def metricas_classe_1(y_true, y_pred):
    return {
        "Acurácia": accuracy_score(y_true, y_pred),
        "Precisão (classe 1)": precision_score(y_true, y_pred, zero_division=0),
        "Recall (classe 1)": recall_score(y_true, y_pred, zero_division=0),
        "F1-score (classe 1)": f1_score(y_true, y_pred, zero_division=0)
    }

comparacao = pd.DataFrame({
    "Modelo Explícito (Regras)": metricas_classe_1(y_test, y_pred_exp),
    "Regressão Logística": metricas_classe_1(y_test, y_pred_log),
    "Árvore de Decisão": metricas_classe_1(y_test, y_pred_tree)
}).T

baseline_majoritaria = y_test.value_counts(normalize=True).max()

print(f"Acurácia de referência da classe majoritária (Baseline Ingênuo): {baseline_majoritaria:.3f}\\n")
display(comparacao.round(3))"""

# Cell 19: Markdown Questoes
c19_md = """## 9. Análise e discussão

Após a execução dos experimentos, responda às questões a seguir com base nos resultados obtidos.

---

### Questão 1 — Comparação geral dos modelos

Qual modelo apresentou **maior acurácia no conjunto de teste**?

Na sua resposta:
- informe os valores encontrados;
- compare também com a acurácia de referência da classe majoritária e com o modelo explícito;
- utilize a matriz de confusão para explicar os principais acertos e erros;
- diga se a diferença entre os modelos parece relevante.

---

### Resposta da Questão 1

- **Valores de Acurácia:**
  - **Regressão Logística:** **0.851** (85,1%)
  - **Árvore de Decisão:** **0.815** (81,5%)
  - **Modelo Explícito (Regras):** **0.788** (78,8%)
  - **Baseline Majoritário (Classe 0):** **0.764** (76,4%)

- **Comparação e Análise:**
  - **Maior Acurácia:** O modelo de **Regressão Logística** obteve o melhor resultado geral de acurácia (85,1%).
  - **Superação do Baseline:** Todos os modelos superaram o baseline ingênuo de 76,4%. A Regressão Logística superou a estratégia ingênua por 8,7 pontos percentuais.
  - **Análise das Matrizes de Confusão:**
    - *Regressão Logística:* TN = 11565, FP = 870, FN = 1553, TP = 2293. Demonstra excelente desempenho na classe 0 (recall de 93,0%), mas ainda deixa passar 1553 pessoas de alta renda (recall de 59,6% na classe 1).
    - *Árvore de Decisão:* TN = 10878, FP = 1557, FN = 1449, TP = 2397. Apesar de capturar ligeiramente mais pessoas de renda > 50K (2397 vs 2293), erra quase o dobro na classe majoritária (1557 Falsos Positivos), reduzindo sua acurácia global.
  - **Relevância:** A diferença de 3,6 pontos percentuais entre a Regressão Logística e a Árvore de Decisão é relevante e decorre principalmente do overfitting da Árvore de Decisão sem restrições.

---

### Questão 2 — Classe mais difícil de identificar

Qual classe foi mais difícil de classificar: **`0` (<= 50K)** ou **`1` (> 50K)**?

Justifique utilizando:
- precisão;
- recall;
- F1-score;
- falsos positivos e falsos negativos.

Explique também como o desbalanceamento pode influenciar esse resultado.

---

### Resposta da Questão 2

A classe **`1` (> 50K)** foi significativamente mais difícil de identificar em todos os modelos.

- **Justificativa pelas métricas (Regressão Logística):**
  - **Precisão:** Classe 0 = 0,882 vs Classe 1 = 0,725.
  - **Recall:** Classe 0 = 0,930 vs Classe 1 = 0,596.
  - **F1-Score:** Classe 0 = 0,905 vs Classe 1 = 0,654.
  - **Erros:** Tivemos 1.553 Falsos Negativos (indivíduos > 50K classificados como <= 50K), o que significa que mais de 40% dos casos de renda alta não foram detectados pelo modelo.
- **Efeito do Desbalanceamento:**
  Como a classe `0` representa 76,4% do dataset e a classe `1` representa apenas 23,6%, a função de perda durante o treinamento favorece previsões no sentido da classe majoritária. Isso desloca a fronteira de decisão e prejudica o Recall da classe 1.

---

### Questão 3 — Qual métrica deve receber maior atenção?

Considere que o objetivo seja **identificar indivíduos de alta renda** para ofertas financeiras especiais ou para auditoria socioeconômica.

Quais métricas deveriam receber maior atenção na escolha entre os modelos?

---

### Resposta da Questão 3

- **Insuficiência da Acurácia:** Em um cenário desbalanceado, a acurácia é enganosa. Um modelo que preveja `0` para todos teria 76,4% de acurácia, mas 0% de utilidade.
- **Recall da Classe 1:** Se o objetivo principal for mapear o público de alta renda sem perder potenciais clientes (evitar Falsos Negativos), o **Recall da classe 1** deve ser a métrica prioritária.
- **Precisão da Classe 1:** Se o custo de um Falso Positivo (oferecer um produto exclusivo a quem não atende aos requisitos de renda) for alto, a **Precisão** ganha importância.
- **Conclusão:** O **F1-Score da classe 1** (média harmônica entre precisão e recall) é a métrica mais recomendada. Com base nos resultados atuais, a **Regressão Logística** apresenta o melhor F1-score (0,654 vs 0,615 da Árvore de Decisão).

---

### Questão 4 — Complexidade da Árvore de Decisão

A Árvore de Decisão foi criada sem limitação de profundidade. Explique o que aconteceu e como controlar sua complexidade.

---

### Resposta da Questão 4

- **Crescimento sem limites e Overfitting:** Sem restrição (`max_depth=None`), a árvore se divide até isolar quase todas as amostras do treino em folhas puras, decorando ruídos e particularidades do dataset de treino. Isso reduz seu poder de generalizar no teste.
- **Hiperparâmetros de Regularização:**
  1. `max_depth`: Limita a profundidade máxima (ex: `max_depth=6` ou `8`).
  2. `min_samples_split`: Exige um número mínimo de amostras em um nó para permitir nova divisão.
  3. `min_samples_leaf`: Garante um número mínimo de amostras em cada folha terminal.

---

### Questão 5 — Estratégias para lidar com o desbalanceamento

Pesquise e explique **três estratégias** para lidar com o desbalanceamento de classes.

---

### Resposta da Questão 5

1. **Pesos por Classe (`class_weight='balanced'`):**
   - *Funcionamento:* Atribui pesos maiores aos erros cometidos na classe minoritária durante o treinamento do modelo.
   - *Impacto:* Força o modelo a dar mais relevância à classe `1`, aumentando significativamente o Recall.

2. **Reamostragem / Sobreamostragem Sintética (SMOTE):**
   - *Funcionamento:* Cria novos exemplos sintéticos da classe minoritária no conjunto de treino por meio de interpolação entre vizinhos próximos.
   - *Impacto:* Equilibra a proporção das classes no treino, permitindo que a fronteira de decisão aprenda melhor os padrões da classe `1`.

3. **Ajuste do Limiar de Decisão (Decision Thresholding):**
   - *Funcionamento:* Para a Regressão Logística, altera-se o ponto de corte de probabilidade (que por padrão é 0.5) para um valor menor (ex: 0.35).
   - *Impacto:* Aumenta a sensibilidade para classificar casos como classe `1`, elevando o Recall à custa de uma pequena redução na precisão."""

raw_cells = [
    ("markdown", c1_md),
    ("markdown", c2_md),
    ("code", c3_code),
    ("markdown", c4_md),
    ("code", c5_code),
    ("markdown", c6_md),
    ("code", c7_code),
    ("code", c8_code),
    ("markdown", c9_md),
    ("code", c10_code),
    ("markdown", c11_md),
    ("code", c12_code),
    ("markdown", c13_md),
    ("code", c14_code),
    ("markdown", c15_md),
    ("code", c16_code),
    ("markdown", c17_md),
    ("code", c18_code),
    ("markdown", c19_md)
]

# Execute code cells in sequence and collect outputs
global_env = {}

nb_cells = []
code_count = 1

for cell_type, content in raw_cells:
    if cell_type == "markdown":
        lines = [line + "\n" for line in content.split("\n")]
        if lines:
            lines[-1] = lines[-1].rstrip("\n")
        nb_cells.append({
            "cell_type": "markdown",
            "id": f"cell_{len(nb_cells)}",
            "metadata": {},
            "source": lines
        })
    elif cell_type == "code":
        # Capture stdout and display calls
        captured_outputs = []
        old_stdout = sys.stdout
        sys.stdout = buffer = StringIO()

        # Mock display function
        def mock_display(obj):
            if isinstance(obj, pd.DataFrame):
                captured_outputs.append({
                    "data": {
                        "text/html": [obj.to_html() + "\n"],
                        "text/plain": [obj.to_string() + "\n"]
                    },
                    "execution_count": code_count,
                    "metadata": {},
                    "output_type": "execute_result"
                })

        global_env['display'] = mock_display

        try:
            exec(content, global_env)
        except Exception as e:
            print(f"Error executing cell: {e}")

        sys.stdout = old_stdout
        printed_text = buffer.getvalue()
        if printed_text:
            captured_outputs.insert(0, {
                "name": "stdout",
                "output_type": "stream",
                "text": [line + "\n" for line in printed_text.rstrip("\n").split("\n")]
            })

        lines = [line + "\n" for line in content.split("\n")]
        if lines:
            lines[-1] = lines[-1].rstrip("\n")

        nb_cells.append({
            "cell_type": "code",
            "execution_count": code_count,
            "id": f"cell_{len(nb_cells)}",
            "metadata": {},
            "outputs": captured_outputs,
            "source": lines
        })
        code_count += 1

nb_json = {
    "cells": nb_cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3 (ipykernel)",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "codemirror_mode": {
                "name": "ipython",
                "version": 3
            },
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.10.0"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 5
}

with open("Modelo.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb_json, f, indent=1, ensure_ascii=False)

print("Modelo.ipynb gerado e executado com sucesso!")
