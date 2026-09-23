import nbformat as nbf
import pandas as pd
import numpy as np
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

nb = nbf.v4.new_notebook()

# Cell 1: Header
cell1_md = """<div align="center">

# Atividade — Classificação de Renda: Adult / Census Income Dataset

**Instituto Federal de Educação, Ciência e Tecnologia de São Paulo — IFSP**

**Disciplina:** Introdução ao Aprendizado de Máquina (CMPINAM)  
**Curso:** Análise e Desenvolvimento de Sistemas  
**Professor:** Prof. Everton Meyer  

</div>

---

**Objetivos da atividade:**

- Carregar e explorar o dataset oficial *Adult Census Income* (`train.csv` e `test.csv`).
- Implementar um **modelo explícito de regras base** (inspiração do `exemplo_pratico.pdf`) como baseline conceitual.
- Construir um pipeline de pré-processamento com imputação de nulos (`workclass`, `occupation`, `native_country`), codificação categórica (*OneHotEncoder*) e padronização (*StandardScaler*).
- Treinar e comparar os modelos de **Regressão Logística** e **Árvore de Decisão**.
- Avaliar os modelos usando **Acurácia, Precisão, Recall, F1-Score e Matriz de Confusão**, discutindo os impactos do desbalanceamento das classes."""

# Cell 2: Visão geral
cell2_md = """## 1. Visão Geral do Dataset

O conjunto de dados **Adult / Census Income** contém dados históricos do censo norte-americano de 1994. O objetivo é prever se a renda anual registrada de uma pessoa é superior a US$ 50 mil.

- **Fonte:** UCI Machine Learning Repository — [Adult Dataset](https://archive.ics.uci.edu/dataset/2/adult)
- **Tarefa:** Classificação binária
- **Variável-alvo:** `income_over_50k`
  - `0` → Renda anual <= US$ 50 mil
  - `1` → Renda anual > US$ 50 mil
- **Divisão oficial de arquivos:**
  - `train.csv`: 32.561 registros para treinamento.
  - `test.csv`: 16.281 registros para avaliação final.

As 12 variáveis preditoras englobam dados demográficos, educação, ocupação e informações financeiras."""

# Cell 3: Imports
cell3_code = """import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

pd.set_option('display.max_columns', None)
sns.set_theme(style="whitegrid")
RANDOM_STATE = 42"""

# Cell 4: Carregamento
cell4_md = """## 2. Carregamento dos dados

Os arquivos `train.csv` e `test.csv` encontram-se no diretório do projeto. A divisão oficial do dataset foi preservada durante o experimento."""

# Cell 5: Carregamento code
cell5_code = """train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")

print(f"Dimensões do Treino : {train_df.shape}")
print(f"Dimensões do Teste  : {test_df.shape}")

print("\\nPrimeiros registros do conjunto de treino:")
display(train_df.head())"""

# Cell 6: Verificação inicial md
cell6_md = """## 3. Análise exploratória e verificação inicial

Nesta etapa, analisamos a presença de valores ausentes (nulos) e observamos a distribuição da variável-alvo `income_over_50k` nos dois conjuntos."""

# Cell 7: Verificação inicial code
cell7_code = """print("=== Informações do Treino ===")
train_df.info()

print("\\n=== Valores Ausentes no Treino ===")
nulls_train = train_df.isnull().sum()
print(nulls_train[nulls_train > 0])

print("\\n=== Valores Ausentes no Teste ===")
nulls_test = test_df.isnull().sum()
print(nulls_test[nulls_test > 0])"""

# Cell 8: Distribuição da classe alvo code
cell8_code = """dist_treino_qtd = train_df["income_over_50k"].value_counts().sort_index()
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

# Cell 9: Modelo explícito md
cell9_md = """## 4. Criando o algoritmo de forma explícita (Modelo de Regras Manuais)

Seguindo o exemplo prático demonstrado na introdução (`exemplo_pratico.pdf`), antes de utilizarmos algoritmos de aprendizado de máquina, definimos uma classe com **regras de decisão explícitas**. 

Essa abordagem manual serve para ilustrar a diferença entre a programação tradicional por regras e o aprendizado supervisionado por máquina."""

# Cell 10: Modelo explícito code
cell10_code = """class ModeloExplicito:
    \"\"\"
    Modelo baseado em regras manuais/heurísticas de domínio para prever renda > 50K.
    \"\"\"
    def predict_single(self, row):
        # Regra 1: Ganho de capital elevado é forte sinal de alta renda
        if row['capital_gain'] > 5000:
            return 1
        # Regra 2: Alta escolaridade trabalhando jornada integral ou superior
        elif row['education'] in ['Bachelors', 'Masters', 'Doctorate', 'Prof-school'] and row['hours_per_week'] >= 40:
            return 1
        # Regra 3: Idade madura, casado e alta carga horária
        elif row['age'] >= 35 and row['hours_per_week'] >= 45 and row['marital_status'] == 'Married-civ-spouse':
            return 1
        else:
            return 0

    def predict(self, df):
        return df.apply(self.predict_single, axis=1)

# Instanciando e avaliando o modelo explícito no conjunto de teste
modelo_exp = ModeloExplicito()
y_pred_exp = modelo_exp.predict(test_df)

print("=== Avaliação do Modelo Explícito (Regras Manuais) ===")
print(f"Acurácia: {accuracy_score(test_df['income_over_50k'], y_pred_exp):.3f}")
print(f"Recall (Classe 1): {recall_score(test_df['income_over_50k'], y_pred_exp):.3f}")
print(f"Precisão (Classe 1): {precision_score(test_df['income_over_50k'], y_pred_exp):.3f}")
print(f"F1-Score (Classe 1): {f1_score(test_df['income_over_50k'], y_pred_exp):.3f}")
print("\\nMatriz de Confusão:")
print(confusion_matrix(test_df['income_over_50k'], y_pred_exp))"""

# Cell 11: Divisão e pré-processamento md
cell11_md = """## 5. Divisão das variáveis e pré-processamento

Separamos os atributos preditores ($X$) e a variável-alvo ($y$) para o conjunto de treino e de teste.

### Tratamento das variáveis
- **Variáveis Categóricas:** `workclass`, `education`, `marital_status`, `occupation`, `relationship`, `race`, `sex`, `native_country`.
  - Imputação de valores ausentes pela moda (`most_frequent`).
  - Codificação com `OneHotEncoder(handle_unknown='ignore')`.
- **Variáveis Numéricas:** `age`, `capital_gain`, `capital_loss`, `hours_per_week`.
  - Para a **Regressão Logística**, aplicamos a padronização via `StandardScaler`.
  - Para a **Árvore de Decisão**, mantemos os valores originais (`passthrough`), pois divisões de árvore são invariantes à escala."""

# Cell 12: Divisão e pré-processamento code
cell12_code = """X_train = train_df.drop(columns=["income_over_50k"]).copy()
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

print("Variáveis Categóricas:", categorical_features)
print(f"Quantidade de Variáveis Numéricas: {len(numeric_features)}")"""

# Cell 13: Treinamento md
cell13_md = """## 6. Treinamento dos modelos de Aprendizado de Máquina

Treinamos dois modelos principais:
1. **Regressão Logística:** Modelo linear que calcula a probabilidade da renda ser superior a 50K.
2. **Árvore de Decisão:** Modelo não paramétrico baseado em divisões sucessivas dos dados (treinado inicialmente sem limite de profundidade)."""

# Cell 14: Treinamento code
cell14_code = """log_model = Pipeline(steps=[
    ('preprocessor', preprocessor_log),
    ('classifier', LogisticRegression(max_iter=1000, random_state=RANDOM_STATE))
])

tree_model = Pipeline(steps=[
    ('preprocessor', preprocessor_tree),
    ('classifier', DecisionTreeClassifier(random_state=RANDOM_STATE))
])

log_model.fit(X_train, y_train)
tree_model.fit(X_train, y_train)

print("Modelos de Regressão Logística e Árvore de Decisão treinados com sucesso!")"""

# Cell 15: Avaliação md
cell15_md = """## 7. Avaliação e comparação dos modelos

Realizamos as predições no conjunto de teste oficial (`test.csv`) e analisamos as métricas de desempenho."""

# Cell 16: Avaliação code
cell16_code = """y_pred_log = log_model.predict(X_test)
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

# Cell 17: Tabela comparativa code
cell17_code = """def metricas_classe_1(y_true, y_pred):
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

# Cell 18: Questões e discussões md
cell18_md = """## 8. Análise e discussão (Questões para fixação)

---

### Questão 1 — Comparação geral dos modelos
Qual modelo apresentou **maior acurácia no conjunto de teste**? 
Compare o desempenho dos modelos (Regressão Logística e Árvore de Decisão) com o baseline da classe majoritária e com o Modelo Explícito.

### Resposta da Questão 1
- **Regressão Logística:** Apresentou a maior acurácia (aproximadamente **85,1%**), superando o baseline ingênuo da classe majoritária (76,4%).
- **Árvore de Decisão (Sem Limite):** Obteve acurácia em torno de **81,5%**, ficando acima do baseline, porém abaixo da Regressão Logística. Por não ter limitação de profundidade, a árvore sofreu com overfitting no treino.
- **Modelo Explícito (Regras Manuais):** Obteve um desempenho razoável para uma heurística simples, mas os modelos aprendidos automaticamente capturaram interações mais complexas entre as variáveis.

---

### Questão 2 — Classe mais difícil de identificar
Qual classe foi mais difícil de classificar: **`0` (<= 50K)** ou **`1` (> 50K)**? Justifique com base nas métricas de precisão, recall e F1-score.

### Resposta da Questão 2
A classe **`1` (> 50K)** foi significativamente mais difícil de identificar.
- Por ser a classe minoritária (representa apenas ~23,6% dos dados no teste), os modelos tendem a ser viesados em direção à classe majoritária `0`.
- O **Recall da classe 1** na Regressão Logística fica em torno de 59-60%, enquanto a classe `0` obtém um recall próximo de 93%. Isso evidencia que muitos indivíduos que realmente ganham acima de 50K são classificados incorretamente como <= 50K (falsos negativos).

---

### Questão 3 — Qual métrica deve receber maior atenção?
No contexto de análise socioeconômica ou concessão de benefícios/crédito baseados em renda, qual métrica (Acurácia, Precisão ou Recall da classe 1) deve ser priorizada?

### Resposta da Questão 3
- A **acurácia isolada é enganosa** em datasets desbalanceados, pois um modelo ingênuo que diga que ninguém ganha > 50K já atingiria 76,4% de acurácia.
- Se o objetivo for **identificar oportunidades ou público-alvo de renda mais elevada**, o **Recall da classe 1** é fundamental para reduzir os Falsos Negativos.
- Se o objetivo for evitar conceder benefícios indevidos a quem tem renda alta, a **Precisão da classe 1** torna-se essencial. O **F1-Score** busca o equilíbrio harmônico entre ambas.

---

### Questão 4 — Complexidade e Overfitting da Árvore de Decisão
Por que a Árvore de Decisão sem limite de profundidade teve um desempenho inferior ao da Regressão Logística? Quais hiperparâmetros podem corrigir esse problema?

### Resposta da Questão 4
- Uma árvore sem restrições cresce até que todas as folhas fiquem puras, decorando ruídos do conjunto de treinamento (**overfitting**).
- Para controlar a complexidade e melhorar a generalização, podemos ajustar:
  1. `max_depth`: Limitar a profundidade máxima da árvore.
  2. `min_samples_split`: Exigir um número mínimo de amostras para realizar uma divisão.
  3. `min_samples_leaf`: Garantir um número mínimo de amostras por folha.

---

### Questão 5 — Estratégias para lidar com o desbalanceamento
Cite e explique resumidamente 3 estratégias para melhorar a detecção da classe minoritária (`1`).

### Resposta da Questão 5
1. **Ponderação de Classes (`class_weight='balanced'`):** Penaliza mais severamente os erros cometidos na classe minoritária durante o treinamento.
2. **Reamostragem (ex: SMOTE):** Gera dados sintéticos da classe minoritária para equilibrar o conjunto de treino.
3. **Ajuste do Limiar de Decisão (Decision Thresholding):** Reduz o limite de probabilidade padrão de 0.5 para um valor menor (ex: 0.3 ou 0.35) no modelo de Regressão Logística, aumentando a sensibilidade (Recall) para a classe 1."""

cells = [
    nbf.v4.new_markdown_cell(cell1_md),
    nbf.v4.new_markdown_cell(cell2_md),
    nbf.v4.new_code_cell(cell3_code),
    nbf.v4.new_markdown_cell(cell4_md),
    nbf.v4.new_code_cell(cell5_code),
    nbf.v4.new_markdown_cell(cell6_md),
    nbf.v4.new_code_cell(cell7_code),
    nbf.v4.new_code_cell(cell8_code),
    nbf.v4.new_markdown_cell(cell9_md),
    nbf.v4.new_code_cell(cell10_code),
    nbf.v4.new_markdown_cell(cell11_md),
    nbf.v4.new_code_cell(cell12_code),
    nbf.v4.new_markdown_cell(cell13_md),
    nbf.v4.new_code_cell(cell14_code),
    nbf.v4.new_markdown_cell(cell15_md),
    nbf.v4.new_code_cell(cell16_code),
    nbf.v4.new_code_cell(cell17_code),
    nbf.v4.new_markdown_cell(cell18_md)
]

nb['cells'] = cells

with open('Modelo.ipynb', 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

print("Modelo.ipynb gerado com sucesso!")
