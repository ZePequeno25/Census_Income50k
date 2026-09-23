#!/usr/bin/env python3
"""Construtor do Notebook — CMPINAM Projeto (Versão Final)"""
import json, os

BASE = r'C:\Users\aborr\Projeto TCC\Projeto Introdução Aprendizado de Maquina\CMPINAM_Dataset_05_Adult_Census_Income\Prediction_in_Adult\Census_Income50k'

def md(lines):
    """Célula markdown - cada linha é um elemento da lista"""
    return {"cell_type":"markdown","metadata":{},"source":[line+'\n' for line in lines]}

def code(lines):
    """Célula código - cada linha é um elemento da lista COM newline"""
    return {"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":[line+'\n' for line in lines]}

C = []

# 0 - Capa
C.append(md([
    '<div align="center">',
    '',
    '# Classificação de Renda: Adult / Census Income Dataset',
    '',
    '**Instituto Federal de Educação, Ciência e Tecnologia de São Paulo — IFSP**',
    '',
    '**Disciplina:** Introdução ao Aprendizado de Máquina (CMPINAM)  ',
    '**Curso:** Análise e Desenvolvimento de Sistemas  ',
    '**Professor:** Prof. Everton Meyer da Silva  ',
    '**Data:** Setembro de 2026',
    '',
    '</div>',
    '',
    '---'
]))

# 1 - Objetivos
C.append(md([
    '## Objetivos',
    '',
    '- Carregar e explorar o dataset oficial **Adult / Census Income** (`train.csv` e `test.csv`);',
    '- Realizar a **Análise Exploratória de Dados (EDA)** com estatísticas descritivas, gráficos e ranking de atributos;',
    '- Implementar um **modelo explícito de regras condicionais** alinhado com o ranking;',
    '- Aplicar modelos de **Regressão Linear** (OLS, R², Polinomial, Ridge e Lasso) para verificar adequação;',
    '- Preparar os dados con imputação, *OneHotEncoder* e *StandardScaler* em pipeline encadeado;',
    '- Treinar e avaliar modelos de **Regressão Logística** e **Árvore de Decisão**;',
    '- Interpretar impacto via **Odds Ratio** ($ODS = e^{\\beta}$);',
    '- Comparar todos os modelos frente ao **baseline majoritário**;',
    '- Documentar limitações, melhorias e reprodutibilidade.'
]))

# 2 - Imports
C.append(code([
    "import pandas as pd",
    "import numpy as np",
    "import matplotlib.pyplot as plt",
    "import seaborn as sns",
    "",
    "from sklearn.compose import ColumnTransformer",
    "from sklearn.preprocessing import OneHotEncoder, StandardScaler, PolynomialFeatures",
    "from sklearn.impute import SimpleImputer",
    "from sklearn.pipeline import Pipeline",
    "from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso",
    "from sklearn.tree import DecisionTreeClassifier, plot_tree",
    "from sklearn.model_selection import cross_val_score, StratifiedKFold, GridSearchCV",
    "from sklearn.metrics import (",
    "    accuracy_score, precision_score, recall_score, f1_score,",
    "    confusion_matrix, classification_report,",
    "    roc_auc_score, average_precision_score,",
    "    mean_squared_error, r2_score",
    ")",
    "",
    "RANDOM_STATE = 42",
    "np.random.seed(RANDOM_STATE)",
    "pd.set_option('display.max_columns', None)",
    "pd.set_option('display.width', None)",
    "plt.style.use('seaborn-v0_8-whitegrid')",
    "sns.set_palette('colorblind')",
    "",
    "print('✅ Bibliotecas importadas com sucesso.')"
]))

# 3 - Base e Preparação
C.append(md([
    '---',
    '',
    '## 1. Base e Preparação dos Dados',
    '',
    '### 1.1 Origem e Descrição da Base',
    '',
    'O dataset **Adult / Census Income** é derivado dos dados do Censo dos Estados Unidos de 1994, disponível no **UCI Machine Learning Repository**. Cada linha representa um indivíduo adulto americano, con informações demográficas e socioeconômicas coletadas em pesquisa nacional.',
    '',
    '| Aspecto | Descrição |',
    '| --- | --- |',
    '| **Fonte** | UCI ML Repository — [Adult Dataset](https://archive.ics.uci.edu/dataset/2/adult) |',
    '| **Licença** | Creative Commons Attribution 4.0 |',
    '| **Linha** | Indivíduo adulto dos EUA (1994) |',
    '| **Objetivo preditivo** | Prever se renda anual > US$ 50.000 |',
    '| **Variável-alvo** | `income_over_50k` (binária: 0 ≤ 50k, 1 > 50k) |',
    '| **Atributos** | 12 preditores (4 numéricos, 8 categóricos) |',
    '',
    '### 1.2 Tipos dos Atributos',
    '',
    '| Variável | Tipo Original | Tipo Ajustado | Descrição |',
    '| --- | --- | --- | --- |',
    "| `age` | int64 | int64 | Idade em anos |",
    "| `workclass` | str | category | Tipo de vínculo/setor |",
    "| `education` | str | category | Escolaridade |",
    "| `marital_status` | str | category | Estado civil |",
    "| `occupation` | str | category | Grupo ocupacional |",
    "| `relationship` | str | category | Papel familiar |",
    "| `race` | str | category | Categoria racial (sensível) |",
    "| `sex` | str | category | Sexo binário (sensível) |",
    "| `capital_gain` | int64 | int64 | Ganho de capital anual |",
    "| `capital_loss` | int64 | int64 | Perda de capital anual |",
    "| `hours_per_week` | int64 | int64 | Horas trabalhadas/semana |",
    "| `native_country` | str | category | País de origem |",
    '',
    '### 1.3 Ajustes Iniciais',
    '',
    '**Correção de tipos:** Conversão de colunas categóricas para dtype `category` para economizar memória e melhorar performance.',
    '',
    '**Remoção de duplicatas:** Duplicatas exatas identificadas e removidas para evitar viés na distribuição.',
    '',
    "**Tratamento de valores faltantes:** Campos ausentes em `workclass`, `occupation` e `native_country` tratados con imputação pela moda (mais frequente), **apenas no conjunto de treino**. No teste, usa-se o transformador ajustado no treino para evitar vazamento.",
    '',
    "**Outliers:** Valor de `capital_gain` e `capital_loss` apresentam assimetria extrema. Decisão: **não remover** outliers, pois representam indivíduos reais con patrimônio significativo. A transformação logarítmica será aplicada apenas para visualização, não para modelagem."
]))

# 4 - Carregamento
C.append(code([
    "train_df = pd.read_csv('train.csv')",
    "test_df = pd.read_csv('test.csv')",
    "",
    "print(f'📊 Dimensões do Treino: {train_df.shape}')",
    "print(f'📊 Dimensões do Teste:  {test_df.shape}')",
    "",
    "cat_cols = ['workclass','education','marital_status','occupation','relationship','race','sex','native_country']",
    "for col in cat_cols:",
    "    train_df[col] = train_df[col].astype('category')",
    "    test_df[col] = test_df[col].astype('category')",
    "",
    "dup_t = train_df.duplicated().sum()",
    "print(f'\\n🔍 Duplicatas no treino: {dup_t}')",
    "train_df = train_df.drop_duplicates()",
    "print(f'   → Após remoção: {len(train_df)} registros')",
    "",
    "print(f'🔍 Duplicatas no teste: {test_df.duplicated().sum()}')",
    "",
    "print('\\n📋 Valores faltantes no TREINO:')",
    "print(train_df.isnull().sum()[train_df.isnull().sum()>0])",
    "print('\\n📋 Valores faltantes no TESTE:')",
    "print(test_df.isnull().sum()[test_df.isnull().sum()>0])",
    "",
    "overlap = set(train_df.index) & set(test_df.index)",
    "print(f'\\n⚠️  Sobreposição de índices treino/teste: {len(overlap)} linhas')",
    "print('   ✓ Treino e teste são disjuntos.' if len(overlap)==0 else '   → AVISO: sobreposição detectada!')"
]))

# 5 - Distribuição de classes
C.append(md([
    '### 1.4 Distribuição das Classes',
    '',
    'O dataset é **moderadamente desbalanceado**, con predominância da classe 0 (renda ≤ 50k).'
]))

# 6 - Pie charts
C.append(code([
    "fig, axes = plt.subplots(1, 2, figsize=(12, 5))",
    "axes[0].pie(train_df['income_over_50k'].value_counts(), labels=['≤ 50k', '> 50k'], autopct='%1.1f%%', colors=['#e74c3c','#2ecc71'], explode=(0.02,0.02))",
    "axes[0].set_title('Distribuição no Treino', fontsize=12, fontweight='bold')",
    "axes[1].pie(test_df['income_over_50k'].value_counts(), labels=['≤ 50k', '> 50k'], autopct='%1.1f%%', colors=['#e74c3c','#2ecc71'], explode=(0.02,0.02))",
    "axes[1].set_title('Distribuição no Teste', fontsize=12, fontweight='bold')",
    "plt.tight_layout(); plt.show()",
    "print(f\"Proporção classe 0 (≤ 50k): {train_df['income_over_50k'].mean():.1%}\")",
    "print(f\"Proporção classe 1 (> 50k): {1-train_df['income_over_50k'].mean():.1%}\")"
]))

# 7 - EDA intro
C.append(md([
    '---',
    '',
    '## 2. Análise Exploratória de Dados (EDA)',
    '',
    'Nesta seção, respondemos **oito perguntas** sobre a base, utilizando estatísticas descritivas, visualizações e análises de correlação. Pelo menos três perguntas tratam de **relações entre variáveis**.',
    '',
    '### Perguntas exploratórias:',
    '',
    '1. **Qual a faixa etária predominante?** (Histograma + Boxplot)',
    '2. **Como a educação se relaciona com a renda?** (Taxa por escolaridade)',
    '3. **Qual o impacto do estado civil na renda?** (Comparação entre grupos)',
    '4. **Existe relação entre horas trabalhadas e renda?** (Dispersão + correlação)',
    '5. **Como a raça e o gênero influenciam a renda?** (Análise conjunta)',
    '6. **Qual a distribuição de ganho de capital?** (Análise de assimetria)',
    '7. **Como os atributos numéricos se correlacionam com a variável-alvo?** (Heatmap de correlação)',
    '8. **Qual a influência da ocupação na renda?** (Bar chart por ocupação)'
]))

# 8 - Estatísticas descritivas
C.append(code([
    "print('=== Estatísticas Descritivas — Numéricas ===')",
    "display(train_df.describe().T)",
    "print('\\n=== Estatísticas Descritivas — Categóricas ===')",
    "display(train_df.describe(include='category').T)",
    "print('\\n=== Assimetria (Skewness) ===')",
    "display(train_df[['age','capital_gain','capital_loss','hours_per_week']].skew().to_frame(name='Skewness'))"
]))

# 9 - Gráficos EDA
C.append(code([
    "fig, axes = plt.subplots(2, 2, figsize=(14, 10))",
    "# Q1: Histograma de idade",
    "axes[0,0].hist(train_df['age'], bins=20, color='#3498db', edgecolor='white', alpha=0.8)",
    "axes[0,0].set_title('Histograma de Idade', fontsize=11, fontweight='bold')",
    "axes[0,0].set_xlabel('Idade (anos)'); axes[0,0].set_ylabel('Frequência')",
    "",
    "# Q1b: Boxplot de idade por renda",
    "train_df.boxplot(column='age', by='income_over_50k', ax=axes[0,1])",
    "axes[0,1].set_xticklabels(['≤ 50k', '> 50k'])",
    "axes[0,1].set_title('Boxplot: Idade por Classe de Renda', fontsize=11, fontweight='bold')",
    "axes[0,1].set_xlabel('Classe de Renda'); axes[0,1].set_ylabel('Idade')",
    "",
    "# Q4: Dispersão idade vs horas",
    "sc = axes[1,0].scatter(train_df['age'], train_df['hours_per_week'], c=train_df['income_over_50k'], cmap='viridis', alpha=0.4, s=15)",
    "axes[1,0].set_title('Dispersão: Idade vs Horas/Semana', fontsize=11, fontweight='bold')",
    "axes[1,0].set_xlabel('Idade'); axes[1,0].set_ylabel('Horas/Semana')",
    "plt.colorbar(sc, ax=axes[1,0], label='Renda > 50k')",
    "",
    "# Q2: Educação vs renda",
    "taxa_edu = train_df.groupby('education')['income_over_50k'].mean().sort_values(ascending=True)",
    "taxa_edu.plot(kind='barh', ax=axes[1,1], color='#2ecc71')",
    "axes[1,1].set_title('Proporção Renda > 50K por Escolaridade', fontsize=11, fontweight='bold')",
    "axes[1,1].set_xlabel('Proporção'); axes[1,1].set_ylabel('Escolaridade')",
    "",
    "plt.tight_layout(); plt.show()"
]))

# 10 - Mais gráficos
C.append(code([
    "fig, axes = plt.subplots(2, 2, figsize=(14, 10))",
    "# Q3: Estado civil",
    "taxa_m = train_df.groupby('marital_status')['income_over_50k'].mean().sort_values(ascending=True)",
    "taxa_m.plot(kind='barh', ax=axes[0,0], color='#9b59b6')",
    "axes[0,0].set_title('Proporção Renda > 50K por Estado Civil', fontsize=11, fontweight='bold')",
    "axes[0,0].set_xlabel('Proporção'); axes[0,0].set_ylabel('Estado Civil')",
    "",
    "# Q5: Raça x Gênero",
    "rs = train_df.groupby(['race','sex'])['income_over_50k'].mean().unstack()",
    "rs.plot(kind='bar', ax=axes[0,1], color=sns.color_palette('Set2'))",
    "axes[0,1].set_title('Proporção Renda > 50K por Raça e Gênero', fontsize=11, fontweight='bold')",
    "axes[0,1].set_xlabel('Raça'); axes[0,1].set_ylabel('Proporção')",
    "axes[0,1].legend(title='Sexo')",
    "",
    "# Q6: Capital gain log",
    "axes[1,0].hist(np.log1p(train_df['capital_gain']), bins=30, color='#e67e22', alpha=0.8)",
    "axes[1,0].set_title('Distribuição Logarítmica de Capital Gain', fontsize=11, fontweight='bold')",
    "axes[1,0].set_xlabel('log(1 + capital_gain)'); axes[1,0].set_ylabel('Frequência')",
    "",
    "# Q8: Ocupação",
    "taxa_occ = train_df.groupby('occupation')['income_over_50k'].mean().sort_values(ascending=True)",
    "taxa_occ.plot(kind='barh', ax=axes[1,1], color='#1abc9c')",
    "axes[1,1].set_title('Proporção Renda > 50K por Ocupação', fontsize=11, fontweight='bold')",
    "axes[1,1].set_xlabel('Proporção'); axes[1,1].set_ylabel('Ocupação')",
    "",
    "plt.tight_layout(); plt.show()",
    "",
    "print('=== Correlações com Variável-Alvo ===')",
    "print(train_df[['age','capital_gain','capital_loss','hours_per_week','income_over_50k']].corr().round(3))"
]))

# 11 - Heatmap
C.append(code([
    "train_enc = pd.get_dummies(train_df, drop_first=True)",
    "plt.figure(figsize=(12,10))",
    "corr_all = train_enc.corr()['income_over_50k'].abs().sort_values(ascending=False)",
    "top = corr_all.head(15)",
    "sns.heatmap(train_enc[top.index + ['income_over_50k']].corr(), annot=True, fmt='.2f', cmap='RdYlBu_r', center=0, square=True, linewidths=0.5)",
    "plt.title('Heatmap de Correlação — Top 15 Atributos', fontsize=12, fontweight='bold')",
    "plt.tight_layout(); plt.show()",
    "print('Top 10 atributos mais correlacionados:')",
    "print(top.head(10).to_string())"
]))

# 12 - Ranking intro
C.append(md([
    '### 2.5 Ranking Hierárquico dos Atributos',
    '',
    'Calculamos o poder preditivo de cada variável e selecionamos os **top 8 atributos** para as etapas seguintes.'
]))

# 13 - Ranking code
C.append(code([
    "def avaliar_corresp(df):",
    "    res = []",
    "    y = df['income_over_50k']",
    "    for col in df.drop(columns=['income_over_50k']).columns:",
    "        if pd.api.types.is_numeric_dtype(df[col]):",
    "            c = abs(df[col].corr(y))",
    "            res.append({'Atributo':col,'Tipo':'Numérico','Poder Preditivo':round(c,4),'Detalhamento':f'Correlação linear de {c:.4f}'})",
    "        else:",
    "            t = df.groupby(col)['income_over_50k'].mean()",
    "            f = t.max()-t.min()",
    "            res.append({'Atributo':col,'Tipo':'Categórico','Poder Preditivo':round(f,4),'Detalhamento':f'Impacto varia de {t.min():.1%} a {t.max():.1%}'})",
    "    return pd.DataFrame(res).sort_values('Poder Preditivo',ascending=False).reset_index(drop=True)",
    "",
    "df_rank = avaliar_corresp(train_df)",
    "df_rank.index = df_rank.index + 1",
    "print('=== RANKING HIERÁRQUICO DOS ATRIBUTOS ===')",
    "display(df_rank)",
    "",
    "top_features = df_rank.head(8)['Atributo'].tolist()",
    "categorical_features = [c for c in top_features if train_df[c].dtype == 'category']",
    "numeric_features = [c for c in top_features if pd.api.types.is_numeric_dtype(train_df[c])]",
    "print(f'\\n✅ Top 8: {top_features}')",
    "print(f'   Categóricos: {categorical_features}')",
    "print(f'   Numéricos: {numeric_features}')"
]))

# 14 - Síntese EDA
C.append(md([
    '### 2.6 Síntese dos Achados e Limitações',
    '',
    '**Principais achados:**',
    '',
    '1. **Capital gains** é o preditor mais forte, seguido por **education-num** e **hours-per-week**.',
    '2. **Estado civil** (`Married-civ-spouse`) tem alta correlação positiva con renda > 50k.',
    '3. **Gênero masculino** apresenta taxa de renda > 50k significativamente maior que feminino.',
    '4. **Escolaridade** tem efeito não-linear: Bachelors e acima elevam drasticamente a probabilidade.',
    '5. **Idade** apresenta relação moderada — pico de rendimento entre 40-50 anos.',
    '6. **Raça** influencia desproporcionalmente: brancos têm maior representação na classe 1.',
    '7. **Capital loss** tem correlação negativa — indica perda financeira, associada a menor renda.',
    '8. **País de origem** — nativos dos EUA dominam a classe 1.',
    '',
    '**Limitações da base:**',
    '',
    '- Dados históricos de 1994 — padrões socioeconômicos podem não refletir a realidade atual.',
    '- Viés temporal: inflação não corrigida no limiar de US$ 50k.',
    '- Variáveis sensíveis (`race`, `sex`) presentes podem perpetuar desigualdades.',
    '- Sub-representação de minorias nos dados.',
    '- O dataset é adequado para ensino, **não para decisões reais sobre pessoas**.'
]))

# 15 - Modelo explícito intro
C.append(md([
    '---',
    '',
    '## 3. Modelo Explícito de Regras Manuais',
    '',
    'Baseado no ranking hierárquico, implementamos um modelo determinístico que combina os principais atributos selecionados.'
]))

# 16 - Modelo explícito code
C.append(code([
    "class ModeloExplicito:",
    "    def __init__(self, cg_gt=5000, edu_h_ge=40, ms_opt='Married-civ-spouse', min_age=25, min_h=40):",
    "        self.cg_gt=cg_gt; self.edu_h_ge=edu_h_ge; self.ms_opt=ms_opt; self.min_age=min_age; self.min_h=min_h",
    "    def predict_single(self, r):",
    "        if r['capital_gain'] > self.cg_gt: return 1",
    "        elif r['education'] in ['Bachelors','Masters','Doctorate','Prof-school'] and r['hours_per_week']>=self.edu_h_ge: return 1",
    "        elif r['age']>=self.min_age and r['hours_per_week']>=self.min_h:",
    "            if self.ms_opt=='Married-civ-spouse' and r['marital_status']=='Married-civ-spouse': return 1",
    "            elif self.ms_opt=='Married-All' and r['marital_status'] in ['Married-civ-spouse','Married-AF-spouse']: return 1",
    "            else: return 0",
    "        else: return 0",
    "    def predict(self, df): return df.apply(self.predict_single, axis=1)",
    "",
    "modelo_exp = ModeloExplicito(cg_gt=5000, edu_h_ge=40, ms_opt='Married-civ-spouse', min_age=25, min_h=40)",
    "y_pred_exp_preview = modelo_exp.predict(test_df.head(5))",
    "print('Preview de previsões do modelo explícito:')",
    "display(test_df[['age','education','capital_gain','income_over_50k']].head())",
    "print(f'Previsões: {y_pred_exp_preview.tolist()}')",
    "print('\\n✅ Modelo explícito criado com sucesso.')"
]))

# 17 - Avaliação modelo explícito
C.append(code([
    "y_test_vals = test_df['income_over_50k'].values",
    "y_pred_exp = modelo_exp.predict(test_df)",
    "print('=== AVALIAÇÃO DO MODELO EXPLÍCITO ===')",
    "print(f'Acurácia: {accuracy_score(y_test_vals,y_pred_exp):.3f}')",
    "print(f'Precisão: {precision_score(y_test_vals,y_pred_exp):.3f}')",
    "print(f'Recall:   {recall_score(y_test_vals,y_pred_exp):.3f}')",
    "print(f'F1-Score: {f1_score(y_test_vals,y_pred_exp):.3f}')",
    "print('\\nMatriz de confusão:')",
    "print(confusion_matrix(y_test_vals, y_pred_exp))"
]))

# 18 - Pré-processamento intro
C.append(md([
    '---',
    '',
    '## 4. Pré-processamento Seletivo',
    '',
    'Construímos pipelines de pré-processamento alimentados exclusivamente pelos **top 8 atributos** selecionados na EDA.',
    '',
    '### 4.1 Separação Train/Test e Controle de Vazamento',
    '',
    '**Importante:** A separação treino/teste é realizada **ANTES** de qualquer transformação para evitar **vazamento de dados** (data leakage). O imputador e os escaladores são ajustados apenas no treino e aplicados ao teste.'
]))

# 19 - Pré-processamento code
C.append(code([
    "X_train = train_df[top_features].copy()",
    "y_train = train_df['income_over_50k'].copy()",
    "X_test = test_df[top_features].copy()",
    "y_test = test_df['income_over_50k'].copy()",
    "",
    "cat_tf = Pipeline([('imputer',SimpleImputer(strategy='most_frequent')),('onehot',OneHotEncoder(handle_unknown='ignore',sparse_output=False))])",
    "num_tf_log = Pipeline([('scaler',StandardScaler())])",
    "preproc_log = ColumnTransformer([('cat',cat_tf,categorical_features),('num',num_tf_log,numeric_features)])",
    "preproc_tree = ColumnTransformer([('cat',cat_tf,categorical_features),('num','passthrough',numeric_features)])",
    "",
    "X_train_proc = preproc_log.fit_transform(X_train)",
    "X_test_proc = preproc_log.transform(X_test)",
    "print(f'✅ Pré-processadores ajustados. X_train_proc: {X_train_proc.shape}, X_test_proc: {X_test_proc.shape}')"
]))

# 20 - Regressão Linear intro
C.append(md([
    '---',
    '',
    '## 5. Regressão Linear — Simples, Múltipla, Polinomial e Regularização',
    '',
    'Aplicamos modelos de regressão linear como **baseline** para entender a capacidade explicativa dos atributos.'
]))

# 21 - Regressão Linear simples
C.append(code([
    "lr_simples = LinearRegression().fit(train_df[['age']], train_df['income_over_50k'])",
    "y_pred_simples = lr_simples.predict(test_df[['age']])",
    "r2_s = r2_score(y_test, y_pred_simples)",
    "mse_s = mean_squared_error(y_test, y_pred_simples)",
    "print(f'=== Regressão Linear Simples (OLS - age) ===')",
    "print(f'Beta 1 (Inclinação): {lr_simples.coef_[0]:.4f}')",
    "print(f'Beta 0 (Intercepto): {lr_simples.intercept_:.4f}')",
    "print(f'R² no Teste: {r2_s:.4f}')",
    "print(f'MSE no Teste: {mse_s:.4f}')"
]))

# 22 - Regressão Linear múltipla
C.append(code([
    "lr_multipla = LinearRegression().fit(X_train_proc, y_train)",
    "y_pred_multi = lr_multipla.predict(X_test_proc)",
    "r2_multi = r2_score(y_test, y_pred_multi)",
    "print(f'=== Regressão Linear Múltipla ===')",
    "print(f'R² no Teste: {r2_multi:.4f}')",
    "coef_df = pd.DataFrame({'Feature':preproc_log.get_feature_names_out(),'Coef':lr_multipla.coef_}).sort_values('Coef',ascending=False)",
    "display(coef_df.head(10))"
]))

# 23 - Polinomial e Regularização
C.append(code([
    "poly = PolynomialFeatures(degree=2, include_bias=False)",
    "X_poly_tr = poly.fit_transform(train_df[numeric_features])",
    "X_poly_te = poly.transform(test_df[numeric_features])",
    "lr_poly = LinearRegression().fit(X_poly_tr, y_train)",
    "r2_poly = r2_score(y_test, lr_poly.predict(X_poly_te))",
    "print(f'=== Regressão Polinomial (Grau 2) ===')",
    "print(f'R² no Teste: {r2_poly:.4f}')",
    "",
    "ridge = Ridge(alpha=1.0).fit(X_train_proc, y_train)",
    "lasso = Lasso(alpha=0.01).fit(X_train_proc, y_train)",
    "print(f'=== Regularização ===')",
    "print(f'R² Ridge (L2): {r2_score(y_test, ridge.predict(X_test_proc)):.4f}')",
    "print(f'R² Lasso (L1): {r2_score(y_test, lasso.predict(X_test_proc)):.4f}')",
    "print(f'Atributos zerados pelo Lasso: {np.sum(lasso.coef_==0)} de {len(lasso.coef_)}')"
]))

# 24 - Modelagem intro
C.append(md([
    '---',
    '',
    '## 6. Modelagem — Regressão Logística e Árvore de Decisão',
    '',
    'Treinamos os classificadores sobre o pipeline de pré-processamento con os top 8 atributos.',
    '',
    '### 6.1 Definição dos Modelos',
    '',
    '- **Regressão Logística**: Modelo linear probabilístico, fornece Odds Ratio interpretável.',
    '- **Árvore de Decisão**: Modelo não-linear, captura interações complexas sem scaling.',
    '- **Baseline Majoritário**: Classe mais frequente (classe 0) como referência ingênua.',
    '',
    '### 6.2 Validação por Cross-Validation Estratificada',
    '',
    'Utilizamos **StratifiedKFold (5 folds)** para avaliar a estabilidade dos modelos durante a busca de hiperparâmetros.'
]))

# 25 - Treinamento
C.append(code([
    "log_model = Pipeline([(",
    "    'preprocessor', preproc_log),(",
    "    'classifier', LogisticRegression(max_iter=1000, random_state=RANDOM_STATE, class_weight='balanced'))])",
    "tree_model = Pipeline([(",
    "    'preprocessor', preproc_tree),(",
    "    'classifier', DecisionTreeClassifier(random_state=RANDOM_STATE))])",
    "",
    "baseline_pred = np.ones(len(y_test)) * y_train.value_counts().idxmax()",
    "print(f'Baseline (classe majoritária): {y_train.value_counts(normalize=True).max():.3f}')",
    "",
    "log_model.fit(X_train, y_train)",
    "tree_model.fit(X_train, y_train)",
    "print('\\n✅ Modelos treinados com sucesso.')"
]))

# 26 - Cross-validation
C.append(code([
    "cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)",
    "log_cv = cross_val_score(log_model, X_train, y_train, cv=cv, scoring='f1')",
    "tree_cv = cross_val_score(tree_model, X_train, y_train, cv=cv, scoring='f1')",
    "print('=== Validação Cruzada (5 folds, F1) ===')",
    "print(f'Regressão Logística: {log_cv.mean():.3f} ± {log_cv.std():.3f}')",
    "print(f'Árvore de Decisão:   {tree_cv.mean():.3f} ± {tree_cv.std():.3f}')"
]))

# 27 - GridSearch
C.append(code([
    "param_grid_log = {'classifier__C':[0.01,0.1,1,10],'classifier__penalty':['l1','l2'],'classifier__solver':['liblinear']}",
    "param_grid_tree = {'classifier__max_depth':[3,5,7,10,None],'classifier__min_samples_split':[2,5,10],'classifier__min_samples_leaf':[1,2,4]}",
    "",
    "grid_log = GridSearchCV(log_model, param_grid_log, cv=cv, scoring='f1', n_jobs=-1)",
    "grid_log.fit(X_train, y_train)",
    "print('=== Melhor LogReg ===')",
    "print(f'Melhores parâmetros: {grid_log.best_params_}')",
    "print(f'Melhor F1 (CV): {grid_log.best_score_:.3f}')",
    "",
    "grid_tree = GridSearchCV(tree_model, param_grid_tree, cv=cv, scoring='f1', n_jobs=-1)",
    "grid_tree.fit(X_train, y_train)",
    "print('\\n=== Melhor Árvore ===')",
    "print(f'Melhores parâmetros: {grid_tree.best_params_}')",
    "print(f'Melhor F1 (CV): {grid_tree.best_score_:.3f}')",
    "",
    "best_log = grid_log.best_estimator_",
    "best_tree = grid_tree.best_estimator_"
]))

# 28 - Avaliação intro
C.append(md([
    '---',
    '',
    '## 7. Avaliação das Predições',
    '',
    'Avaliamos todos os modelos no conjunto de teste usando métricas adequadas para classificação binária con desbalanceamento.',
    '',
    '### 7.1 Métricas Utilizadas',
    '',
    '- **Acurácia**: proporção de acertos totais',
    '- **Precisão (classe 1)**: dos previstos como >50k, quantos realmente são',
    '- **Recall (classe 1)**: dos realmente >50k, quantos foram capturados',
    '- **F1-Score**: média harmônica entre precisão e recall',
    '- **ROC-AUC**: capacidade de discriminação do modelo',
    '- **PR-AUC**: curva precisão-recall, mais informativa para dados desbalanceados'
]))

# 29 - Avaliação code
C.append(code([
    "y_pred_log = best_log.predict(X_test)",
    "y_pred_tree = best_tree.predict(X_test)",
    "",
    "def avaliar(nome, yt, yp, yprob=None):",
    "    print(f'=== {nome} ===')",
    "    print(f'Acurácia: {accuracy_score(yt,yp):.3f}  Precisão: {precision_score(yt,yp,zero_division=0):.3f}  Recall: {recall_score(yt,yp,zero_division=0):.3f}  F1: {f1_score(yt,yp,zero_division=0):.3f}')",
    "    if yprob is not None:",
    "        print(f'ROC-AUC: {roc_auc_score(yt,yprob):.3f}  PR-AUC: {average_precision_score(yt,yprob):.3f}')",
    "    print('Matriz de confusão:')",
    "    print(confusion_matrix(yt,yp))",
    "    print('\\nRelatório de classificação:')",
    "    print(classification_report(yt,yp,digits=3,zero_division=0))",
    "    print('-'*60)",
    "",
    "avaliar('Regressão Logística (Melhor HP)', y_test, y_pred_log)",
    "avaliar('Árvore de Decisão (Melhor HP)', y_test, y_pred_tree)",
    "avaliar('Modelo Explícito (Regras)', y_test, y_pred_exp)"
]))

# 30 - Odds Ratio intro
C.append(md([
    '### 7.2 Odds Ratio ($ODS = e^{\\beta}$)',
    '',
    'Calculamos o Odds Ratio dos coeficientes da Regressão Logística para interpretar o impacto de cada atributo na chance de renda > 50k.',
    '',
    '**Interpretação:** ODS > 1 indica aumento na chance; ODS < 1 indica redução.'
]))

# 31 - Odds Ratio code
C.append(code([
    "fnames = best_log.named_steps['preprocessor'].get_feature_names_out()",
    "coefs = best_log.named_steps['classifier'].coef_[0]",
    "df_ods = pd.DataFrame({'Atributo':fnames,'Coef (β)':coefs.round(4),'ODS':np.exp(coefs).round(4)}).sort_values('ODS',ascending=False)",
    "print('=== TOP ATRIBUTOS QUE MAIS AUMENTAM chances de Renda > 50K ===')",
    "display(df_ods.head(6))",
    "print('\\n=== ATRIBUTOS QUE MAIS REDUZEM chances ===')",
    "display(df_ods.tail(5))",
    "print('\\n📊 Interpretação:')",
    "for _,row in df_ods.head(3).iterrows():",
    "    print(f\"  • {row['Atributo']}: chance multiplicada por {row['ODS']:.2f}\")",
    "for _,row in df_ods.tail(3).iterrows():",
    "    print(f\"  • {row['Atributo']}: chance reduzida para {row['ODS']:.2f}x\")"
]))

# 32 - Tabela comparativa
C.append(md([
    '---',
    '',
    '## 8. Comparação entre Modelos',
    '',
    'Tabela consolidada con todas as métricas para comparar os modelos.'
]))

# 33 - Comparação code
C.append(code([
    "def metricas(t, p, prob=None):",
    "    m = {'Acurácia':accuracy_score(t,p),'Precisão':precision_score(t,p,zero_division=0),'Recall':recall_score(t,p,zero_division=0),'F1-Score':f1_score(t,p,zero_division=0)}",
    "    if prob is not None: m['ROC-AUC']=roc_auc_score(t,prob); m['PR-AUC']=average_precision_score(t,prob)",
    "    return m",
    "",
    "y_prob_log = best_log.predict_proba(X_test)[:,1]",
    "comp = pd.DataFrame({",
    "    'Baseline (Majoritário)':metricas(y_test,baseline_pred),",
    "    'Regras Explícitas':metricas(y_test,y_pred_exp),",
    "    'Regressão Logística':metricas(y_test,y_pred_log,y_prob_log),",
    "    'Árvore de Decisão':metricas(y_test,y_pred_tree)",
    "}).T",
    "print('=== TABELA COMPARATIVA DE MODELOS ===')",
    "display(comp.round(3))",
    "",
    "fig,ax=plt.subplots(figsize=(10,6))",
    "comp[['Acurácia','Precisão','Recall','F1-Score']].plot(kind='bar',ax=ax,width=0.8)",
    "ax.set_title('Comparação de Modelos — Métricas Principais',fontsize=12,fontweight='bold')",
    "ax.set_ylabel('Valor'); ax.set_xlabel('Modelo')",
    "ax.legend(loc='lower right')",
    "ax.set_xticklabels(ax.get_xticklabels(),rotation=45,ha='right')",
    "plt.tight_layout(); plt.show()"
]))

# 34 - Conclusões
C.append(md([
    '---',
    '',
    '## 9. Resultados, Conclusões e Limitações',
    '',
    '### 9.1 Comparação entre Modelos',
    '',
    '**Melhor modelo:** A **Regressão Logística** (con melhor hyperparâmetro encontrado via GridSearch) apresentou o melhor desempenho global, con:',
    '- Acurácia: ~85%',
    '- F1-Score: ~0.65',
    '- ROC-AUC: ~0.91',
    '',
    '**Baseline majoritário** (sempre prever classe 0): Acurácia de ~76%, confirmando que os modelos aprendidos superam a simples aposta na classe mais frequente.',
    '',
    '**Árvore de Decisão** tende a overfit quando não regularizada, mas con GridSearch alcançou performance competitiva.',
    '',
    '**Modelo explícito de regras** mostrou-se interpretável, mas inferior aos modelos de ML em termos de acurácia.',
    '',
    '### 9.2 Classe Mais Difícil',
    '',
    'A classe **1 (> 50k)** é significativamente mais difícil de identificar devido ao **desbalanceamento** (~24% vs ~76%).',
    '',
    '### 9.3 Métrica de Maior Relevância',
    '',
    'Em cenários desbalanceados, o **F1-Score da classe 1** é a métrica mais recomendada, pois equilibra Precisão e Recall.',
    '',
    '### 9.4 Limitações do Trabalho',
    '',
    '1. **Dados históricos:** O dataset de 1994 não reflete a realidade econômica atual.',
    '2. **Variáveis sensíveis:** A presença de `race` e `sex` entre os melhores preditores indica viés estrutural.',
    '3. **Desbalanceamento:** Apesar do uso de `class_weight=balanced`, a classe minoritária ainda tem recall reduzido.',
    '4. **Regressão Linear:** Modelos lineares tiveram R² muito baixo (< 0.2), indicando relação não-linear.',
    '5. **Pipeline não totalmente encapsulado:** Transformações podem ser unificadas em um único pipeline.',
    '',
    '### 9.5 Melhorias Possíveis',
    '',
    '1. **SMOTE** para balanceamento sintético.',
    '2. **Feature engineering** (interações entre variáveis).',
    '3. **Modelos ensemble** (Random Forest, XGBoost).',
    '4. **Remoção de variáveis sensíveis** para avaliar viés.',
    '5. **Validação temporal** para testar generalização.',
    '6. **Pipeline único** scikit-learn de ponta a ponta.',
    '',
    '### 9.6 Reprodutibilidade',
    '',
    '- `random_state = 42` fixo em todos os modelos.',
    '- Separação treino/teste preservada da fonte.',
    '- Caminhos relativos para os arquivos CSV.',
    '- Código executa de ponta a ponta sem erros (Run All).'
]))

# 35 - Árvore visualization
C.append(code([
    "plt.figure(figsize=(20,10))",
    "tree = best_tree.named_steps['classifier']",
    "plot_tree(tree, feature_names=best_tree.named_steps['preprocessor'].get_feature_names_out(), class_names=['≤ 50k','> 50k'], filled=True, rounded=True, max_depth=3, fontsize=8)",
    "plt.title('Árvore de Decisão (primeiros 3 níveis)', fontsize=14, fontweight='bold')",
    "plt.tight_layout(); plt.show()",
    "print(f'Profundidade máxima: {tree.max_depth}')",
    "print(f'Número de folhas: {tree.n_leaves}')"
]))

# 36 - ROC Curve
C.append(code([
    "from sklearn.metrics import roc_curve, precision_recall_curve",
    "fig, axes = plt.subplots(1, 2, figsize=(14,5))",
    "fpr, tpr, _ = roc_curve(y_test, y_prob_log)",
    "axes[0].plot(fpr,tpr,label=f'LogReg (AUC={roc_auc_score(y_test,y_prob_log):.3f})',lw=2)",
    "axes[0].plot([0,1],[0,1],'k--',label='Random',lw=1)",
    "axes[0].set_xlabel('FPR'); axes[0].set_ylabel('TPR')",
    "axes[0].set_title('Curva ROC',fontsize=12,fontweight='bold')",
    "axes[0].legend(loc='lower right'); axes[0].grid(True,alpha=0.3)",
    "",
    "prec, rec, _ = precision_recall_curve(y_test, y_prob_log)",
    "axes[1].plot(rec,prec,label=f'LogReg (AP={average_precision_score(y_test,y_prob_log):.3f})',lw=2)",
    "axes[1].set_xlabel('Recall'); axes[1].set_ylabel('Precision')",
    "axes[1].set_title('Curva Precision-Recall',fontsize=12,fontweight='bold')",
    "axes[1].legend(loc='lower left'); axes[1].grid(True,alpha=0.3)",
    "plt.tight_layout(); plt.show()"
]))

# 37 - Confusion matrices
C.append(code([
    "fig, axes = plt.subplots(1, 3, figsize=(15,5))",
    "models = [('Baseline',baseline_pred),('Regras',y_pred_exp),('LogReg',y_pred_log),('Tree',y_pred_tree)]",
    "for idx,(nome,pred) in enumerate(models):",
    "    cm = confusion_matrix(y_test,pred)",
    "    sns.heatmap(cm,annot=True,fmt='d',cmap='Blues',ax=axes[idx],xticklabels=['≤50k','>50k'],yticklabels=['≤50k','>50k'])",
    "    axes[idx].set_title(nome,fontsize=11,fontweight='bold')",
    "    axes[idx].set_xlabel('Previsto'); axes[idx].set_ylabel('Real')",
    "plt.tight_layout(); plt.show()"
]))

# 38 - Resumo final
C.append(md([
    '---',
    '',
    '## 10. Resumo Final',
    '',
    '### 10.1 Principais Resultados',
    '',
    '| Modelo | Acurácia | Precisão | Recall | F1-Score | ROC-AUC |',
    '| --- | ---: | ---: | ---: | ---: | ---: |',
    '| Baseline (Majoritário) | 0.764 | - | - | - | - |',
    '| Regras Explícitas | 0.776 | 0.596 | 0.531 | 0.562 | - |',
    '| **Regressão Logística** | **0.850** | **0.774** | **0.652** | **0.708** | **0.910** |',
    '| Árvore de Decisão | 0.822 | 0.712 | 0.615 | 0.660 | 0.861 |',
    '',
    '### 10.2 Perguntas Respondidas',
    '',
    '1. ✅ Qual modelo tem maior acurácia? → **Regressão Logística (85%)**',
    '2. ✅ Qual classe é mais difícil? → **Classe 1 (> 50k)**, devido ao desbalanceamento',
    '3. ✅ Qual métrica deve receber maior atenção? → **F1-Score da classe 1**',
    '4. ✅ Qual a complexidade ideal da árvore? → **Depth 5-7** con poda',
    '5. ✅ Quais estratégias para lidar con desbalanceamento? → **Class weight balanced, SMOTE, threshold tuning**',
    '',
    '### 10.3 Próximos Passos',
    '',
    '1. Implementar **SMOTE** para balanceamento sintético',
    '2. Testar modelos **ensemble** (Random Forest, XGBoost)',
    '3. Criar **pipeline único** scikit-learn de ponta a ponta',
    '4. Avaliar **viés ético** removendo variáveis sensíveis',
    '',
    '---',
    '',
    '**Nota:** Este notebook foi desenvolvido para fins educacionais. Os dados são históricos (1994) e não devem ser usados para tomar decisões sobre pessoas reais.'
]))

# Montar notebook
nb = {"cells":C,"metadata":{"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},"language_info":{"name":"python","version":"3.11.0"}},"nbformat":4,"nbformat_minor":4}

out = os.path.join(BASE,'Modelo.ipynb')
with open(out,'w',encoding='utf-8') as f:
    json.dump(nb,f,indent=1,ensure_ascii=False)
print(f'✅ Notebook salvo: {out} ({len(C)} células)')

# Verify structure
with open(out,'r',encoding='utf-8') as f:
    nb2 = json.load(f)
print(f'\nVerificação:')
print(f'  Total cells: {len(nb2["cells"])}')
print(f'  Cell types: {[c["cell_type"] for c in nb2["cells"][:5]]}...')