# CMPINAM — Classificação de Renda (Adult / Census Income)

## Sobre o Projeto

Projeto da disciplina de **Introdução ao Aprendizado de Máquina (CMPINAM)** do curso de **Análise e Desenvolvimento de Sistemas** do **IFSP Câmpus Campinas**.

**Professor:** Prof. Everton Meyer da Silva  
**Semestre:** 2º semestre de 2026

---

## Problema

Classificação binária para prever se a renda anual de um indivíduo adulto americano é superior a US$ 50 mil, baseada em dados do Censo dos EUA de 1994.

**Variável-alvo:** `income_over_50k` (0 ≤ 50k, 1 > 50k)

---

## Base de Dados

- **Fonte:** UCI Machine Learning Repository — [Adult Dataset](https://archive.ics.uci.edu/dataset/2/adult)
- **Licença:** Creative Commons Attribution 4.0
- **Treino:** 32.561 registros
- **Teste:** 16.281 registros
- **Atributos:** 12 preditores (4 numéricos, 8 categóricos)

---

## Integrantes

> *Insira os nomes dos integrantes do grupo aqui*

---

## Metodologia

1. **Preparação dos dados:** Correção de tipos, remoção de duplicatas, tratamento de valores faltantes e análise de outliers
2. **Análise Exploratória (EDA):** 8 perguntas respondidas com visualizações e ranking de atributos
3. **Modelagem:**
   - Modelo explícito de regras manuais
   - Regressão Linear (OLS, Polinomial, Ridge, Lasso)
   - Regressão Logística com busca de hiperparâmetros (GridSearchCV)
   - Árvore de Decisão com busca de hiperparâmetros
4. **Avaliação:** Acurácia, Precisão, Recall, F1-Score, ROC-AUC, PR-AUC
5. **Interpretação:** Odds Ratio para análise de impacto dos atributos

---

## Principais Resultados

| Modelo | Acurácia | Precisão | Recall | F1-Score | ROC-AUC |
| --- | ---: | ---: | ---: | ---: | ---: |
| Baseline (Majoritário) | 0.764 | - | - | - | - |
| Regras Explícitas | 0.776 | 0.596 | 0.531 | 0.562 | - |
| **Regressão Logística** | **0.850** | **0.774** | **0.652** | **0.708** | **0.910** |
| Árvore de Decisão | 0.822 | 0.712 | 0.615 | 0.660 | 0.861 |

**Melhor modelo:** Regressão Logística com 85% de acurácia e 0.91 de ROC-AUC.

---

## Estrutura do Repositório

```
Census_Income50k/
├── Modelo.ipynb              # Notebook principal (código + markdown)
├── train.csv                 # Dados de treino
├── test.csv                  # Dados de teste (fornecidos pelo professor)
├── README.md                 # Este arquivo
├── .gitignore                # Arquivos ignorados pelo Git
├── notebooks/                # Notebooks auxiliares e scripts
│   ├── build_notebook.py     # Script gerador do notebook
│   └── Exemple*.ipynb        # Exemplos de aula
├── docs/                     # Material de referência (PDFs das aulas)
├── results/                  # Resultados executados
│   └── Modelo_executado.ipynb
└── figures/                  # Figuras geradas (se houver)
```

---

## Como Executar

### Pré-requisitos

- Python 3.11+
- Bibliotecas: pandas, numpy, matplotlib, seaborn, scikit-learn

### Passos

1. Clone o repositório:
   ```bash
   git clone <url-do-repositorio>
   cd Census_Income50k
   ```

2. Instale as dependências:
   ```bash
   pip install -r requirements.txt
   ```

3. Execute o notebook:
   - Abra `Modelo.ipynb` no Jupyter Notebook ou JupyterLab
   - Execute todas as células (Run All)

### Reprodutibilidade

- `random_state = 42` fixo em todos os modelos
- Separação treino/teste preservada da fonte original
- Caminhos relativos para os arquivos CSV
- Notebook executa de ponta a ponta sem erros

---

## Limitações

- Dados históricos de 1994 não refletem a realidade econômica atual
- Variáveis sensíveis (`race`, `sex`) presentes nos melhores preditores indicam viés estrutural
- Desbalanceamento de classes (76% classe 0 vs 24% classe 1)
- Regressão linear teve R² muito baixo (< 0.2), indicando relação não-linear

---

## Melhorias Possíveis

1. Implementar SMOTE para balanceamento sintético
2. Feature engineering com interações entre variáveis
3. Modelos ensemble (Random Forest, XGBoost)
4. Remoção de variáveis sensíveis para avaliar viés
5. Validação temporal para testar generalização
6. Pipeline único scikit-learn de ponta a ponta

---

## Referências

- Becker, B. e Kohavi, R. (1996). *Adult* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5XW20
- IFSP Campinas — Disciplina CMPINAM (Introdução ao Aprendizado de Máquina)

---

**Nota:** Este notebook foi desenvolvido para fins educacionais. Os dados são históricos e não devem ser usados para tomar decisões reais sobre pessoas.
