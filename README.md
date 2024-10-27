# KNeighborsClassifier para Análise de Concessão de Crédito

Este projeto utiliza o algoritmo K-Nearest Neighbors Classifier para analisar e prever a concessão de crédito em instituições financeiras. O modelo é treinado usando um conjunto de dados que inclui informações sobre os solicitantes de crédito, como idade, propriedade da casa, intenção de empréstimo e histórico de inadimplência.

## Tecnologias Utilizadas

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn

## Estrutura do Projeto

1. **Importação de Bibliotecas**: Carregamos as bibliotecas necessárias para manipulação de dados, visualização e modelagem.
  
2. **Carregamento de Dados**: O arquivo `credit_risk_dataset.csv` é lido utilizando a biblioteca Pandas. Os dados contêm informações relevantes para a análise de crédito.

3. **Análise Exploratória de Dados (EDA)**: Realizamos uma análise preliminar para entender melhor os dados, incluindo a verificação de valores ausentes e duplicados, bem como a visualização de algumas características dos dados.

4. **Pré-processamento dos Dados**:
    - **Codificação de Variáveis Categóricas**: As variáveis categóricas são convertidas em valores numéricos usando o `OrdinalEncoder`.
    - **Imputação de Dados Faltantes**: O `KNNImputer` é utilizado para preencher valores ausentes.
    - **Normalização dos Dados**: Os dados são escalonados usando `StandardScaler` para melhorar a performance do modelo.

5. **Divisão dos Dados**: Os dados são divididos em conjuntos de treinamento e teste (80% treino e 20% teste).

6. **Treinamento do Modelo**: O modelo KNeighborsClassifier é treinado com os dados de treinamento.

7. **Predições e Avaliação**: O modelo faz previsões sobre um novo conjunto de dados (`novos_clientes.csv`) e exibe a previsão de aprovação e a probabilidade correspondente.

## Instruções de Uso

### Pré-requisitos

- **Python** (versão 3.12 ou superior)

### Passo a Passo

1. **Clone o Repositório**:
   ```bash
   git clone https://github.com/yasmincsouza/kNN-analise-credito-instituicao-financeira

2. **Instale as Dependências**:
   ```bash
   pip install numpy pandas matplotlib seaborn scikit-learn missingno

2. **Execute o Script**:
   python kNN_analise_credito.py