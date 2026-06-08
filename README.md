# AgroSat — Sistema de Monitoramento de Risco de Seca no Agronegócio Brasileiro

## Integrantes
João Pedro Ferreira Alencar - rm573473 
Alisson vinicius de Souza rabelo texeira - rm573512 
Lucas Michels Kuntz - rm570050 

---

## Proposta

O Brasil é responsável por grande parte da produção mundial de soja, cana-de-açúcar, milho e algodão. A seca é o principal fator de risco climático para essas culturas, causando perdas bilionárias anualmente.

O **AgroSat** usa dados reais coletados de satélites da NASA para monitorar precipitação e temperatura nas principais regiões agrícolas do Brasil. A partir desses dados, o sistema aplica Machine Learning para classificar o risco de seca em três níveis — **Baixo**, **Médio** e **Alto** — gerando visualizações que permitem ao produtor e ao gestor agro entender o cenário climático de cada região.

---

## Tecnologias Utilizadas

| Tecnologia | Versão | Uso |
|------------|--------|-----|
| Python | 3.10+ | Linguagem principal |
| Pandas | 2.0+ | Manipulação de dados |
| Matplotlib | 3.7+ | Visualizações estáticas (CLI) |
| Seaborn | 0.12+ | Visualizações estatísticas |
| Scikit-learn | 1.3+ | Modelo de classificação (Random Forest) |
| Requests | 2.28+ | Consumo da API NASA POWER |
| NumPy | 1.24+ | Operações numéricas |
| Streamlit | 1.32+ | Dashboard visual interativo |
| Plotly | 5.18+ | Gráficos interativos no dashboard |

**API de dados:** [NASA POWER](https://power.larc.nasa.gov/) — dados climáticos de satélite, gratuita e sem autenticação.

---

## Estrutura do Repositório

```
AgroSat/
│
├── main.py                  # Script CLI — ponto de entrada (terminal)
├── dashboard.py             # Dashboard visual interativo (Streamlit)
│
├── src/
│   ├── coletor_nasa.py      # Coleta de dados via API NASA POWER
│   ├── processador.py       # Limpeza, agregação e rotulagem dos dados
│   ├── visualizador.py      # Geração dos gráficos estáticos (PNG)
│   └── modelo_ml.py         # Treinamento e avaliação do modelo ML
│
├── data/
│   ├── dados_nasa_AAAA_AAAA.csv  # Cache por período (ex: dados_nasa_2024_2024.csv)
│   └── dados_mensais.csv          # Dados mensais processados
│
├── graficos/                # 6 gráficos PNG gerados pelo CLI
│
├── requirements.txt
└── README.md
```

---

## Como Executar

### Pré-requisitos
- Python 3.10 ou superior instalado
- Conexão com a internet (para a primeira execução — após isso, usa cache local)

### 1. Clone o repositório
```bash
git clone https://github.com/SEU_USUARIO/AgroSat.git
cd AgroSat
```

### 2. Instale as dependências
```bash
pip install -r requirements.txt
```

### 3a. Execute via terminal (CLI)
```bash
# Ano padrão (2024)
python main.py

# Ano específico
python main.py 2022

# Período (ex: 2021 a 2023)
python main.py 2021 2023
```

### 3b. Execute o Dashboard Visual (Streamlit)
```bash
streamlit run dashboard.py
```
Acesse em: **http://localhost:8501**

Na **primeira execução**, o sistema fará requisições à API da NASA (~30 a 60 segundos). Nas execuções seguintes, usa os dados em cache (instantâneo).

---

## O que o sistema faz

1. **Coleta dados reais** de precipitação e temperatura para 6 regiões agrícolas do Brasil (2024) via API NASA POWER
2. **Processa e limpa** os dados diários com Pandas, removendo outliers e valores ausentes
3. **Agrega** os dados em métricas mensais: precipitação total, temperatura média, dias secos
4. **Rotula** cada mês/região com nível de risco de seca (Baixo/Médio/Alto) via regras agronômicas
5. **Gera 6 gráficos** salvos em `/graficos`
6. **Treina um Random Forest** para classificar o risco de seca automaticamente
7. **Exibe diagnóstico** do risco atual de cada região no terminal

---

## Regiões Monitoradas

| Região | Cultura Principal | Coordenadas |
|--------|-------------------|-------------|
| Sorriso-MT | Soja | -12.55°, -55.72° |
| Ribeirão Preto-SP | Cana-de-açúcar | -21.17°, -47.81° |
| Rondonópolis-MT | Algodão | -16.47°, -54.64° |
| Passo Fundo-RS | Trigo | -28.26°, -52.40° |
| Barreiras-BA | Grãos (MATOPIBA) | -12.15°, -44.99° |
| Uberlândia-MG | Milho | -18.91°, -48.28° |

---

## Links

- **Repositório GitHub:** https://github.com/opedroalencar/AgroSat

---

## Referências

- NASA POWER API Documentation: https://power.larc.nasa.gov/docs/
- INMET — Instituto Nacional de Meteorologia: https://www.inmet.gov.br
- Embrapa Monitoramento por Satélite: https://www.embrapa.br
