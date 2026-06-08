"""
AgroSat — Sistema de Monitoramento de Risco de Seca no Agronegócio Brasileiro
Utiliza dados reais da NASA POWER API para análise climática e classificação
de risco de seca por regiões agrícolas com Machine Learning.

FIAP — Global Solution 2026.1

USO:
    python main.py            -> executa com o ano padrão (2024)
    python main.py 2022       -> analisa o ano de 2022
    python main.py 2021 2023  -> analisa o período 2021 a 2023
"""

import os
import sys
import pandas as pd

# Adiciona a pasta /src ao caminho para importar os módulos
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from coletor_nasa import coletar_todas_regioes
from processador import processar
from visualizador import gerar_todos_graficos
from modelo_ml import executar_ml

ANO_MINIMO = 2015
ANO_MAXIMO = 2024


def exibir_cabecalho():
    """Exibe o cabeçalho do sistema."""
    print("=" * 65)
    print("   AgroSat - Monitoramento de Risco de Seca no Agronegocio")
    print("   Fonte de dados: NASA POWER API (satelite)")
    print("   FIAP - Global Solution 2026.1")
    print("=" * 65)


def selecionar_periodo():
    """
    Determina o período de análise.
    Aceita argumentos de linha de comando ou pede ao usuário interativamente.
    """
    args = sys.argv[1:]

    # Modo linha de comando: python main.py 2022  ou  python main.py 2021 2023
    if len(args) == 1 and args[0].isdigit():
        ano = int(args[0])
        return ano, ano

    if len(args) == 2 and args[0].isdigit() and args[1].isdigit():
        return int(args[0]), int(args[1])

    # Modo interativo: pergunta ao usuário
    print(f"\nQual ano deseja analisar? ({ANO_MINIMO}-{ANO_MAXIMO})")
    print("  - Digite um ano (ex: 2022) para analisar apenas aquele ano")
    print("  - Digite dois anos separados por espaco (ex: 2021 2023) para um periodo")
    print(f"  - Pressione Enter para usar o padrao (2024)")

    entrada = input("\nAno(s): ").strip()

    if not entrada:
        return 2024, 2024

    partes = entrada.split()
    try:
        if len(partes) == 1:
            ano = int(partes[0])
            if ANO_MINIMO <= ano <= ANO_MAXIMO:
                return ano, ano
            else:
                print(f"Ano fora do intervalo valido ({ANO_MINIMO}-{ANO_MAXIMO}). Usando 2024.")
                return 2024, 2024
        elif len(partes) == 2:
            ano_i, ano_f = int(partes[0]), int(partes[1])
            if ano_i <= ano_f and ANO_MINIMO <= ano_i and ano_f <= ANO_MAXIMO:
                return ano_i, ano_f
            else:
                print("Periodo invalido. Usando 2024.")
                return 2024, 2024
    except ValueError:
        print("Entrada invalida. Usando 2024.")
        return 2024, 2024


def caminho_cache(ano_inicio, ano_fim):
    """Gera o nome do arquivo de cache baseado no período escolhido."""
    pasta = os.path.join(os.path.dirname(__file__), "data")
    nome  = f"dados_nasa_{ano_inicio}_{ano_fim}.csv"
    return os.path.join(pasta, nome)


def carregar_ou_coletar_dados(ano_inicio, ano_fim):
    """
    Tenta carregar dados do cache local para o período escolhido.
    Se não existir, coleta da API NASA e salva em cache.
    """
    arquivo = caminho_cache(ano_inicio, ano_fim)

    if os.path.exists(arquivo):
        nome_curto = os.path.basename(arquivo)
        print(f"\nCache encontrado: data/{nome_curto}")
        df = pd.read_csv(arquivo, parse_dates=["data"])
        print(f"  {len(df)} registros carregados.")
        return df

    print(f"\nSem cache para {ano_inicio}-{ano_fim}. Coletando da NASA...")
    df = coletar_todas_regioes(ano_inicio=ano_inicio, ano_fim=ano_fim)

    if df.empty:
        print("\nERRO: Nao foi possivel coletar os dados. Verifique a internet.")
        sys.exit(1)

    os.makedirs(os.path.dirname(arquivo), exist_ok=True)
    df.to_csv(arquivo, index=False)
    print(f"  Dados salvos em cache: data/{os.path.basename(arquivo)}")
    return df


def exibir_resumo(df_mensal, ano_inicio, ano_fim):
    """Imprime um resumo dos dados processados no terminal."""
    periodo = f"{ano_inicio}" if ano_inicio == ano_fim else f"{ano_inicio} a {ano_fim}"
    print(f"\n=== Resumo dos Dados Processados — {periodo} ===")
    print(f"  Regioes analisadas : {df_mensal['regiao'].nunique()}")
    print(f"  Total de registros : {len(df_mensal)} (mensais por regiao)")

    stats  = df_mensal["precipitacao_total_mm"].describe()
    stats_t = df_mensal["temperatura_media_c"].describe()

    print(f"\n  Precipitacao (mm/mes): min={stats['min']:.1f}  "
          f"media={stats['mean']:.1f}  max={stats['max']:.1f}")
    print(f"  Temperatura  (C)    : min={stats_t['min']:.1f}  "
          f"media={stats_t['mean']:.1f}  max={stats_t['max']:.1f}")


def main():
    """Função principal — orquestra todo o pipeline."""
    exibir_cabecalho()

    # Seleção de período (interativo ou linha de comando)
    ano_inicio, ano_fim = selecionar_periodo()
    periodo = f"{ano_inicio}" if ano_inicio == ano_fim else f"{ano_inicio} a {ano_fim}"
    print(f"\nPeriodo selecionado: {periodo}")

    # ETAPA 1: Coleta de dados
    df_bruto = carregar_ou_coletar_dados(ano_inicio, ano_fim)

    # ETAPA 2: Processamento
    df_mensal, _ = processar(df_bruto)
    exibir_resumo(df_mensal, ano_inicio, ano_fim)

    # Exibe ISA no terminal
    print("\n=== ISA — Indice de Seca AgroSat (mes mais recente) ===")
    mais_rec = df_mensal.sort_values("mes").groupby("regiao").last().reset_index()
    for _, row in mais_rec.iterrows():
        nome = row["regiao"].split(" (")[0] if " (" in row["regiao"] else row["regiao"]
        barra = "#" * int(row["isa"] / 5)
        print(f"  {nome:<30} ISA={row['isa']:>5.1f} [{row['isa_categoria']:<8}] {barra}")

    # Salva os dados mensais processados
    caminho_mensal = os.path.join(os.path.dirname(__file__), "data", "dados_mensais.csv")
    df_mensal.to_csv(caminho_mensal, index=False)
    print(f"\n  Dados mensais salvos: data/dados_mensais.csv")

    # ETAPA 3: Visualizações estáticas
    gerar_todos_graficos(df_mensal)

    # ETAPA 4: Machine Learning
    executar_ml(df_mensal)

    print("\n" + "=" * 65)
    print("  Execucao concluida!")
    print("  Graficos em: /graficos")
    print("  Para o dashboard visual: streamlit run dashboard.py")
    print("=" * 65)


if __name__ == "__main__":
    main()
