"""
Módulo de processamento e limpeza dos dados climáticos.
Transforma os dados brutos da NASA em métricas mensais úteis
e gera os rótulos de risco de seca para o modelo de ML.
"""

import pandas as pd
import numpy as np


def limpar_dados(df_bruto):
    """
    Remove duplicatas, trata valores ausentes e garante tipos corretos.
    """
    print("\n=== Limpeza e processamento dos dados ===")

    df = df_bruto.copy()

    # Remove linhas completamente duplicadas
    antes = len(df)
    df = df.drop_duplicates()
    print(f"  Duplicatas removidas: {antes - len(df)}")

    # Garante que a coluna 'data' é do tipo datetime
    df["data"] = pd.to_datetime(df["data"])

    # Extrai mês e ano para agregações futuras
    df["mes"] = df["data"].dt.month
    df["ano"] = df["data"].dt.year
    df["mes_ano"] = df["data"].dt.to_period("M")

    # Remove valores fora de faixas fisicamente possíveis para o Brasil
    registros_antes = len(df)
    df = df[
        (df["temperatura_media_c"].between(-5, 50)) &
        (df["temperatura_max_c"].between(-5, 55)) &
        (df["precipitacao_mm"].between(0, 500))
    ]
    print(f"  Registros fora de faixa removidos: {registros_antes - len(df)}")
    print(f"  Registros validos: {len(df)}")

    return df


def agregar_por_mes(df):
    """
    Agrega os dados diários em métricas mensais por região.
    Calcula precipitação total, temperatura média e dias sem chuva no mês.
    """
    print("\n  Agregando dados diarios em metricas mensais...")

    # Conta dias sem chuva (precipitação < 1 mm é considerado dia seco)
    df["dia_seco"] = (df["precipitacao_mm"] < 1.0).astype(int)

    df_mensal = (
        df.groupby(["regiao", "ano", "mes"])
        .agg(
            precipitacao_total_mm=("precipitacao_mm", "sum"),
            temperatura_media_c=("temperatura_media_c", "mean"),
            temperatura_max_media_c=("temperatura_max_c", "mean"),
            dias_secos=("dia_seco", "sum"),
            total_dias=("dia_seco", "count"),
        )
        .reset_index()
    )

    # Percentual de dias secos no mês
    df_mensal["pct_dias_secos"] = (
        df_mensal["dias_secos"] / df_mensal["total_dias"] * 100
    ).round(1)

    df_mensal["precipitacao_total_mm"]  = df_mensal["precipitacao_total_mm"].round(1)
    df_mensal["temperatura_media_c"]    = df_mensal["temperatura_media_c"].round(2)
    df_mensal["temperatura_max_media_c"] = df_mensal["temperatura_max_media_c"].round(2)

    print(f"  Registros mensais gerados: {len(df_mensal)}")
    return df_mensal


def rotular_risco_seca(df_mensal):
    """
    Cria o rótulo de risco de seca com base em regras agronômicas:

    ALTO   — precipitação < 50 mm/mês OU > 70% de dias secos
    MEDIO  — precipitação entre 50 e 100 mm/mês OU entre 40% e 70% de dias secos
    BAIXO  — precipitação >= 100 mm/mês E < 40% de dias secos

    Faixas baseadas em limiares da literatura de monitoramento de secas
    agrícolas no Brasil (INMET / Embrapa).
    """
    print("\n  Classificando risco de seca por regras agronomicas...")

    def classificar(row):
        chuva   = row["precipitacao_total_mm"]
        pct_seco = row["pct_dias_secos"]

        if chuva < 50 or pct_seco > 70:
            return "Alto"
        elif chuva < 100 or pct_seco > 40:
            return "Medio"
        else:
            return "Baixo"

    df_mensal["risco_seca"] = df_mensal.apply(classificar, axis=1)

    # Versão sem acento para compatibilidade em labels de gráficos
    df_mensal["risco_display"] = df_mensal["risco_seca"].replace({"Medio": "Medio"})

    # Converte para valor numérico para o modelo de ML
    mapa_risco = {"Baixo": 0, "Medio": 1, "Alto": 2}
    df_mensal["risco_numerico"] = df_mensal["risco_seca"].map(mapa_risco)

    contagem = df_mensal["risco_seca"].value_counts()
    print("\n  Distribuicao de risco de seca:")
    for nivel, qtd in contagem.items():
        print(f"    {nivel}: {qtd} registros ({qtd/len(df_mensal)*100:.1f}%)")

    return df_mensal


def calcular_isa(df_mensal):
    """
    Calcula o ISA — Índice de Seca AgroSat — para cada registro mensal.

    O ISA é um score de 0 a 100 que resume o risco de seca em um único número:
        0–29  → Normal    (condições favoráveis)
        30–54 → Atenção   (déficit hídrico leve)
        55–74 → Alerta    (déficit hídrico moderado)
        75–100 → Crítico  (seca severa)

    Componentes e pesos:
        50% Precipitação  — quanto menos chuva, maior o ISA
        35% Dias secos    — percentual de dias sem chuva
        15% Temperatura   — temperatura acima de 20°C eleva o ISA gradualmente

    Fundamentado nos limiares da Embrapa para déficit hídrico em culturas de grãos.
    """
    df = df_mensal.copy()

    def _isa_linha(row):
        # Componente chuva: 0mm = 100pts, 200mm+ = 0pts
        comp_chuva = max(0.0, min(100.0, 100.0 - (row["precipitacao_total_mm"] / 200.0 * 100.0)))
        # Componente dias secos: igual ao percentual de dias secos (0-100)
        comp_seco  = min(100.0, row["pct_dias_secos"])
        # Componente temperatura: 20°C → 0pts, 35°C → 50pts
        comp_temp  = max(0.0, min(50.0, (row["temperatura_media_c"] - 20.0) / 15.0 * 50.0))
        # ISA ponderado
        isa = comp_chuva * 0.50 + comp_seco * 0.35 + comp_temp * 0.15
        return round(min(100.0, max(0.0, isa)), 1)

    def _cat_isa(v):
        if v < 30:  return "Normal"
        if v < 55:  return "Atencao"
        if v < 75:  return "Alerta"
        return "Critico"

    df["isa"]          = df.apply(_isa_linha, axis=1)
    df["isa_categoria"] = df["isa"].apply(_cat_isa)

    print("\n  ISA medio por regiao (mes mais recente):")
    mais_rec = df.sort_values("mes").groupby("regiao").last()
    for regiao, row in mais_rec.iterrows():
        nome = regiao.split(" (")[0] if " (" in regiao else regiao
        print(f"    {nome:<30} ISA={row['isa']:>5.1f}  [{row['isa_categoria']}]")

    return df


def processar(df_bruto):
    """
    Pipeline completo: limpeza → agregação → rotulagem → ISA.
    Retorna (df_mensal, df_diario).
    """
    df_limpo  = limpar_dados(df_bruto)
    df_mensal = agregar_por_mes(df_limpo)
    df_rot    = rotular_risco_seca(df_mensal)
    df_final  = calcular_isa(df_rot)
    return df_final, df_limpo
