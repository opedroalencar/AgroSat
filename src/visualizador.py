"""
Módulo de visualização — gera e salva os gráficos estáticos do projeto AgroSat.
Todos os gráficos são salvos na pasta /graficos como arquivos PNG.
"""

import matplotlib
matplotlib.use("Agg")   # renderiza sem abrir janela
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os

# Diretório onde os gráficos serão salvos
PASTA_GRAFICOS = os.path.join(os.path.dirname(os.path.dirname(__file__)), "graficos")

# Paleta de cores do projeto (verde satelital / âmbar colheita / vermelho perigo)
CORES_RISCO = {"Baixo": "#00C48C", "Medio": "#FFA500", "Alto": "#FF4B4B"}

# Estilo visual padrão
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "figure.facecolor": "#F8F9FA",
    "axes.facecolor": "#FFFFFF",
})

NOMES_MESES = ["Jan", "Fev", "Mar", "Abr", "Mai", "Jun",
               "Jul", "Ago", "Set", "Out", "Nov", "Dez"]


def _nome_curto(nome_regiao):
    """Extrai 'Sorriso-MT' de 'Sorriso-MT (Soja)'."""
    return nome_regiao.split(" (")[0] if " (" in nome_regiao else nome_regiao


def _salvar(nome_arquivo, fig):
    """Salva a figura no diretório de gráficos."""
    os.makedirs(PASTA_GRAFICOS, exist_ok=True)
    caminho = os.path.join(PASTA_GRAFICOS, nome_arquivo)
    fig.savefig(caminho, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Grafico salvo: graficos/{nome_arquivo}")


# ─────────────────────────────────────────────────────────
# Gráfico 1 — Série temporal de precipitação
# ─────────────────────────────────────────────────────────
def grafico_precipitacao_temporal(df_mensal):
    """Linha por região, precipitação mensal ao longo dos meses."""
    print("\n  Gerando Grafico 1: Precipitacao mensal por regiao...")

    fig, ax = plt.subplots(figsize=(14, 6))
    regioes  = df_mensal["regiao"].unique()
    palette  = sns.color_palette("tab10", n_colors=len(regioes))

    for i, regiao in enumerate(regioes):
        dados = df_mensal[df_mensal["regiao"] == regiao].sort_values("mes")
        ax.plot(
            dados["mes"],
            dados["precipitacao_total_mm"],
            marker="o", linewidth=2, markersize=5,
            label=_nome_curto(regiao), color=palette[i],
        )

    ax.set_title("Precipitacao Mensal por Regiao Agricola — Brasil", fontweight="bold")
    ax.set_xlabel("Mes")
    ax.set_ylabel("Precipitacao Total (mm)")
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(NOMES_MESES)
    ax.axhline(y=100, color="gray", linestyle="--", linewidth=1, alpha=0.6)
    ax.axhline(y=50,  color="red",  linestyle=":",  linewidth=1, alpha=0.4)
    ax.legend(loc="upper right", fontsize=9)

    _salvar("01_precipitacao_temporal.png", fig)


# ─────────────────────────────────────────────────────────
# Gráfico 2 — Heatmap de risco (regiões × meses)
# ─────────────────────────────────────────────────────────
def grafico_heatmap_risco(df_mensal):
    """Heatmap: linhas = regiões, colunas = meses, cor = nível de risco."""
    print("  Gerando Grafico 2: Heatmap de risco de seca...")

    mapa_num = {"Baixo": 0, "Medio": 1, "Alto": 2}
    df_plot = df_mensal.copy()
    df_plot["regiao_curta"] = df_plot["regiao"].apply(_nome_curto)
    df_plot["risco_num"]    = df_plot["risco_seca"].map(mapa_num)

    anos_unicos = sorted(df_plot["ano"].unique())
    if len(anos_unicos) > 1:
        df_plot["linha"] = df_plot["regiao_curta"] + " (" + df_plot["ano"].astype(str) + ")"
        index_col = "linha"
    else:
        index_col = "regiao_curta"

    pivot = df_plot.pivot_table(
        index=index_col, columns="mes",
        values="risco_num", aggfunc="max"
    )
    pivot.columns = [NOMES_MESES[m - 1] for m in pivot.columns]

    fig, ax = plt.subplots(figsize=(13, 5))
    cmap = sns.color_palette(["#00C48C", "#FFA500", "#FF4B4B"], as_cmap=True)

    sns.heatmap(
        pivot, ax=ax, cmap=cmap,
        linewidths=0.5, linecolor="white",
        annot=True, fmt=".0f",
        cbar_kws={"ticks": [0, 1, 2], "label": "0=Baixo  1=Medio  2=Alto"},
        vmin=0, vmax=2,
    )
    ax.set_title("Heatmap de Risco de Seca por Regiao e Mes", fontweight="bold")
    ax.set_xlabel("Mes")
    ax.set_ylabel("Regiao")
    ax.tick_params(axis="y", rotation=0)

    _salvar("02_heatmap_risco.png", fig)


# ─────────────────────────────────────────────────────────
# Gráfico 3 — Distribuição de risco por região (barras empilhadas)
# ─────────────────────────────────────────────────────────
def grafico_distribuicao_risco(df_mensal):
    """Barras empilhadas: percentual de meses em cada nível de risco por região."""
    print("  Gerando Grafico 3: Distribuicao de risco por regiao...")

    df_plot = df_mensal.copy()
    df_plot["regiao_curta"] = df_plot["regiao"].apply(_nome_curto)

    contagem = (
        df_plot.groupby(["regiao_curta", "risco_seca"])
        .size().unstack(fill_value=0)
    )
    for col in ["Baixo", "Medio", "Alto"]:
        if col not in contagem.columns:
            contagem[col] = 0
    contagem = contagem[["Baixo", "Medio", "Alto"]]
    pct = contagem.div(contagem.sum(axis=1), axis=0) * 100

    fig, ax = plt.subplots(figsize=(12, 6))
    bottom = np.zeros(len(pct))

    for nivel in ["Baixo", "Medio", "Alto"]:
        valores = pct[nivel].values
        ax.bar(
            pct.index, valores, bottom=bottom,
            label=nivel, color=CORES_RISCO[nivel],
            edgecolor="white", linewidth=0.7,
        )
        for j, (v, b) in enumerate(zip(valores, bottom)):
            if v > 8:
                ax.text(j, b + v / 2, f"{v:.0f}%",
                        ha="center", va="center",
                        fontsize=9, fontweight="bold", color="white")
        bottom += valores

    ax.set_title("Distribuicao de Risco de Seca por Regiao Agricola", fontweight="bold")
    ax.set_ylabel("Percentual de Meses (%)")
    ax.set_xlabel("Regiao")
    ax.set_ylim(0, 105)
    ax.legend(title="Nivel de Risco", loc="upper right")
    ax.tick_params(axis="x", rotation=25)

    _salvar("03_distribuicao_risco.png", fig)


# ─────────────────────────────────────────────────────────
# Gráfico 4 — Scatter: temperatura × precipitação
# ─────────────────────────────────────────────────────────
def grafico_scatter_temp_chuva(df_mensal):
    """Dispersão: temperatura no eixo X, precipitação no eixo Y, cor = risco."""
    print("  Gerando Grafico 4: Dispersao precipitacao x temperatura...")

    fig, ax = plt.subplots(figsize=(10, 6))

    for nivel, cor in CORES_RISCO.items():
        subset = df_mensal[df_mensal["risco_seca"] == nivel]
        ax.scatter(
            subset["temperatura_media_c"],
            subset["precipitacao_total_mm"],
            c=cor, label=f"Risco {nivel}",
            alpha=0.75, edgecolors="white", linewidth=0.5, s=60,
        )

    ax.set_title("Relacao Temperatura x Precipitacao por Nivel de Risco", fontweight="bold")
    ax.set_xlabel("Temperatura Media (°C)")
    ax.set_ylabel("Precipitacao Total Mensal (mm)")
    ax.legend(title="Nivel de Risco")
    ax.axhline(y=100, color="gray", linestyle="--", linewidth=1, alpha=0.5)
    ax.axhline(y=50,  color="red",  linestyle=":",  linewidth=1, alpha=0.4)

    _salvar("04_scatter_temp_chuva.png", fig)


# ─────────────────────────────────────────────────────────
# Pipeline completo
# ─────────────────────────────────────────────────────────
def gerar_todos_graficos(df_mensal):
    """Executa todos os gráficos estáticos em sequência."""
    print("\n=== Gerando visualizacoes estaticas ===")
    os.makedirs(PASTA_GRAFICOS, exist_ok=True)

    grafico_precipitacao_temporal(df_mensal)
    grafico_heatmap_risco(df_mensal)
    grafico_distribuicao_risco(df_mensal)
    grafico_scatter_temp_chuva(df_mensal)

    print(f"\n  Todos os graficos salvos em: /graficos")
