"""
AgroSat — Dashboard Visual Interativo
Plataforma web para exploração dos dados climáticos e resultados de ML.

Execute com:  streamlit run dashboard.py
"""

import os
import sys
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from coletor_nasa import coletar_todas_regioes, REGIOES_AGRICOLAS
from processador import processar
from modelo_ml import preparar_dados_ml, treinar_modelo, FEATURES
from sklearn.metrics import accuracy_score, confusion_matrix

from ui.components import (
    load_css,
    sidebar_subtitle,
    sidebar_info,
    sidebar_footer,
    hero,
    banner_critico,
    kpi_card,
    kpi_grid,
    isa_legend,
    region_card_isa,
    region_card_risco,
    region_card_ml,
)

# ──────────────────────────────────────────────────────────
# CONFIGURAÇÃO DA PÁGINA
# ──────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AgroSat",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────
# CSS — tema satelital escuro, fonte Space Grotesk (em ui/styles.css)
# ──────────────────────────────────────────────────────────
st.markdown(load_css(), unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────
# PALETAS E CONSTANTES
# ──────────────────────────────────────────────────────────
COR_RISCO = {"Baixo": "#00C48C", "Medio": "#FFA500", "Alto": "#FF4B4B"}
COR_ISA   = {"Normal": "#00C48C", "Atencao": "#FFD000", "Alerta": "#FF7800", "Critico": "#FF2222"}
NOMES_MESES = ["Jan","Fev","Mar","Abr","Mai","Jun","Jul","Ago","Set","Out","Nov","Dez"]

# Configuração base de layout Plotly sem `legend` — evita conflito de chave
_PLOTLY_BASE = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(255,255,255,0.03)",
    font=dict(family="Space Grotesk, sans-serif", color="#C8D6E2"),
    margin=dict(l=20, r=20, t=40, b=20),
)
_LEGEND_BASE = dict(bgcolor="rgba(255,255,255,0.05)",
                    bordercolor="rgba(255,255,255,0.1)", borderwidth=1)


def _layout(**kwargs):
    """
    Monta o dicionário de layout Plotly sem conflito de chave.
    Mescla 'legend' extra com a base em vez de duplicar.
    """
    cfg = dict(_PLOTLY_BASE)
    legend_extra = kwargs.pop("legend", {})
    cfg["legend"] = {**_LEGEND_BASE, **legend_extra}
    cfg.update(kwargs)
    return cfg


def _nome_curto(r):
    return r.split(" (")[0] if " (" in r else r


# ──────────────────────────────────────────────────────────
# CACHE
# ──────────────────────────────────────────────────────────
_REGIOES_ESPERADAS = set(REGIOES_AGRICOLAS.keys())


def _cache_valido(df):
    """Cache é válido se contém todas as regiões esperadas e não está vazio."""
    if df is None or df.empty or "regiao" not in df.columns:
        return False
    return _REGIOES_ESPERADAS.issubset(set(df["regiao"].unique()))


def _listar_csvs():
    """Lista todos os dados_nasa_*.csv em /data, ordenados do mais recente
    pro mais antigo (mtime). Retorna lista de dicts com path/nome/anos/mtime."""
    import glob, re
    pasta = os.path.join(os.path.dirname(__file__), "data")
    if not os.path.isdir(pasta):
        return []
    paths = sorted(
        glob.glob(os.path.join(pasta, "dados_nasa_*.csv")),
        key=os.path.getmtime,
        reverse=True,
    )
    fallback = os.path.join(pasta, "dados_nasa_brutos.csv")
    if os.path.exists(fallback) and fallback not in paths:
        paths.append(fallback)
    out = []
    for p in paths:
        nome = os.path.basename(p)
        m = re.match(r"dados_nasa_(\d{4})_(\d{4})\.csv", nome)
        anos = f"{m.group(1)}–{m.group(2)}" if m else "—"
        out.append({
            "path": p,
            "nome": nome,
            "anos": anos,
            "mtime": os.path.getmtime(p),
        })
    return out


@st.cache_data(show_spinner=False)
def _ler_cache(caminho: str | None = None):
    """Lê um CSV de cache. Se `caminho` for dado, lê esse arquivo específico
    (validando as 6 regiões). Se None, cai no auto-discover: pega o CSV
    mais recente que contenha todas as regiões esperadas."""
    if caminho:
        try:
            df = pd.read_csv(caminho, parse_dates=["data"])
        except Exception:
            return None
        return df if _cache_valido(df) else None

    # auto-discover (comportamento antigo, mantido como fallback)
    for c in _listar_csvs():
        try:
            df = pd.read_csv(c["path"], parse_dates=["data"])
        except Exception:
            continue
        if _cache_valido(df):
            return df
    return None


@st.cache_data(show_spinner=False)
def _coletar_completo(ano_inicio=2024, ano_fim=2024):
    """Coleta SEMPRE as 6 regiões — nunca subset, pra não corromper o cache."""
    return coletar_todas_regioes(ano_inicio=ano_inicio, ano_fim=ano_fim,
                                  regioes=list(_REGIOES_ESPERADAS))


@st.cache_data(show_spinner=False)
def _processar(df):
    df = df.copy()
    df["data"] = pd.to_datetime(df["data"])
    return processar(df)


@st.cache_resource(show_spinner=False)
def _treinar(df):
    X_tr, X_te, y_tr, y_te, enc = preparar_dados_ml(df)
    modelo = treinar_modelo(X_tr, y_tr)
    return modelo, enc, X_tr, X_te, y_tr, y_te


# ──────────────────────────────────────────────────────────
# SIDEBAR
# ──────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🛰️ AgroSat")
    st.markdown(
        sidebar_subtitle("Monitoramento de Seca via Satélite"),
        unsafe_allow_html=True,
    )
    st.divider()

    st.markdown(
        sidebar_info(
            titulo="Fonte: NASA POWER",
            texto="Dados diários de satélite — período definido pelo CSV escolhido abaixo.",
        ),
        unsafe_allow_html=True,
    )
    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("#### Arquivo de dados")
    _csvs_disponiveis = _listar_csvs()
    csv_escolhido = None
    if _csvs_disponiveis:
        _labels = [
            f"{c['anos']}  ·  {c['nome']}" if c['anos'] != "—" else c['nome']
            for c in _csvs_disponiveis
        ]
        _idx = st.selectbox(
            "CSV",
            options=list(range(len(_csvs_disponiveis))),
            format_func=lambda i: _labels[i],
            index=0,
            label_visibility="collapsed",
        )
        csv_escolhido = _csvs_disponiveis[_idx]["path"]
    else:
        st.caption("Nenhum CSV em /data — será coletado da NASA POWER.")

    # Coleta manual de período arbitrário
    from datetime import date as _date
    _ano_atual = _date.today().year
    with st.expander("➕ Coletar novo período"):
        st.caption("NASA POWER tem dados de 1981 até ontem. Coleta as 6 regiões.")
        col_a, col_b = st.columns(2)
        ano_ini_novo = col_a.number_input(
            "Ano início", min_value=1981, max_value=_ano_atual,
            value=_ano_atual - 1, step=1, key="ano_ini_novo",
        )
        ano_fim_novo = col_b.number_input(
            "Ano fim", min_value=1981, max_value=_ano_atual,
            value=_ano_atual - 1, step=1, key="ano_fim_novo",
        )
        sobrescrever = st.checkbox("Sobrescrever se já existir", value=False)

        if st.button("Coletar NASA POWER", use_container_width=True, key="btn_coletar_novo"):
            if int(ano_fim_novo) < int(ano_ini_novo):
                st.error("Ano fim deve ser maior ou igual ao ano início.")
            else:
                nome_arq = f"dados_nasa_{int(ano_ini_novo)}_{int(ano_fim_novo)}.csv"
                pasta_data = os.path.join(os.path.dirname(__file__), "data")
                destino = os.path.join(pasta_data, nome_arq)
                if os.path.exists(destino) and not sobrescrever:
                    st.warning(f"{nome_arq} já existe — marque 'sobrescrever' ou selecione no dropdown.")
                else:
                    with st.spinner(f"Coletando {int(ano_ini_novo)}–{int(ano_fim_novo)} da NASA POWER (pode levar 1–2 min por ano)..."):
                        try:
                            df_novo = coletar_todas_regioes(
                                ano_inicio=int(ano_ini_novo),
                                ano_fim=int(ano_fim_novo),
                                regioes=list(_REGIOES_ESPERADAS),
                            )
                        except Exception as e:
                            df_novo = None
                            st.error(f"Falha na coleta: {e}")
                    if df_novo is not None and not df_novo.empty:
                        os.makedirs(pasta_data, exist_ok=True)
                        df_novo.to_csv(destino, index=False)
                        st.cache_data.clear()
                        st.success(f"✓ {nome_arq} salvo ({len(df_novo)} linhas). Recarregando...")
                        st.rerun()
                    elif df_novo is not None:
                        st.error("Coleta retornou vazia.")

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("#### Regiões Monitoradas")
    todas = list(REGIOES_AGRICOLAS.keys())
    regioes_sel = st.multiselect("Regiões", todas, default=todas,
                                  label_visibility="collapsed",
                                  format_func=_nome_curto)
    if not regioes_sel:
        regioes_sel = todas

    st.divider()
    st.markdown(
        sidebar_footer(
            href="https://power.larc.nasa.gov/",
            link_label="NASA POWER API",
            linha2="FIAP — Global Solution 2026.1",
        ),
        unsafe_allow_html=True,
    )

# ──────────────────────────────────────────────────────────
# CABEÇALHO
# ──────────────────────────────────────────────────────────
st.markdown(
    hero(
        title="AgroSat",
        subtitle="Monitoramento de Risco de Seca &nbsp;·&nbsp; Regiões Agrícolas Brasileiras &nbsp;·&nbsp; Satélite NASA",
        pill="● ao vivo · NASA POWER",
    ),
    unsafe_allow_html=True,
)

# ──────────────────────────────────────────────────────────
# COLETA / CARREGAMENTO  (invalidação automática)
# ──────────────────────────────────────────────────────────
df_bruto = _ler_cache(csv_escolhido)

if df_bruto is None:
    st.info("Nenhum cache válido encontrado. Coletando dados da NASA POWER (todas as 6 regiões)...")
    with st.spinner("Coletando dados da NASA POWER API... (30–60s na 1ª vez)"):
        df_bruto = _coletar_completo()
    if df_bruto is None or df_bruto.empty:
        st.error("Não foi possível coletar dados. Verifique a conexão com a internet.")
        st.stop()
    pasta = os.path.join(os.path.dirname(__file__), "data")
    os.makedirs(pasta, exist_ok=True)
    ano_min = int(pd.to_datetime(df_bruto["data"]).dt.year.min())
    ano_max = int(pd.to_datetime(df_bruto["data"]).dt.year.max())
    nome = f"dados_nasa_{ano_min}_{ano_max}.csv"
    df_bruto.to_csv(os.path.join(pasta, nome), index=False)
    st.success(f"Coleta concluída ({len(df_bruto)} registros, {ano_min}–{ano_max}).")

# Filtra regiões selecionadas
df_bruto = df_bruto[df_bruto["regiao"].isin(regioes_sel)]

with st.spinner("Processando dados..."):
    df_mensal, _ = _processar(df_bruto)

# ──────────────────────────────────────────────────────────
# BANNER DE ALERTAS (ISA Crítico ou Alerta)
# ──────────────────────────────────────────────────────────
mais_rec = df_mensal.sort_values(["ano", "mes"]).groupby("regiao").last().reset_index()
criticos  = mais_rec[mais_rec["isa_categoria"].isin(["Critico", "Alerta"])]

if not criticos.empty:
    nomes_alert = ", ".join(criticos["regiao"].apply(_nome_curto).tolist())
    st.markdown(banner_critico(nomes=nomes_alert), unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────
# KPI CARDS
# ──────────────────────────────────────────────────────────
n_tot   = len(df_mensal)
n_alto  = (df_mensal["risco_seca"] == "Alto").sum()
n_medio = (df_mensal["risco_seca"] == "Medio").sum()
n_baixo = (df_mensal["risco_seca"] == "Baixo").sum()
isa_med = df_mensal["isa"].mean()
chuva_med = df_mensal["precipitacao_total_mm"].mean()
temp_med  = df_mensal["temperatura_media_c"].mean()
n_reg     = df_mensal["regiao"].nunique()

ano_min = int(df_mensal["ano"].min())
ano_max = int(df_mensal["ano"].max())
periodo_str = f"{ano_min}" if ano_min == ano_max else f"{ano_min}–{ano_max}"

isa_class = "warn" if 30 <= isa_med < 55 else ("danger" if isa_med >= 55 else "")
alto_pct  = (n_alto / n_tot * 100) if n_tot else 0
# média de meses em Alto por região (escala 0–12 faz sentido com o rótulo "Meses")
alto_por_regiao = (n_alto / n_reg) if n_reg else 0

isa_sub = "risco elevado" if isa_med >= 55 else ("atenção" if isa_med >= 30 else "condições normais")
alto_variant = "danger" if alto_pct >= 20 else ("warn" if alto_pct > 0 else "muted")

st.markdown(
    kpi_grid([
        kpi_card(label="Regiões",            value=f"{n_reg}",              unit="",     sub=f"período {periodo_str}",                       variant="muted"),
        kpi_card(label="Chuva / mês",        value=f"{chuva_med:.0f}",      unit="mm",   sub="média mensal",                                 variant=""),
        kpi_card(label="Temperatura",        value=f"{temp_med:.1f}",       unit="°C",   sub="média do período",                             variant="muted"),
        kpi_card(label="ISA Médio",          value=f"{isa_med:.1f}",        unit="/100", sub=isa_sub,                                        variant=isa_class),
        kpi_card(label="Meses Risco Alto",   value=f"{alto_por_regiao:.1f}", unit="/região", sub=f"{n_alto} de {n_tot} ({alto_pct:.0f}%)", variant=alto_variant),
    ]),
    unsafe_allow_html=True,
)

# ──────────────────────────────────────────────────────────
# ABAS
# ──────────────────────────────────────────────────────────
tab_prec, tab_isa, tab_risco, tab_dist, tab_corr, tab_ml, tab_raw = st.tabs([
    "Precipitação",
    "Índice ISA",
    "Mapa de Risco",
    "Distribuição",
    "Correlações",
    "Machine Learning",
    "Dados Brutos",
])


# ── TAB 1 — Precipitação ──────────────────────────────────
with tab_prec:
    st.markdown(f"#### Precipitação Mensal por Região · {periodo_str}")
    st.caption("Chuva acumulada em mm/mês. Linhas de referência: 100mm (seguro) e 50mm (crítico).")

    fig = go.Figure()
    palette = px.colors.qualitative.Safe
    regioes_unicas = df_mensal["regiao"].unique()

    for i, reg in enumerate(regioes_unicas):
        d = df_mensal[df_mensal["regiao"] == reg].sort_values("mes")
        cor = palette[i % len(palette)]
        fig.add_trace(go.Scatter(
            x=d["mes"], y=d["precipitacao_total_mm"],
            mode="lines+markers", name=_nome_curto(reg),
            line=dict(width=2.5, color=cor),
            marker=dict(size=7, color=cor, line=dict(width=1.5, color="#0A0E17")),
            hovertemplate="<b>%{fullData.name}</b><br>Mês %{x}: %{y:.1f} mm<extra></extra>",
        ))

    fig.add_hline(y=100, line_dash="dash", line_color="rgba(200,214,226,0.25)",
                  annotation_text="100mm", annotation_font_color="#7B92A8")
    fig.add_hline(y=50,  line_dash="dot",  line_color="rgba(255,75,75,0.3)",
                  annotation_text="50mm",  annotation_font_color="#FF6B6B")

    fig.update_layout(
        **_layout(
            xaxis=dict(tickmode="array", tickvals=list(range(1,13)),
                       ticktext=NOMES_MESES, gridcolor="rgba(255,255,255,0.06)"),
            yaxis=dict(title="Precipitação (mm)", gridcolor="rgba(255,255,255,0.06)"),
            legend=dict(title="Região"),
            height=440,
        )
    )
    st.plotly_chart(fig, use_container_width=True)


# ── TAB 2 — Índice ISA ────────────────────────────────────
with tab_isa:
    st.markdown("#### ISA — Índice de Seca AgroSat")
    st.caption(
        "Score 0–100 que combina precipitação (50%), dias secos (35%) e temperatura (15%). "
        "Quanto maior o ISA, maior o risco de déficit hídrico para as lavouras."
    )

    # Faixas explicativas — grid responsivo
    itens_leg = [
        ("Normal",  "#00D99A", "0–29"),
        ("Atenção", "#FFD000", "30–54"),
        ("Alerta",  "#FF7800", "55–74"),
        ("Crítico", "#FF2222", "75–100"),
    ]
    st.markdown(isa_legend(itens_leg), unsafe_allow_html=True)

    # Gauge charts por região (mês mais recente)
    n_regioes = len(mais_rec)
    cols_gauge = st.columns(min(3, n_regioes))

    for i, (_, row) in enumerate(mais_rec.iterrows()):
        col = cols_gauge[i % 3]
        isa_v  = row["isa"]
        cat    = row["isa_categoria"]
        nome   = _nome_curto(row["regiao"])
        cor_g  = COR_ISA.get(cat, "#7B92A8")

        fig_g = go.Figure(go.Indicator(
            mode="gauge+number",
            value=isa_v,
            number=dict(font=dict(color=cor_g, size=28), suffix=" ISA"),
            title=dict(text=nome, font=dict(color="#C8D6E2", size=13)),
            gauge=dict(
                axis=dict(range=[0, 100], tickcolor="#3A5060",
                          tickfont=dict(color="#3A5060", size=10)),
                bar=dict(color=cor_g, thickness=0.25),
                bgcolor="rgba(255,255,255,0.03)",
                borderwidth=0,
                steps=[
                    dict(range=[0,  30], color="rgba(0,196,140,0.12)"),
                    dict(range=[30, 55], color="rgba(255,208,0,0.10)"),
                    dict(range=[55, 75], color="rgba(255,120,0,0.12)"),
                    dict(range=[75,100], color="rgba(255,34,34,0.15)"),
                ],
                threshold=dict(line=dict(color=cor_g, width=3), thickness=0.8, value=isa_v),
            ),
        ))
        fig_g.update_layout(
            paper_bgcolor="rgba(255,255,255,0.03)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Space Grotesk, sans-serif"),
            height=220,
            margin=dict(l=15, r=15, t=40, b=10),
        )
        col.plotly_chart(fig_g, use_container_width=True)

        css_cat = {"Normal":"rc-normal","Atencao":"rc-atencao","Alerta":"rc-alerta","Critico":"rc-critico"}.get(cat,"")
        col.markdown(
            region_card_isa(
                cat=cat, cor=cor_g, css_cat=css_cat,
                chuva_mm=f"{row['precipitacao_total_mm']:.0f}",
                dias_secos=str(int(row['dias_secos'])),
                temp_c=f"{row['temperatura_media_c']:.1f}",
            ),
            unsafe_allow_html=True,
        )

    st.markdown("---")
    # Série temporal do ISA por região
    st.markdown("##### Evolução do ISA ao longo do ano")
    fig_isa_t = go.Figure()
    for i, reg in enumerate(regioes_unicas):
        d = df_mensal[df_mensal["regiao"] == reg].sort_values("mes")
        cor = px.colors.qualitative.Safe[i % 10]
        fig_isa_t.add_trace(go.Scatter(
            x=d["mes"], y=d["isa"],
            mode="lines+markers", name=_nome_curto(reg),
            line=dict(width=2, color=cor),
            marker=dict(size=6),
            hovertemplate="<b>%{fullData.name}</b><br>Mês %{x}: ISA %{y:.1f}<extra></extra>",
        ))
    # Faixas de fundo
    for y0, y1, cor_f in [(0,30,"rgba(0,196,140,0.05)"),
                           (30,55,"rgba(255,208,0,0.05)"),
                           (55,75,"rgba(255,120,0,0.07)"),
                           (75,100,"rgba(255,34,34,0.08)")]:
        fig_isa_t.add_hrect(y0=y0, y1=y1, fillcolor=cor_f, line_width=0)

    fig_isa_t.update_layout(
        **_layout(
            xaxis=dict(tickmode="array", tickvals=list(range(1,13)),
                       ticktext=NOMES_MESES, gridcolor="rgba(255,255,255,0.06)"),
            yaxis=dict(title="ISA (0–100)", range=[0,100],
                       gridcolor="rgba(255,255,255,0.06)"),
            legend=dict(title="Região"),
            height=380,
        )
    )
    st.plotly_chart(fig_isa_t, use_container_width=True)


# ── TAB 3 — Mapa de Risco ────────────────────────────────
with tab_risco:
    st.markdown("#### Heatmap de Risco de Seca por Região e Mês")
    st.caption("Verde = Baixo · Âmbar = Médio · Vermelho = Alto")

    mapa_num = {"Baixo": 0, "Medio": 1, "Alto": 2}
    df_h = df_mensal.copy()
    df_h["rc"]  = df_h["regiao"].apply(_nome_curto)
    df_h["rn"]  = df_h["risco_seca"].map(mapa_num)

    anos_unicos = sorted(df_h["ano"].unique())
    if len(anos_unicos) > 1:
        df_h["rc"] = df_h["rc"] + " (" + df_h["ano"].astype(str) + ")"

    pivot = df_h.pivot_table(index="rc", columns="mes", values="rn", aggfunc="max")
    for m in range(1, 13):
        if m not in pivot.columns:
            pivot[m] = np.nan
    pivot = pivot[sorted(pivot.columns)]
    pivot.columns = [NOMES_MESES[m - 1] for m in pivot.columns]

    inv_m = {0: "Baixo", 1: "Médio", 2: "Alto"}
    # compatível com pandas antigo e novo
    texto = pivot.apply(
        lambda col: col.map(lambda v: inv_m.get(int(v), "") if pd.notna(v) else "")
    )

    fig2 = go.Figure(go.Heatmap(
        z=pivot.values, x=list(pivot.columns), y=list(pivot.index),
        text=texto.values, texttemplate="%{text}",
        textfont=dict(size=11, color="white"),
        colorscale=[[0,"#00C48C"],[0.5,"#FFA500"],[1,"#FF4B4B"]],
        zmin=0, zmax=2, showscale=True,
        colorbar=dict(tickvals=[0,1,2], ticktext=["Baixo","Médio","Alto"],
                      title="Risco", title_font_color="#7B92A8",
                      tickfont_color="#7B92A8", bgcolor="rgba(0,0,0,0)",
                      outlinecolor="rgba(255,255,255,0.08)"),
        hovertemplate="<b>%{y}</b><br>%{x}<br>Risco: %{text}<extra></extra>",
        xgap=2, ygap=2,
    ))
    fig2.update_layout(
        **_layout(height=max(280, 80 * len(pivot) + 80),
                  yaxis=dict(autorange="reversed"))
    )
    st.plotly_chart(fig2, use_container_width=True)

    # Cards por região
    st.markdown("#### Diagnóstico — Mês Mais Recente")
    cols_d = st.columns(min(3, len(mais_rec)))
    for i, (_, row) in enumerate(mais_rec.iterrows()):
        nivel = row["risco_seca"]
        icon  = {"Baixo":"✅","Medio":"⚠️","Alto":"🚨"}.get(nivel,"❓")
        css   = {"Baixo":"rc-baixo","Medio":"rc-medio","Alto":"rc-alto"}.get(nivel,"")
        nome  = _nome_curto(row["regiao"])
        cols_d[i % 3].markdown(
            region_card_risco(
                icon=icon, nome=nome, nivel=nivel, css=css,
                chuva_mm=f"{row['precipitacao_total_mm']:.0f}",
                temp_c=f"{row['temperatura_media_c']:.1f}",
                dias_secos=str(int(row['dias_secos'])),
            ),
            unsafe_allow_html=True,
        )


# ── TAB 4 — Distribuição ─────────────────────────────────
with tab_dist:
    st.markdown(f"#### Distribuição de Risco por Região · {periodo_str}")

    df_d = df_mensal.copy()
    df_d["rc"] = df_d["regiao"].apply(_nome_curto)
    cnt = df_d.groupby(["rc","risco_seca"]).size().reset_index(name="n")
    tot = cnt.groupby("rc")["n"].transform("sum")
    cnt["pct"] = (cnt["n"] / tot * 100).round(1)

    fig3 = go.Figure()
    for nivel in ["Baixo","Medio","Alto"]:
        sub = cnt[cnt["risco_seca"] == nivel]
        fig3.add_trace(go.Bar(
            name=nivel, x=sub["rc"], y=sub["pct"],
            marker_color=COR_RISCO[nivel],
            text=sub["pct"].apply(lambda v: f"{v:.0f}%" if v > 6 else ""),
            textposition="inside", textfont_color="white",
            hovertemplate=f"<b>%{{x}}</b><br>{nivel}: %{{y:.1f}}%<extra></extra>",
        ))
    fig3.update_layout(
        **_layout(
            barmode="stack",
            xaxis=dict(title="Região", gridcolor="rgba(255,255,255,0.04)"),
            yaxis=dict(title="% de Meses", range=[0,105],
                       gridcolor="rgba(255,255,255,0.06)"),
            legend=dict(title="Nível"),
            height=420,
        )
    )
    st.plotly_chart(fig3, use_container_width=True)

    k1, k2, k3 = st.columns(3)
    k1.metric("Meses Risco Baixo", n_baixo, f"{n_baixo/n_tot*100:.0f}%")
    k2.metric("Meses Risco Médio", n_medio, f"{n_medio/n_tot*100:.0f}%", delta_color="off")
    k3.metric("Meses Risco Alto",  n_alto,  f"{n_alto/n_tot*100:.0f}%", delta_color="inverse")


# ── TAB 5 — Correlações ──────────────────────────────────
with tab_corr:
    st.markdown("#### Temperatura × Precipitação × ISA")

    fig4 = px.scatter(
        df_mensal,
        x="temperatura_media_c", y="precipitacao_total_mm",
        color="risco_seca", color_discrete_map=COR_RISCO,
        size="isa", size_max=24,
        hover_name=df_mensal["regiao"].apply(_nome_curto),
        hover_data={"precipitacao_total_mm":":.1f","temperatura_media_c":":.1f",
                    "isa":":.1f","risco_seca":True,"mes":True},
        labels={"temperatura_media_c":"Temperatura Média (°C)",
                "precipitacao_total_mm":"Precipitação (mm)",
                "risco_seca":"Risco","isa":"ISA"},
        category_orders={"risco_seca":["Baixo","Medio","Alto"]},
    )
    fig4.add_hline(y=100, line_dash="dash", line_color="rgba(200,214,226,0.2)",
                   annotation_text="100mm")
    fig4.add_hline(y=50,  line_dash="dot",  line_color="rgba(255,75,75,0.25)",
                   annotation_text="50mm")
    fig4.update_layout(
        **_layout(
            xaxis=dict(gridcolor="rgba(255,255,255,0.06)"),
            yaxis=dict(gridcolor="rgba(255,255,255,0.06)"),
            height=460,
        )
    )
    st.plotly_chart(fig4, use_container_width=True)

    st.markdown("#### Matriz de Correlação entre Variáveis")
    cols_corr = ["precipitacao_total_mm","temperatura_media_c",
                 "dias_secos","pct_dias_secos","isa","risco_numerico"]
    corr = df_mensal[cols_corr].corr()
    labels_corr = ["Chuva","Temp","Dias Secos","% Secos","ISA","Risco Num."]

    fig_corr = px.imshow(
        corr, text_auto=".2f",
        x=labels_corr, y=labels_corr,
        color_continuous_scale=[[0,"#FF4B4B"],[0.5,"#1B2329"],[1,"#00C48C"]],
        zmin=-1, zmax=1, aspect="auto",
    )
    fig_corr.update_layout(**_layout(height=380))
    st.plotly_chart(fig_corr, use_container_width=True)


# ── TAB 6 — Machine Learning ─────────────────────────────
with tab_ml:
    st.markdown("#### Classificador de Risco — Random Forest")
    st.caption("Modelo treinado com features climáticas mensais para prever o nível de risco (Baixo/Médio/Alto).")

    if len(df_mensal) < 10:
        st.warning("Poucos dados para treinar o modelo.")
    else:
        with st.spinner("Treinando Random Forest..."):
            modelo, enc, X_tr, X_te, y_tr, y_te = _treinar(df_mensal)

        y_pred  = modelo.predict(X_te)
        acc_tr  = accuracy_score(y_tr, modelo.predict(X_tr))
        acc_te  = accuracy_score(y_te, y_pred)

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Acurácia Treino",  f"{acc_tr:.1%}")
        m2.metric("Acurácia Teste",   f"{acc_te:.1%}")
        m3.metric("Amostras Treino",  len(X_tr))
        m4.metric("Features",         len(FEATURES))

        st.markdown("<br>", unsafe_allow_html=True)
        col_ml1, col_ml2 = st.columns(2)

        with col_ml1:
            st.markdown("##### Importância das Variáveis")
            imp = pd.DataFrame({"Feature": FEATURES,
                                 "Importancia": modelo.feature_importances_}
                               ).sort_values("Importancia", ascending=True)
            fig_imp = px.bar(
                imp, x="Importancia", y="Feature", orientation="h",
                color="Importancia",
                color_continuous_scale=["#1B2329","#00C48C"],
            )
            fig_imp.update_layout(
                **_layout(height=340, coloraxis_showscale=False,
                           xaxis=dict(gridcolor="rgba(255,255,255,0.06)"),
                           yaxis=dict(gridcolor="rgba(0,0,0,0)"))
            )
            st.plotly_chart(fig_imp, use_container_width=True)

        with col_ml2:
            st.markdown("##### Matriz de Confusão")
            nomes_cl = enc.classes_
            cm       = confusion_matrix(y_te, y_pred)
            fig_cm   = go.Figure(go.Heatmap(
                z=cm, x=nomes_cl, y=nomes_cl,
                text=cm, texttemplate="%{text}",
                textfont=dict(size=14, color="white"),
                colorscale=[[0,"#1B2329"],[1,"#00C48C"]],
                showscale=False, xgap=3, ygap=3,
                hovertemplate="Real: %{y}<br>Predito: %{x}<br>Count: %{z}<extra></extra>",
            ))
            fig_cm.update_layout(
                **_layout(height=340,
                           xaxis=dict(title="Predito"),
                           yaxis=dict(title="Real", autorange="reversed"))
            )
            st.plotly_chart(fig_cm, use_container_width=True)

        st.markdown("#### Diagnóstico ML — Mês Mais Recente")
        for _, row in mais_rec.iterrows():
            feat_row   = pd.DataFrame([row[FEATURES]])
            pred_num   = modelo.predict(feat_row)[0]
            pred_proba = modelo.predict_proba(feat_row)[0]
            pred_label = enc.inverse_transform([pred_num])[0]
            confianca  = max(pred_proba) * 100

            nome  = _nome_curto(row["regiao"])
            css   = {"Baixo":"rc-baixo","Medio":"rc-medio","Alto":"rc-alto"}.get(pred_label,"")
            icon  = {"Baixo":"✅","Medio":"⚠️","Alto":"🚨"}.get(pred_label,"❓")
            st.markdown(
                region_card_ml(
                    icon=icon, nome=nome, pred_label=pred_label, css=css,
                    confianca=f"{confianca:.0f}",
                    chuva_mm=f"{row['precipitacao_total_mm']:.0f}",
                    isa=f"{row['isa']:.1f}",
                    isa_cat=row['isa_categoria'],
                    dias_secos=str(int(row['dias_secos'])),
                    pct_secos=f"{row['pct_dias_secos']:.0f}",
                ),
                unsafe_allow_html=True,
            )


# ── TAB 7 — Dados Brutos ─────────────────────────────────
with tab_raw:
    st.markdown("#### Tabela de Dados Mensais Processados")

    exib = df_mensal.copy()
    exib["Região"] = exib["regiao"].apply(_nome_curto)
    exib = exib.rename(columns={
        "mes": "Mês", "ano": "Ano",
        "precipitacao_total_mm": "Chuva (mm)",
        "temperatura_media_c": "Temp Média (°C)",
        "temperatura_max_media_c": "Temp Máx (°C)",
        "dias_secos": "Dias Secos",
        "pct_dias_secos": "% Dias Secos",
        "risco_seca": "Risco",
        "isa": "ISA",
        "isa_categoria": "Categ. ISA",
    })
    colunas_exib = ["Região","Mês","Chuva (mm)","Temp Média (°C)",
                    "Dias Secos","% Dias Secos","Risco","ISA","Categ. ISA"]
    exib_final = exib[[c for c in colunas_exib if c in exib.columns]]

    f1, f2 = st.columns([1, 3])
    with f1:
        filtro_risco = st.multiselect(
            "Filtrar por risco", ["Baixo","Medio","Alto"],
            default=["Baixo","Medio","Alto"],
        )
    exib_filtrado = exib_final[exib_final["Risco"].isin(filtro_risco)]

    st.dataframe(exib_filtrado.reset_index(drop=True),
                 use_container_width=True, height=420)

    csv = exib_filtrado.to_csv(index=False).encode("utf-8")
    st.download_button("⬇ Baixar CSV", data=csv,
                       file_name=f"agrosat_{periodo_str.replace('–','_')}.csv",
                       mime="text/csv")
