"""
Módulo de Machine Learning — classifica risco de seca com Random Forest.
Usa features climáticas mensais para prever o nível de risco (Baixo/Medio/Alto).
"""

import matplotlib
matplotlib.use("Agg")   # renderiza sem abrir janela
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import os

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
)

PASTA_GRAFICOS = os.path.join(os.path.dirname(os.path.dirname(__file__)), "graficos")

# Features (variáveis de entrada) que o modelo usa para aprender
FEATURES = [
    "precipitacao_total_mm",    # Chuva acumulada no mês
    "temperatura_media_c",      # Temperatura média do mês
    "temperatura_max_media_c",  # Temperatura máxima média
    "dias_secos",               # Quantidade de dias sem chuva
    "pct_dias_secos",           # Percentual de dias secos
    "mes",                      # Mês do ano (sazonalidade)
]

TARGET = "risco_seca"


def preparar_dados_ml(df_mensal):
    """Separa features e target, codifica e divide em treino/teste (80/20)."""
    df_ml = df_mensal[FEATURES + [TARGET]].dropna()
    X = df_ml[FEATURES]
    y = df_ml[TARGET]

    encoder   = LabelEncoder()
    y_encoded = encoder.fit_transform(y)

    # Estratificação garante proporção das classes em treino e teste
    try:
        X_treino, X_teste, y_treino, y_teste = train_test_split(
            X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )
    except ValueError:
        # Poucos dados: sem estratificação
        X_treino, X_teste, y_treino, y_teste = train_test_split(
            X, y_encoded, test_size=0.2, random_state=42
        )

    print(f"\n  Amostras de treino: {len(X_treino)} | Amostras de teste: {len(X_teste)}")
    return X_treino, X_teste, y_treino, y_teste, encoder


def treinar_modelo(X_treino, y_treino):
    """
    Treina um Random Forest com 100 árvores.
    Random Forest é robusto, não precisa de normalização e lida bem com
    poucos dados — ideal para este cenário.
    """
    print("\n  Treinando modelo Random Forest...")

    modelo = RandomForestClassifier(
        n_estimators=100,
        max_depth=5,
        random_state=42,
        class_weight="balanced",
    )
    modelo.fit(X_treino, y_treino)
    print("  Treinamento concluido!")
    return modelo


def avaliar_modelo(modelo, X_treino, X_teste, y_treino, y_teste, encoder):
    """Avalia o desempenho do modelo, exibe métricas e gera gráficos."""
    print("\n=== Avaliacao do Modelo ===")

    y_pred = modelo.predict(X_teste)

    acc_treino = accuracy_score(y_treino, modelo.predict(X_treino))
    acc_teste  = accuracy_score(y_teste, y_pred)

    print(f"\n  Acuracia no treino : {acc_treino:.2%}")
    print(f"  Acuracia no teste  : {acc_teste:.2%}")

    scores_cv = cross_val_score(modelo, X_treino, y_treino, cv=min(5, len(X_treino)),
                                scoring="accuracy")
    print(f"  Validacao cruzada  : {scores_cv.mean():.2%} +/- {scores_cv.std():.2%}")

    nomes_classes = encoder.classes_
    print("\n  Relatorio de Classificacao:")
    print(classification_report(y_teste, y_pred, target_names=nomes_classes))

    importancias = pd.Series(
        modelo.feature_importances_, index=FEATURES
    ).sort_values(ascending=False)

    print("\n  Importancia das variaveis para o modelo:")
    for feat, imp in importancias.items():
        barra = "#" * int(imp * 40)
        print(f"    {feat:<30} {barra} {imp:.3f}")

    _grafico_matriz_confusao(y_teste, y_pred, nomes_classes)
    _grafico_importancia_features(importancias)

    return acc_teste, scores_cv.mean()


def _grafico_matriz_confusao(y_teste, y_pred, nomes_classes):
    """Gera e salva o gráfico da matriz de confusão."""
    cm = confusion_matrix(y_teste, y_pred)

    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=nomes_classes, yticklabels=nomes_classes,
        ax=ax, linewidths=0.5,
    )
    ax.set_title("Matriz de Confusao — Classificador de Risco de Seca", fontweight="bold")
    ax.set_xlabel("Predito pelo Modelo")
    ax.set_ylabel("Valor Real")

    os.makedirs(PASTA_GRAFICOS, exist_ok=True)
    caminho = os.path.join(PASTA_GRAFICOS, "05_matriz_confusao.png")
    fig.savefig(caminho, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Grafico salvo: graficos/05_matriz_confusao.png")


def _grafico_importancia_features(importancias):
    """Gera gráfico de barras com a importância de cada variável."""
    fig, ax = plt.subplots(figsize=(9, 5))
    cores = ["#FF4B4B"] + ["#3498db"] * (len(importancias) - 1)
    importancias.plot(kind="barh", ax=ax, color=cores[::-1], edgecolor="white")
    ax.set_title("Importancia das Variaveis — Random Forest", fontweight="bold")
    ax.set_xlabel("Importancia (Gini)")
    ax.invert_yaxis()

    caminho = os.path.join(PASTA_GRAFICOS, "06_importancia_features.png")
    fig.savefig(caminho, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Grafico salvo: graficos/06_importancia_features.png")


def prever_risco_atual(modelo, encoder, df_mensal):
    """
    Usa o modelo treinado para prever o risco do mês mais recente de cada região.
    """
    print("\n=== Diagnostico de Risco Atual (mes mais recente) ===\n")

    mais_recente = df_mensal.sort_values(["ano", "mes"]).groupby("regiao").last().reset_index()

    for _, row in mais_recente.iterrows():
        features_row = pd.DataFrame([row[FEATURES]])
        pred_num   = modelo.predict(features_row)[0]
        pred_label = encoder.inverse_transform([pred_num])[0]

        icone     = {"Baixo": "[OK] ", "Medio": "[!!] ", "Alto": "[SOS]"}.get(pred_label, "[ ? ]")
        nome_curto = row["regiao"].split(" (")[0] if " (" in row["regiao"] else row["regiao"]

        print(f"  {icone} {nome_curto:<30} -> Risco: {pred_label:<6} "
              f"| Chuva: {row['precipitacao_total_mm']:>6.1f}mm "
              f"| Temp: {row['temperatura_media_c']:>5.1f}C "
              f"| Dias secos: {int(row['dias_secos']):>2d}")

    return mais_recente


def executar_ml(df_mensal):
    """Pipeline completo de ML: prepara -> treina -> avalia -> prediz."""
    print("\n=== Modulo de Machine Learning ===")

    if len(df_mensal) < 10:
        print("  AVISO: Dados insuficientes para treinar o modelo.")
        return None, None

    X_treino, X_teste, y_treino, y_teste, encoder = preparar_dados_ml(df_mensal)
    modelo = treinar_modelo(X_treino, y_treino)
    avaliar_modelo(modelo, X_treino, X_teste, y_treino, y_teste, encoder)
    prever_risco_atual(modelo, encoder, df_mensal)

    return modelo, encoder
