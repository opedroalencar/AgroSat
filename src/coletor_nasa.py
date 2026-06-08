"""
Módulo responsável por coletar dados climáticos da API NASA POWER.
A API NASA POWER fornece dados de satélite sobre precipitação, temperatura
e outros parâmetros climáticos para qualquer ponto do globo.

Agora aceita parâmetros de ano de início e fim para flexibilidade.
"""

import requests
import pandas as pd
import time

# Regiões agrícolas importantes do Brasil com suas coordenadas (latitude, longitude)
REGIOES_AGRICOLAS = {
    "Sorriso-MT (Soja)":        {"lat": -12.55, "lon": -55.72},
    "Ribeirao Preto-SP (Cana)": {"lat": -21.17, "lon": -47.81},
    "Rondonopolis-MT (Algodao)": {"lat": -16.47, "lon": -54.64},
    "Passo Fundo-RS (Trigo)":   {"lat": -28.26, "lon": -52.40},
    "Barreiras-BA (Graos)":     {"lat": -12.15, "lon": -44.99},
    "Uberlandia-MG (Milho)":    {"lat": -18.91, "lon": -48.28},
}

# Parâmetros climáticos que vamos buscar na NASA POWER
# PRECTOTCORR = precipitação corrigida (mm/dia)
# T2M         = temperatura média a 2 metros do solo (°C)
# T2M_MAX     = temperatura máxima diária (°C)
PARAMETROS_NASA = "PRECTOTCORR,T2M,T2M_MAX"


def buscar_dados_regiao(nome_regiao, lat, lon, ano_inicio=2024, ano_fim=2024):
    """
    Faz a requisição à API NASA POWER para uma região e período específicos.
    Retorna um DataFrame com os dados climáticos diários.

    Parâmetros:
        nome_regiao (str): nome da região agrícola
        lat (float): latitude
        lon (float): longitude
        ano_inicio (int): ano de início da coleta (padrão: 2024)
        ano_fim (int): ano de fim da coleta (padrão: 2024)
    """
    url = "https://power.larc.nasa.gov/api/temporal/daily/point"

    # Parâmetros da requisição conforme documentação da NASA POWER
    params = {
        "parameters": PARAMETROS_NASA,
        "community": "AG",          # Comunidade: Agronegócio
        "longitude": lon,
        "latitude": lat,
        "start": f"{ano_inicio}0101",
        "end": f"{ano_fim}1231",
        "format": "JSON",
    }

    print(f"  Buscando: {nome_regiao} ({ano_inicio}-{ano_fim}) ...")

    try:
        # Timeout de 30 segundos para não travar o programa
        resposta = requests.get(url, params=params, timeout=30)
        resposta.raise_for_status()

        dados_json = resposta.json()
        propriedades = dados_json["properties"]["parameter"]

        precipitacao    = propriedades["PRECTOTCORR"]
        temperatura_med = propriedades["T2M"]
        temperatura_max = propriedades["T2M_MAX"]

        # Monta lista de dicionários para criar o DataFrame
        registros = []
        for data_str in precipitacao.keys():
            try:
                data = pd.to_datetime(data_str, format="%Y%m%d")
            except Exception:
                continue

            chuva    = precipitacao[data_str]
            temp     = temperatura_med[data_str]
            temp_max = temperatura_max[data_str]

            # A NASA usa -999 como valor ausente — descartamos esses registros
            if chuva == -999 or temp == -999 or temp_max == -999:
                continue

            registros.append({
                "data": data,
                "regiao": nome_regiao,
                "precipitacao_mm": round(chuva, 2),
                "temperatura_media_c": round(temp, 2),
                "temperatura_max_c": round(temp_max, 2),
            })

        df = pd.DataFrame(registros)
        print(f"    OK — {len(df)} registros coletados.")
        return df

    except requests.exceptions.Timeout:
        print(f"    ERRO: Timeout ao buscar {nome_regiao}.")
        return pd.DataFrame()
    except requests.exceptions.RequestException as e:
        print(f"    ERRO de conexao para {nome_regiao}: {e}")
        return pd.DataFrame()
    except KeyError as e:
        print(f"    ERRO ao processar JSON de {nome_regiao}: chave ausente {e}")
        return pd.DataFrame()


def coletar_todas_regioes(ano_inicio=2024, ano_fim=2024, regioes=None):
    """
    Coleta dados de todas (ou de um subconjunto de) regiões agrícolas.
    Retorna um único DataFrame combinado com todos os dados.

    Parâmetros:
        ano_inicio (int): primeiro ano do período
        ano_fim (int): último ano do período
        regioes (list|None): lista de nomes de regiões; None = todas
    """
    print(f"\n=== Coleta de dados — NASA POWER API ({ano_inicio} a {ano_fim}) ===\n")

    mapa_regioes = regioes if regioes else list(REGIOES_AGRICOLAS.keys())
    todos_dados = []

    for nome in mapa_regioes:
        if nome not in REGIOES_AGRICOLAS:
            print(f"  AVISO: regiao '{nome}' nao encontrada. Pulando.")
            continue
        coords = REGIOES_AGRICOLAS[nome]
        df_reg = buscar_dados_regiao(nome, coords["lat"], coords["lon"], ano_inicio, ano_fim)
        if not df_reg.empty:
            todos_dados.append(df_reg)
        time.sleep(1)  # pausa gentil para a API da NASA

    if not todos_dados:
        print("\nERRO CRITICO: Nenhum dado coletado. Verifique a conexao.")
        return pd.DataFrame()

    df_completo = pd.concat(todos_dados, ignore_index=True)
    df_completo = df_completo.sort_values(["regiao", "data"]).reset_index(drop=True)

    print(f"\nColeta concluida! Total de registros: {len(df_completo)}")
    print(f"Regioes com dados: {df_completo['regiao'].nunique()}")
    return df_completo
