import pandas as pd
import pytest

from ingest.transform import build_fato


def make_raw(rows):
    """
    Monta um DataFrame no formato do CSV da APS. Cada linha é um dict com
    ano, mes, terminal e teu; os demais campos têm valores padrão de contêiner.
    """
    defaults = {
        "natureza": "CARGA CONTEINERIZADA",
        "mercadoria": "DIVERSOS",
        "movimento": "CONVENCIONAL",
        "navegacao": "LONGO CURSO",
        "sentido": "EMBARQUE",
        "teu": 0,
        "unid": None,
        "ton": "0",
    }
    records = []
    for row in rows:
        r = {**defaults, **row}
        records.append(
            {
                "ANO": r["ano"],
                "MES": r["mes"],
                "TERMINAIS": r["terminal"],
                "NATUREZA_CARGA": r["natureza"],
                "TIPO_NAVEGACAO": r["navegacao"],
                "MOVIMENTO": r["movimento"],
                "SENTIDO": r["sentido"],
                "TIPO_INSTALACAO": "PORTO ORGANIZADO",
                "CLASSENAVIO": "PORTA-CONTAINERS",
                "MERCADORIAS": r["mercadoria"],
                "TOTAL_TEU": r["teu"],
                "TOTAL_UNID": r["teu"] if r["unid"] is None else r["unid"],
                "TOTAL_TONELADAS": r["ton"],
            }
        )
    return pd.DataFrame(records)


def make_store(rows):
    # Import local: os testes do pipeline não dependem do app (streamlit, duckdb).
    from tools.port_data_tools import PortDataStore

    return PortDataStore(build_fato(make_raw(rows)))


@pytest.fixture
def store():
    return make_store(
        [
            {"ano": 2015, "mes": 1, "terminal": "SANTOS BRASIL", "teu": 300},
            {"ano": 2015, "mes": 6, "terminal": "SANTOS BRASIL", "teu": 300},
            {"ano": 2015, "mes": 7, "terminal": "SANTOS BRASIL", "teu": 1000},
            {"ano": 2015, "mes": 1, "terminal": "LIBRA 35", "teu": 400},
            {"ano": 2015, "mes": 6, "terminal": "DPWORLD (EMBRAPORT)", "teu": 0},
        ]
    )
