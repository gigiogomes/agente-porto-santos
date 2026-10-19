import pytest

from conftest import make_store
from tools import port_data_tools as pdt
from ui.charts import DEMAIS, visualizacao


@pytest.fixture
def store():
    terminais = ["SANTOS BRASIL", "BTP", "DPWORLD (EMBRAPORT)", "ECOPORTO", "TEV", "MARIMEX"]
    rows = [
        {"ano": ano, "mes": mes, "terminal": t, "teu": (i + 1) * 100 + ano - 2020}
        for ano in (2024, 2025) for mes in range(1, 13) for i, t in enumerate(terminais)
    ]
    return make_store(rows)


def _vis(store, **kwargs):
    return visualizacao(pdt.run_query(store, **kwargs))


def test_total_vira_cartao(store):
    tipo, visual = _vis(store, start_date="2025", end_date="2025")
    assert tipo == "cartao"
    assert visual["label"] == "Total (TEU)"


def test_yoy_sem_dimensao_vira_cartao_com_delta(store):
    tipo, visual = _vis(store, start_date="2024", end_date="2025", compare_with_previous=True)
    assert tipo == "cartao"
    assert visual["delta"].endswith("vs 2024")


def test_share_por_terminal_vira_barras_com_rotulo_percentual(store):
    tipo, fig = _vis(store, metric="market_share", start_date="2025", end_date="2025", group_by=["terminal"])
    assert tipo == "figura"
    barra = fig.data[0]
    assert barra.orientation == "h"
    assert all(t.endswith("%") and "," in t for t in barra.text)
    assert fig.layout.separators == ",."


def test_serie_temporal_vira_linha(store):
    tipo, fig = _vis(store, start_date="2024", end_date="2025", group_by=["mes"])
    assert tipo == "figura"
    assert fig.data[0].type == "scatter" and len(fig.data[0].x) == 24


def test_muitas_series_juntam_o_resto_em_demais(store):
    tipo, fig = _vis(store, start_date="2025", end_date="2025", group_by=["mes", "terminal"])
    nomes = [trace.name for trace in fig.data]
    assert len(nomes) == 5 and nomes[-1] == DEMAIS


def test_yoy_por_terminal_vira_barras_em_dois_tons(store):
    tipo, fig = _vis(store, start_date="2024", end_date="2025", compare_with_previous=True, group_by=["terminal"])
    assert tipo == "figura"
    assert [trace.name for trace in fig.data] == ["2024", "2025"]
    assert fig.data[0].marker.color != fig.data[1].marker.color


def test_cagr_sem_terminal_vira_cartao():
    rows = [{"ano": ano, "mes": m, "terminal": "BTP", "teu": v} for ano, v in ((2023, 100), (2024, 110)) for m in range(1, 13)]
    tipo, visual = _vis(make_store(rows), metric="cagr")
    assert tipo == "cartao"
    assert visual["value"] == "10,00% a.a."


def test_aviso_nao_tem_visualizacao(store):
    assert _vis(store, terminals=["XYZ"]) == ("nenhuma", None)
