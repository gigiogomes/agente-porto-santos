import json
from io import BytesIO

import pandas as pd
import pytest

from conftest import make_raw, make_store
from ingest.transform import build_fato
from tools import port_data_tools as pdt


# ------------------------------------------------------------------ #
# Terminais sem alias não podem sumir do market share
# ------------------------------------------------------------------ #
def test_market_share_inclui_terminais_sem_alias(store):
    out = pdt.query_port_data(store, metric="market_share", start_date="2015-01", end_date="2015-06", group_by=["terminal"])
    # Jan-Jun/2015: SBSA 600 e LIBRA 35 400 -> 60% / 40%.
    assert "LIBRA 35" in out
    assert "60.0%" in out
    assert "40.0%" in out


def test_market_share_ordena_pelo_volume_mesmo_com_share_empatado():
    store = make_store(
        [
            {"ano": 2015, "mes": 1, "terminal": "SANTOS BRASIL", "teu": 1_000_000},
            {"ano": 2015, "mes": 1, "terminal": "LIBRA 37", "teu": 2},
            {"ano": 2015, "mes": 1, "terminal": "LOCALFRIO", "teu": 40},
        ]
    )
    out = pdt.query_port_data(store, metric="market_share", start_date="2015", end_date="2015", group_by=["terminal"])
    # Os dois arredondam para 0.0%; LOCALFRIO (40 TEU) deve vir antes de LIBRA 37 (2 TEU).
    assert out.index("LOCALFRIO") < out.index("LIBRA 37")


def test_filtro_de_terminal_aceita_nome_por_extenso(store):
    por_sigla = pdt.query_port_data(store, terminals=["SBSA"], start_date="2015", end_date="2015")
    por_nome = pdt.query_port_data(store, terminals=["Santos Brasil"], start_date="2015", end_date="2015")
    assert por_sigla == por_nome
    assert "1600" in por_nome


def test_coverage_lista_terminais_por_volume(store):
    cov = pdt._extract_coverage(build_fato(make_raw([
        {"ano": 2015, "mes": 1, "terminal": "SANTOS BRASIL", "teu": 600},
        {"ano": 2015, "mes": 1, "terminal": "LIBRA 35", "teu": 400},
    ])), "APS")
    assert cov.terminal_list[:2] == ["SBSA", "LIBRA 35"]


# ------------------------------------------------------------------ #
# Datas: a base é mensal, o dia é ignorado
# ------------------------------------------------------------------ #
@pytest.mark.parametrize("end_date", ["2015-06-31", "2015-06-30", "2015-06", "2015-6-1"])
def test_end_date_inclui_o_mes_inteiro_e_aceita_dia_invalido(store, end_date):
    out = pdt.query_port_data(store, terminals=["SBSA"], start_date="2015-01-01", end_date=end_date)
    # Jan + Jun = 600; Jul (1000) fica de fora.
    assert "600" in out
    assert "1600" not in out


def test_ano_sozinho_cobre_o_ano_inteiro():
    assert pdt._parse_month_bound("2015", is_end=False) == pd.Timestamp("2015-01-01")
    assert pdt._parse_month_bound("2015", is_end=True) == pd.Timestamp("2015-12-01")


@pytest.mark.parametrize("value", ["2015-13", "ontem", "15/06/2015"])
def test_data_invalida_gera_erro_claro(value):
    with pytest.raises(ValueError, match="inválid"):
        pdt._parse_month_bound(value, is_end=True)


def test_cagr_aceita_fim_de_mes_invalido():
    rows = [{"ano": 2020, "mes": m, "terminal": "BTP", "teu": 100} for m in range(1, 13)]
    rows += [{"ano": 2021, "mes": m, "terminal": "BTP", "teu": 110} for m in range(1, 13)]
    rows += [{"ano": 2022, "mes": m, "terminal": "BTP", "teu": 121} for m in range(1, 13)]
    out = pdt.query_port_data(make_store(rows), metric="cagr", start_date="2020-01-01", end_date="2022-11-31")
    # 2022 vai só até novembro na janela -> anos completos 2020 e 2021: 1200 -> 1320 TEU.
    assert "2020" in out and "2021" in out
    assert "10.0%" in out


# ------------------------------------------------------------------ #
# Toneladas, naturezas, mercadorias e sentido
# ------------------------------------------------------------------ #
@pytest.fixture
def store_misto():
    return make_store(
        [
            {"ano": 2025, "mes": 1, "terminal": "BTP", "teu": 100, "unid": 60, "ton": "1.000"},
            {"ano": 2025, "mes": 1, "terminal": "ADM", "natureza": "GRANEL SOLIDO", "mercadoria": "SOJA EM GRÃOS", "ton": "5.000"},
            {"ano": 2025, "mes": 1, "terminal": "ADM", "natureza": "GRANEL SOLIDO", "mercadoria": "FARELO DE SOJA", "ton": "2.000"},
            {"ano": 2025, "mes": 1, "terminal": "TGG", "natureza": "GRANEL SOLIDO", "mercadoria": "MILHO", "ton": "3.000", "sentido": "DESEMBARQUE"},
            {"ano": 2025, "mes": 1, "terminal": "TEV", "natureza": "CARGA GERAL", "mercadoria": "VEÍCULOS", "unid": 900, "ton": "1.500"},
        ]
    )


def test_teu_e_conteineres_consideram_so_carga_conteinerizada(store_misto):
    # A carga geral também tem "unidades" (veículos), que não podem entrar como contêiner.
    assert pdt.query_port_data(store_misto, metric="conteineres").split()[-1] == "60"
    assert pdt.query_port_data(store_misto, metric="teus").split()[-1] == "100"


def test_toneladas_considera_o_porto_inteiro_e_descreve_o_escopo(store_misto):
    out = pdt.query_port_data(store_misto, metric="toneladas")
    assert out.split()[-1] == "12500"
    assert out.startswith("[Escopo: medida=Toneladas; naturezas=todas]")


def test_mercadoria_nome_exato_ou_parcial(store_misto):
    exato = pdt.query_port_data(store_misto, metric="toneladas", mercadorias=["Soja em grãos"])
    parcial = pdt.query_port_data(store_misto, metric="toneladas", mercadorias=["soja"])
    assert exato.split()[-1] == "5000"
    assert parcial.split()[-1] == "7000"
    assert "FARELO DE SOJA" in parcial


def test_mercadoria_inexistente_lista_as_validas(store_misto):
    out = pdt.query_port_data(store_misto, metric="toneladas", mercadorias=["ouro"])
    assert out.startswith("[AVISO DO SISTEMA] Mercadoria não encontrada")
    assert "MILHO" in out


def test_sentido_e_group_by_por_natureza(store_misto):
    embarque = pdt.query_port_data(store_misto, metric="toneladas", sentido=["EMBARQUE"])
    assert embarque.split()[-1] == "9500"

    por_natureza = pdt.query_port_data(store_misto, metric="market_share", medida="toneladas", group_by=["natureza_carga"])
    linhas = por_natureza.splitlines()
    assert "GRANEL SOLIDO" in linhas[2] and "80.0%" in linhas[2]


def test_exportacao_e_so_longo_curso():
    store = make_store(
        [
            {"ano": 2025, "mes": 1, "terminal": "ADM", "natureza": "GRANEL SOLIDO", "mercadoria": "SOJA EM GRÃOS", "ton": "1.000"},
            {"ano": 2025, "mes": 1, "terminal": "ADM", "natureza": "GRANEL SOLIDO", "mercadoria": "SOJA EM GRÃOS", "ton": "40", "navegacao": "CABOTAGEM"},
        ]
    )
    embarque = pdt.query_port_data(store, metric="toneladas", sentido=["EMBARQUE"])
    exportacao = pdt.query_port_data(store, metric="toneladas", sentido=["EMBARQUE"], navegacao=["LONGO CURSO"])
    assert embarque.split()[-1] == "1040"
    assert exportacao.split()[-1] == "1000"
    assert "navegacao=LONGO CURSO" in exportacao


def test_yoy_em_toneladas_rotula_a_medida():
    rows = [{"ano": a, "mes": 1, "terminal": "ADM", "natureza": "GRANEL SOLIDO", "ton": t} for a, t in [(2024, "100"), (2025, "150")]]
    out = pdt.query_port_data(make_store(rows), metric="toneladas", compare_with_previous=True, group_by=["natureza_carga"])
    assert "Toneladas 2024" in out and "Toneladas 2025" in out
    assert "50.0%" in out


def test_medida_invalida():
    with pytest.raises(ValueError, match="Medida inválida"):
        pdt.query_port_data(make_store([{"ano": 2025, "mes": 1, "terminal": "BTP", "teu": 1}]), medida="litros")


# ------------------------------------------------------------------ #
# Ordem das fontes: publicado -> APS -> backup
# ------------------------------------------------------------------ #
def _parquet_bytes(fato):
    buffer = BytesIO()
    fato.to_parquet(buffer, index=False)
    return buffer.getvalue()


@pytest.fixture
def respostas():
    raw = make_raw([{"ano": 2025, "mes": 1, "terminal": "BTP", "teu": 10}])
    return {
        pdt.PUBLISHED_FATO_URL: _parquet_bytes(build_fato(raw)),
        pdt.PUBLISHED_MANIFEST_URL: json.dumps({"generated_at": "2026-10-19T09:00:00+00:00"}).encode(),
        pdt.URL_MENSARIO: raw.to_csv(index=False).encode(),
        pdt.BACKUP_CSV_URL: raw.to_csv(index=False).encode(),
    }


def _fake_reader(respostas, falham=()):
    chamadas = []

    def fake(url, **kwargs):
        chamadas.append(url)
        if url in falham:
            raise RuntimeError("fora do ar")
        return respostas[url]

    return fake, chamadas


def test_usa_os_dados_publicados_sem_baixar_o_csv(monkeypatch, respostas):
    fake, chamadas = _fake_reader(respostas)
    monkeypatch.setattr(pdt, "_read_bytes", fake)
    _, coverage, diagnostics = pdt._load_port_data_uncached()

    assert coverage.source_name == "PUBLICADO"
    assert coverage.generated_at == "2026-10-19T09:00:00+00:00"
    assert pdt.URL_MENSARIO not in chamadas
    assert diagnostics == [{"source": "PUBLICADO", "status": "success"}]


def test_cai_para_a_aps_quando_os_dados_publicados_falham(monkeypatch, respostas):
    fake, chamadas = _fake_reader(respostas, falham={pdt.PUBLISHED_FATO_URL})
    monkeypatch.setattr(pdt, "_read_bytes", fake)
    fato, coverage, diagnostics = pdt._load_port_data_uncached()

    assert coverage.source_name == "APS"
    assert pdt.BACKUP_CSV_URL not in chamadas
    assert fato["teu"].sum() == 10


def test_backup_e_o_ultimo_recurso(monkeypatch, respostas):
    fake, _ = _fake_reader(respostas, falham={pdt.PUBLISHED_FATO_URL, pdt.URL_MENSARIO})
    monkeypatch.setattr(pdt, "_read_bytes", fake)
    _, coverage, diagnostics = pdt._load_port_data_uncached()

    assert coverage.source_name == "BACKUP"
    assert [d["status"] for d in diagnostics] == ["error", "error", "success"]


def test_erro_quando_todas_as_fontes_falham(monkeypatch, respostas):
    fake, _ = _fake_reader(respostas, falham=set(respostas))
    monkeypatch.setattr(pdt, "_read_bytes", fake)
    with pytest.raises(RuntimeError, match="Falha fatal"):
        pdt._load_port_data_uncached()


def test_urls_usam_https():
    for url in [pdt.URL_MENSARIO, pdt.BACKUP_CSV_URL, pdt.PUBLISHED_FATO_URL, pdt.PUBLISHED_MANIFEST_URL]:
        assert url.startswith("https://")
