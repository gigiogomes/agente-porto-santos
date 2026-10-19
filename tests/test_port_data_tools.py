import pandas as pd
import pytest

from tools import port_data_tools as pdt


def _raw(rows):
    """Monta um DataFrame no formato do CSV da APS a partir de (ano, mes, terminal, teu)."""
    return pd.DataFrame(
        [
            {
                "ANO": ano,
                "MES": mes,
                "TERMINAIS": terminal,
                "TOTAL_TEU": teu,
                "TOTAL_UNID": teu,
                "MERCADORIAS": "DIVERSOS",
                "MOVIMENTO": "CONVENCIONAL",
                "TIPO_NAVEGACAO": "LONGO CURSO",
                "SENTIDO": "EMBARQUE",
            }
            for ano, mes, terminal, teu in rows
        ]
    )


@pytest.fixture
def df():
    return pdt._normalize_dataframe(
        _raw(
            [
                (2015, 1, "SANTOS BRASIL", 300),
                (2015, 6, "SANTOS BRASIL", 300),
                (2015, 7, "SANTOS BRASIL", 1000),
                (2015, 1, "LIBRA 35", 400),
                (2015, 6, "DPWORLD (EMBRAPORT)", 0),
            ]
        )
    )


# ------------------------------------------------------------------ #
# Bug 1: terminais sem alias eram descartados do groupby
# ------------------------------------------------------------------ #
@pytest.mark.parametrize(
    "raw_name, expected",
    [
        ("SANTOS BRASIL", "SBSA"),
        ("Santos Brasil", "SBSA"),
        ("DPWORLD (EMBRAPORT)", "DPW"),
        ("BANDEIRANTES", "BANDEIRANTES-DEICMAR"),
        ("DEICMAR", "BANDEIRANTES-DEICMAR"),
        ("LIBRA 35", "LIBRA 35"),
        ("Outros", "OUTROS"),
        ("", pdt.TERMINAL_NAO_INFORMADO),
        (None, pdt.TERMINAL_NAO_INFORMADO),
    ],
)
def test_canonical_terminal(raw_name, expected):
    assert pdt._canonical_terminal(raw_name) == expected


def test_nenhuma_linha_fica_sem_terminal(df):
    assert df["TERMINAL_SIGLA_CANONICA"].notna().all()


def test_market_share_inclui_terminais_sem_alias(df):
    out = pdt.query_port_data(
        df, metric="market_share", start_date="2015-01", end_date="2015-06", group_by=["terminal"]
    )
    # Jan-Jun/2015: SBSA 600 e LIBRA 35 400 -> 60% / 40%. Antes do fix, LIBRA 35
    # sumia e a SBSA aparecia com 100%.
    assert "LIBRA 35" in out
    assert "60.0%" in out
    assert "40.0%" in out


def test_market_share_ordena_pelo_volume_mesmo_com_share_empatado():
    df = pdt._normalize_dataframe(
        _raw(
            [
                (2015, 1, "SANTOS BRASIL", 1_000_000),
                (2015, 1, "LIBRA 37", 2),
                (2015, 1, "LOCALFRIO", 40),
            ]
        )
    )
    out = pdt.query_port_data(df, metric="market_share", start_date="2015", end_date="2015", group_by=["terminal"])
    # Os dois arredondam para 0.0%; LOCALFRIO (40 TEU) deve vir antes de LIBRA 37 (2 TEU).
    assert out.index("LOCALFRIO") < out.index("LIBRA 37")


def test_filtro_de_terminal_aceita_nome_por_extenso(df):
    por_sigla = pdt.query_port_data(df, terminals=["SBSA"], start_date="2015", end_date="2015")
    por_nome = pdt.query_port_data(df, terminals=["Santos Brasil"], start_date="2015", end_date="2015")
    assert por_sigla == por_nome
    assert "1600" in por_nome


def test_coverage_lista_terminais_por_volume(df):
    cov = pdt._extract_coverage(df, "APS")
    assert cov.terminal_list[:2] == ["SBSA", "LIBRA 35"]


# ------------------------------------------------------------------ #
# Bug 2: datas como '2015-06-31' quebravam a consulta
# ------------------------------------------------------------------ #
@pytest.mark.parametrize("end_date", ["2015-06-31", "2015-06-30", "2015-06", "2015-6-1"])
def test_end_date_inclui_o_mes_inteiro_e_aceita_dia_invalido(df, end_date):
    out = pdt.query_port_data(df, terminals=["SBSA"], start_date="2015-01-01", end_date=end_date)
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
    rows = [(ano, mes, "BTP", 100 if ano == 2020 else 121) for ano in (2020, 2022) for mes in range(1, 13)]
    rows += [(2021, mes, "BTP", 110) for mes in range(1, 13)]
    df = pdt._normalize_dataframe(_raw(rows))
    out = pdt.query_port_data(df, metric="cagr", start_date="2020-01-01", end_date="2022-11-31")
    # 2022 vai só até novembro na janela -> anos completos 2020 e 2021: 1200 -> 1320 TEU.
    assert "2020" in out and "2021" in out
    assert "10.0%" in out


# ------------------------------------------------------------------ #
# Bug 4: fonte em https e backup só quando a APS falha
# ------------------------------------------------------------------ #
def test_url_principal_usa_https():
    assert pdt.URL_MENSARIO.startswith("https://")


def test_backup_nao_e_baixado_quando_a_aps_responde(monkeypatch):
    chamadas = []

    def fake_read(url, **kwargs):
        chamadas.append(url)
        return _raw([(2025, 1, "BTP", 10)])

    monkeypatch.setattr(pdt, "_read_csv_from_url", fake_read)
    _, coverage, diagnostics = pdt._load_port_data_uncached()

    assert chamadas == [pdt.URL_MENSARIO]
    assert coverage.source_name == "APS"
    assert diagnostics == [{"source": "APS", "status": "success"}]


def test_backup_e_usado_quando_a_aps_falha(monkeypatch):
    def fake_read(url, **kwargs):
        if url == pdt.URL_MENSARIO:
            raise RuntimeError("timeout")
        return _raw([(2024, 12, "BTP", 10)])

    monkeypatch.setattr(pdt, "_read_csv_from_url", fake_read)
    _, coverage, diagnostics = pdt._load_port_data_uncached()

    assert coverage.source_name == "BACKUP"
    assert [d["status"] for d in diagnostics] == ["error", "success"]


def test_erro_quando_todas_as_fontes_falham(monkeypatch):
    def fake_read(url, **kwargs):
        raise RuntimeError("fora do ar")

    monkeypatch.setattr(pdt, "_read_csv_from_url", fake_read)
    with pytest.raises(RuntimeError, match="Falha fatal"):
        pdt._load_port_data_uncached()
