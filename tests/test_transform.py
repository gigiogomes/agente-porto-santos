import pandas as pd
import pytest

from conftest import make_raw
from ingest import transform


@pytest.mark.parametrize(
    "raw_name, expected",
    [
        ("SANTOS BRASIL", "SBSA"),
        ("Santos Brasil", "SBSA"),
        ("DPWORLD (EMBRAPORT)", "DPW"),
        ("Embraport", "DPW"),
        ("BANDEIRANTES", "BANDEIRANTES-DEICMAR"),
        ("DEICMAR", "BANDEIRANTES-DEICMAR"),
        ("LIBRA 35", "LIBRA 35"),
        ("Outros", "OUTROS"),
        # Terminal diferente que só compartilha parte do nome com a SBSA.
        ("SANTOS BRASIL (SSZ35.2 Saboó)", "SANTOS BRASIL (SSZ35.2 SABOO)"),
        ("", transform.TERMINAL_NAO_INFORMADO),
        (None, transform.TERMINAL_NAO_INFORMADO),
    ],
)
def test_canonical_terminal(raw_name, expected):
    assert transform.canonical_terminal(raw_name) == expected


@pytest.mark.parametrize(
    "text, expected",
    [("19,135", 19.135), ("1.234,5", 1234.5), ("38", 38.0), ("", 0.0), ("abc", 0.0)],
)
def test_parse_br_number(text, expected):
    assert transform.parse_br_number(pd.Series([text])).iloc[0] == pytest.approx(expected)


def test_build_fato_agrega_por_mes_e_dimensoes():
    raw = make_raw(
        [
            {"ano": 2025, "mes": 1, "terminal": "BTP", "teu": 10, "ton": "100,5"},
            {"ano": 2025, "mes": 1, "terminal": "BTP", "teu": 5, "ton": "1.000"},
            {"ano": 2025, "mes": 2, "terminal": "BTP", "teu": 7, "ton": "1"},
        ]
    )
    fato = transform.build_fato(raw)
    jan = fato[(fato["ano"] == 2025) & (fato["mes"] == 1)]
    assert len(fato) == 2
    assert jan["teu"].iloc[0] == 15
    assert jan["toneladas"].iloc[0] == pytest.approx(1100.5)


def test_cargo_mix_so_existe_para_conteiner():
    raw = make_raw(
        [
            {"ano": 2025, "mes": 1, "terminal": "BTP", "teu": 10},
            {"ano": 2025, "mes": 1, "terminal": "ADM", "natureza": "GRANEL SOLIDO", "mercadoria": "SOJA EM GRÃOS", "ton": "500"},
            {"ano": 2025, "mes": 1, "terminal": "BTP", "movimento": "REMOÇÃO", "mercadoria": "SEM CARGAS", "teu": 2},
        ]
    )
    fato = transform.build_fato(raw).set_index("terminal")
    assert fato.loc["ADM", "cargo_mix"] is None or pd.isna(fato.loc["ADM", "cargo_mix"])
    assert set(fato.loc["BTP", "cargo_mix"]) == {"EXPORTACAO CHEIO", "REMOCAO VAZIO"}
    assert fato.loc["ADM", "mercadoria"] == "SOJA EM GRAOS"


def test_build_fato_descarta_mes_invalido():
    raw = make_raw([{"ano": 2025, "mes": 13, "terminal": "BTP", "teu": 1}, {"ano": 2025, "mes": 1, "terminal": "BTP", "teu": 1}])
    assert transform.build_fato(raw)["mes"].tolist() == [1]


def test_build_fato_exige_colunas_da_aps():
    with pytest.raises(ValueError, match="TOTAL_TONELADAS"):
        transform.build_fato(make_raw([{"ano": 2025, "mes": 1, "terminal": "BTP"}]).drop(columns="TOTAL_TONELADAS"))


def test_dim_terminal_tem_periodo_de_atividade():
    raw = make_raw(
        [
            {"ano": 2010, "mes": 3, "terminal": "LIBRA 35", "teu": 5},
            {"ano": 2019, "mes": 4, "terminal": "LIBRA 35", "teu": 5},
            {"ano": 2025, "mes": 1, "terminal": "SANTOS BRASIL", "teu": 50},
        ]
    )
    dim = transform.build_dim_terminal(transform.build_fato(raw)).set_index("terminal")
    assert dim.loc["LIBRA 35", ["primeiro_mes", "ultimo_mes"]].tolist() == ["2010-03", "2019-04"]
    assert dim.loc["SBSA", "nome"] == "Santos Brasil"
    assert dim.index[0] == "SBSA"  # ordenada por volume
