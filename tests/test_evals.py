import pytest

from conftest import make_store
from evals.oracle import Oracle
from evals.scoring import extrair_numeros, formato_brasileiro, tem_percentual, tem_valor


@pytest.mark.parametrize(
    "texto, esperado",
    [
        ("movimentou 2.595.312 TEUs", 2595312),
        ("movimentou 2,595,312 TEUs", 2595312),
        ("cerca de 5,9 milhões de TEUs", 5.9e6),
        ("186,5 milhões de toneladas", 186.5e6),
        ("23 mil toneladas", 23000),
    ],
)
def test_extrai_numeros_em_varios_formatos(texto, esperado):
    assert tem_valor(extrair_numeros(texto), esperado)


def test_percentual_compara_o_modulo():
    numeros = extrair_numeros("queda de 10,25% no período")
    assert tem_percentual(numeros, -10.25)
    assert not tem_percentual(numeros, 10.5)


def test_formato_brasileiro():
    assert formato_brasileiro("2.595.312 TEUs (43,91%)")
    assert not formato_brasileiro("2,595,312 TEUs")


def test_oraculo():
    store = make_store(
        [
            {"ano": 2024, "mes": m, "terminal": "BTP", "teu": 10} for m in range(1, 13)
        ]
        + [{"ano": 2025, "mes": m, "terminal": "BTP", "teu": 11} for m in range(1, 13)]
        + [{"ano": 2025, "mes": 1, "terminal": "SANTOS BRASIL", "teu": 132}]
    )
    o = Oracle(store)
    assert (o.max_year, o.max_month) == (2025, 12)
    assert o.total(ano=2025) == 264
    assert o.shares("terminal", ano=2025) == {"BTP": 50.0, "SBSA": 50.0}
    assert o.yoy_pct(2024, 2025, terminais=["BTP"]) == 10.0
    assert o.cagr_pct(2024, 2025, terminais=["BTP"]) == 10.0
