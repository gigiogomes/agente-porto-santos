from datetime import date

import pandas as pd

from ingest import contract


def _fato(rows):
    base = {"terminal": "BTP", "teu": 1.0, "unidades": 1.0, "toneladas": 1.0}
    return pd.DataFrame([{**base, **r} for r in rows])


def _full_year(ano, toneladas_mes):
    return [{"ano": ano, "mes": m, "toneladas": toneladas_mes} for m in range(1, 13)]


def _by_rule(results):
    return {r.regra: r for r in results}


def test_base_valida_passa():
    results = contract.validate_fato(_fato(_full_year(2024, 10.0)), today=date(2026, 10, 19))
    assert not contract.has_errors(results)


def test_volume_negativo_bloqueia():
    results = contract.validate_fato(_fato([{"ano": 2024, "mes": 1, "teu": -5.0}]), today=date(2026, 10, 19))
    assert not _by_rule(results)["medidas_nao_negativas"].ok
    assert contract.has_errors(results)


def test_mes_no_futuro_bloqueia():
    results = contract.validate_fato(_fato([{"ano": 2026, "mes": 11}]), today=date(2026, 10, 19))
    assert not _by_rule(results)["sem_meses_no_futuro"].ok


def test_referencia_oficial_de_2025():
    perto = contract.validate_fato(_fato(_full_year(2025, 186.4e6 / 12)), today=date(2026, 10, 19))
    longe = contract.validate_fato(_fato(_full_year(2025, 150e6 / 12)), today=date(2026, 10, 19))
    assert _by_rule(perto)["referencia_toneladas_2025"].ok
    assert not _by_rule(longe)["referencia_toneladas_2025"].ok


def test_cobertura_nao_pode_regredir():
    fato = _fato([{"ano": 2026, "mes": 7}])
    anterior = {"coverage": {"min_year": 2026, "max_year": 2026, "max_month": 8}, "yearly_totals": {}}
    results = contract.validate_fato(fato, anterior, today=date(2026, 10, 19))
    assert not _by_rule(results)["cobertura_nao_regrediu"].ok


def test_revisao_pequena_e_aviso_e_grande_e_erro():
    anterior = {
        "coverage": {"min_year": 2024, "max_year": 2024, "max_month": 12},
        "yearly_totals": {"2024": {"teu": 12.0, "toneladas": 120.0}},
    }
    pequena = contract.validate_fato(_fato(_full_year(2024, 10.1)), anterior, today=date(2026, 10, 19))
    grande = contract.validate_fato(_fato(_full_year(2024, 12.0)), anterior, today=date(2026, 10, 19))

    assert _by_rule(pequena)["revisao_2024"].severidade == "aviso"
    assert not contract.has_errors(pequena)
    assert _by_rule(grande)["revisao_2024"].severidade == "erro"
    assert contract.has_errors(grande)
