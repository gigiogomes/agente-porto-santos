"""
Contrato de dados: regras que a tabela fato precisa cumprir para ser publicada.

Uma regra com severidade "erro" bloqueia a publicação (o app continua com a
última versão boa). "aviso" só fica registrado no manifesto.
"""
from dataclasses import asdict, dataclass
from datetime import date
from typing import Dict, List, Optional

import pandas as pd

# Revisão de anos fechados em relação à versão publicada anterior.
REVISAO_AVISO = 0.005  # 0,5%: a APS revisou números; vale registrar
REVISAO_ERRO = 0.05  # 5%: mudança grande demais para ser revisão normal

# Totais anuais divulgados oficialmente, usados como teste de sanidade.
# 2025: recorde de 186,4 mi t divulgado pela APS (ver docs/teste_porto.txt).
REFERENCIAS_TONELADAS = {2025: 186.4e6}
TOLERANCIA_REFERENCIA = 0.01


@dataclass
class CheckResult:
    regra: str
    ok: bool
    severidade: str  # "erro" | "aviso"
    detalhe: str


def yearly_totals(fato: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    totals = fato.groupby("ano")[["teu", "toneladas"]].sum()
    return {str(ano): {k: round(float(v), 3) for k, v in row.items()} for ano, row in totals.iterrows()}


def coverage(fato: pd.DataFrame) -> Dict[str, int]:
    max_year = int(fato["ano"].max())
    return {
        "min_year": int(fato["ano"].min()),
        "max_year": max_year,
        "max_month": int(fato.loc[fato["ano"] == max_year, "mes"].max()),
    }


def _complete_years(fato: pd.DataFrame) -> set:
    months = fato.groupby("ano")["mes"].nunique()
    return {int(ano) for ano, n in months.items() if n >= 12}


def validate_fato(
    fato: pd.DataFrame, previous_manifest: Optional[dict] = None, today: Optional[date] = None
) -> List[CheckResult]:
    today = today or date.today()
    results = []

    def check(regra, ok, detalhe, severidade="erro"):
        results.append(CheckResult(regra, bool(ok), severidade, detalhe))

    if fato.empty:
        check("base_nao_vazia", False, "A tabela fato não tem linhas.")
        return results
    check("base_nao_vazia", True, f"{len(fato):,} linhas.")

    negativos = int((fato[["teu", "unidades", "toneladas"]] < 0).any(axis=1).sum())
    check("medidas_nao_negativas", negativos == 0, f"{negativos} linhas com volume negativo.")

    periodo = fato["ano"] * 100 + fato["mes"]
    futuro = int((periodo > today.year * 100 + today.month).sum())
    check("sem_meses_no_futuro", futuro == 0, f"{futuro} linhas com mês posterior a {today:%Y-%m}.")

    sem_terminal = int(fato["terminal"].isna().sum() + (fato["terminal"] == "").sum())
    check("terminal_preenchido", sem_terminal == 0, f"{sem_terminal} linhas sem terminal.")

    cov = coverage(fato)
    complete = _complete_years(fato)
    totals = yearly_totals(fato)

    for ano, referencia in REFERENCIAS_TONELADAS.items():
        if ano not in complete:
            continue
        atual = totals[str(ano)]["toneladas"]
        desvio = atual / referencia - 1
        check(
            f"referencia_toneladas_{ano}",
            abs(desvio) <= TOLERANCIA_REFERENCIA,
            f"{atual / 1e6:.1f} mi t contra {referencia / 1e6:.1f} mi t oficiais ({desvio:+.2%}).",
        )

    if previous_manifest:
        prev_cov = previous_manifest["coverage"]
        atual_mes = (cov["max_year"], cov["max_month"])
        anterior_mes = (prev_cov["max_year"], prev_cov["max_month"])
        check(
            "cobertura_nao_regrediu",
            atual_mes >= anterior_mes,
            f"Último mês {atual_mes[0]}-{atual_mes[1]:02d}; versão anterior ia até "
            f"{anterior_mes[0]}-{anterior_mes[1]:02d}.",
        )

        for ano_str, prev in previous_manifest.get("yearly_totals", {}).items():
            if int(ano_str) not in complete or ano_str not in totals or prev["toneladas"] <= 0:
                continue
            variacao = totals[ano_str]["toneladas"] / prev["toneladas"] - 1
            if abs(variacao) > REVISAO_ERRO:
                check(f"revisao_{ano_str}", False, f"Toneladas de {ano_str} mudaram {variacao:+.2%}.")
            elif abs(variacao) > REVISAO_AVISO:
                check(f"revisao_{ano_str}", False, f"Toneladas de {ano_str} revisadas em {variacao:+.2%}.", "aviso")

    return results


def has_errors(results: List[CheckResult]) -> bool:
    return any(not r.ok and r.severidade == "erro" for r in results)


def as_dicts(results: List[CheckResult]) -> List[dict]:
    return [asdict(r) for r in results]
