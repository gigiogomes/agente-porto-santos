"""
Oráculo da avaliação: calcula as respostas esperadas direto na tabela fato.

Usa SQL simples e independente das ferramentas do agente, para que um erro na
ferramenta não "confirme a si mesmo". Como os valores são calculados (e não
digitados), continuam certos quando a base é atualizada.
"""
from typing import Dict, List, Optional

from ingest.transform import NATUREZA_CONTEINER


class Oracle:
    def __init__(self, store):
        self.store = store
        ultimo = int(store.sql("SELECT MAX(ano * 100 + mes) AS p FROM fato")["p"].iloc[0])
        self.max_year, self.max_month = divmod(ultimo, 100)

    def _where(
        self,
        medida: str,
        ano: Optional[int] = None,
        de: Optional[int] = None,
        ate: Optional[int] = None,
        terminais: Optional[List[str]] = None,
        naturezas: Optional[List[str]] = None,
        mercadorias: Optional[List[str]] = None,
        sentido: Optional[str] = None,
        navegacao: Optional[str] = None,
        cargo_mix: Optional[str] = None,
        meses: Optional[List[int]] = None,
    ):
        """de/ate no formato AAAAMM; mercadorias e cargo_mix casam por trecho do nome."""
        conds, params = [], []
        if ano:
            de, ate = ano * 100 + 1, ano * 100 + 12
        if de:
            conds.append("ano * 100 + mes >= ?")
            params.append(de)
        if ate:
            conds.append("ano * 100 + mes <= ?")
            params.append(ate)
        if meses:
            conds.append(f"mes IN ({', '.join('?' * len(meses))})")
            params += meses
        if naturezas is None and medida in ("teu", "unidades"):
            naturezas = [NATUREZA_CONTEINER]
        if naturezas:
            conds.append(f"natureza_carga IN ({', '.join('?' * len(naturezas))})")
            params += naturezas
        if terminais:
            conds.append(f"terminal IN ({', '.join('?' * len(terminais))})")
            params += terminais
        if mercadorias:
            conds.append("(" + " OR ".join("mercadoria LIKE ?" for _ in mercadorias) + ")")
            params += [f"%{m}%" for m in mercadorias]
        if sentido:
            conds.append("sentido = ?")
            params.append(sentido)
        if navegacao:
            conds.append("tipo_navegacao = ?")
            params.append(navegacao)
        if cargo_mix:
            conds.append("cargo_mix LIKE ?")
            params.append(f"{cargo_mix}%")
        return ("WHERE " + " AND ".join(conds)) if conds else "", params

    def total(self, medida: str = "teu", **filtros) -> float:
        where, params = self._where(medida, **filtros)
        return float(self.store.sql(f"SELECT COALESCE(SUM({medida}), 0) AS v FROM fato {where}", params)["v"].iloc[0])

    def por(self, dim: str, medida: str = "teu", **filtros) -> Dict[str, float]:
        where, params = self._where(medida, **filtros)
        df = self.store.sql(
            f"SELECT {dim} AS k, SUM({medida}) AS v FROM fato {where} GROUP BY 1 ORDER BY 2 DESC", params
        )
        return {str(k): float(v) for k, v in zip(df["k"], df["v"])}

    def shares(self, dim: str, medida: str = "teu", **filtros) -> Dict[str, float]:
        valores = self.por(dim, medida, **filtros)
        total = sum(valores.values())
        return {k: round(v / total * 100, 2) for k, v in valores.items()}

    def variacao_pct(self, antes: float, depois: float) -> float:
        return round((depois / antes - 1) * 100, 2)

    def yoy_pct(self, ano_anterior: int, ano: int, medida: str = "teu", ate_mes: int = 12, **filtros) -> float:
        meses = list(range(1, ate_mes + 1))
        antes = self.total(medida, ano=ano_anterior, meses=meses, **filtros)
        depois = self.total(medida, ano=ano, meses=meses, **filtros)
        return self.variacao_pct(antes, depois)

    def cagr_pct(self, ano_inicial: int, ano_final: int, medida: str = "teu", **filtros) -> float:
        v0 = self.total(medida, ano=ano_inicial, **filtros)
        vn = self.total(medida, ano=ano_final, **filtros)
        return round(((vn / v0) ** (1 / (ano_final - ano_inicial)) - 1) * 100, 2)
