import json
import logging
import re
import time
from dataclasses import dataclass, field
from datetime import datetime
from io import BytesIO
from typing import List, Optional, Tuple

import duckdb
import pandas as pd
import requests
import streamlit as st

from ingest import sources
from ingest.transform import (
    FLUXOS_CARGO_MIX,
    NATUREZA_CONTEINER,
    build_fato,
    canonical_terminal,
    normalize_text,
)

logger = logging.getLogger(__name__)

URL_MENSARIO = sources.APS_CSV_URL
BACKUP_CSV_URL = sources.BACKUP_CSV_URL
PUBLISHED_FATO_URL = f"{sources.PUBLISHED_BASE_URL}/{sources.FATO_FILE}"
PUBLISHED_MANIFEST_URL = f"{sources.PUBLISHED_BASE_URL}/{sources.MANIFEST_FILE}"

# Medida -> (coluna no resultado da consulta, rótulo exibido nas tabelas)
MEDIDAS = {
    "teu": ("TOTAL_TEU", "TEU"),
    "unidades": ("TOTAL_UNID", "Unidades"),
    "toneladas": ("TOTAL_TON", "Toneladas"),
}
METRICA_MEDIDA_PADRAO = {
    "teus": "teu",
    "conteineres": "unidades",
    "toneladas": "toneladas",
    "market_share": "teu",
    "cagr": "teu",
}

# Dimensões opcionais da tabela fato: só entram na consulta quando pedidas no
# group_by, para manter o resultado pequeno. Chave = valor aceito no group_by.
OPTIONAL_DIMENSIONS = {
    "natureza_carga": "natureza_carga",
    "mercadoria": "mercadoria",
    "sentido": "sentido",
    "navegacao": "tipo_navegacao",
}
GROUP_BY_COLUMNS = {
    "ano": "ANO",
    "semestre": "ANO_SEMESTRE",
    "quadrimestre": "ANO_QUADRIMESTRE",
    "trimestre": "ANO_TRIMESTRE",
    "bimestre": "ANO_BIMESTRE",
    "mes": "ANO_MES",
    "terminal": "TERMINAL_SIGLA_CANONICA",
    "cargo_mix": "ESTADO",
    **{key: column.upper() for key, column in OPTIONAL_DIMENSIONS.items()},
}
NON_TEMPORAL_GROUP_BY = ["terminal", "cargo_mix", *OPTIONAL_DIMENSIONS]


@dataclass
class DataCoverage:
    source_name: str
    min_year: int
    min_month: int
    max_year: int
    max_month: int
    row_count: int
    loaded_at: str
    cargo_mix_list: list
    terminal_list: list
    mercadoria_list: list = field(default_factory=list)
    natureza_list: list = field(default_factory=list)
    generated_at: Optional[str] = None


# =================================================================
# CARGA DOS DADOS (arquivos publicados -> APS -> backup)
# =================================================================
def _read_bytes(url: str, timeout: int = 60, retries: int = 2) -> bytes:
    for attempt in range(retries + 1):
        try:
            response = requests.get(url, timeout=timeout, headers={"Cache-Control": "no-cache"})
            response.raise_for_status()
            return response.content
        except Exception as exc:
            if attempt == retries:
                raise RuntimeError(f"Falha ao ler {url}: {exc}")
            time.sleep(1.0)


def _load_published() -> Tuple[pd.DataFrame, Optional[str]]:
    """Tabela fato já processada pelo pipeline de ingestão (~2 MB)."""
    fato = pd.read_parquet(BytesIO(_read_bytes(PUBLISHED_FATO_URL)))
    try:
        generated_at = json.loads(_read_bytes(PUBLISHED_MANIFEST_URL)).get("generated_at")
    except Exception:
        generated_at = None
    return fato, generated_at


def _load_csv(url: str) -> Tuple[pd.DataFrame, Optional[str]]:
    """Plano B: baixa o CSV bruto (~180 MB) e aplica a mesma transformação do pipeline."""
    raw = pd.read_csv(BytesIO(_read_bytes(url)), low_memory=False)
    return build_fato(raw), None


# Ordem de prioridade. Cada fonte só é tentada se a anterior falhar.
DATA_SOURCES = [
    ("PUBLICADO", _load_published),
    ("APS", lambda: _load_csv(URL_MENSARIO)),
    ("BACKUP", lambda: _load_csv(BACKUP_CSV_URL)),
]


def _extract_coverage(fato: pd.DataFrame, source_name: str, generated_at: Optional[str] = None) -> DataCoverage:
    anos = fato.loc[fato["ano"] > 0, "ano"]
    meses_max = fato.loc[fato["ano"] == anos.max(), "mes"] if not anos.empty else pd.Series(dtype=int)

    cm_list = fato.loc[fato["natureza_carga"] == NATUREZA_CONTEINER, "cargo_mix"].dropna().unique().tolist()
    # Ordenada por volume, para o prompt listar primeiro os terminais relevantes.
    terminal_list = (
        fato.groupby("terminal")[["teu", "toneladas"]].sum().sort_values(["teu", "toneladas"], ascending=False).index.tolist()
    )

    return DataCoverage(
        source_name=source_name,
        min_year=int(anos.min()) if not anos.empty else 0,
        min_month=1,
        max_year=int(anos.max()) if not anos.empty else 0,
        max_month=int(meses_max.max()) if not meses_max.empty else 0,
        row_count=len(fato),
        loaded_at=datetime.now().isoformat(timespec="seconds"),
        cargo_mix_list=cm_list,
        terminal_list=terminal_list,
        mercadoria_list=sorted(m for m in fato["mercadoria"].unique() if m),
        natureza_list=sorted(n for n in fato["natureza_carga"].unique() if n),
        generated_at=generated_at,
    )


def _load_port_data_uncached():
    diagnostics = []
    for source_name, loader in DATA_SOURCES:
        try:
            fato, generated_at = loader()
            coverage = _extract_coverage(fato, source_name, generated_at)
            diagnostics.append({"source": source_name, "status": "success"})
            if source_name != DATA_SOURCES[0][0]:
                logger.warning("Fonte principal indisponível; usando %s.", source_name)
            return fato, coverage, diagnostics
        except Exception as exc:
            logger.warning("Falha ao carregar a fonte %s: %s", source_name, exc)
            diagnostics.append({"source": source_name, "status": "error", "error": str(exc)})

    error_details = "\n".join(
        [f"- {d['source']}: {d.get('error', 'Erro desconhecido')}" for d in diagnostics if d["status"] == "error"]
    )
    raise RuntimeError(f"Falha fatal ao carregar dados. Detalhes dos erros:\n{error_details}")


@st.cache_data(show_spinner=False)
def load_port_data():
    return _load_port_data_uncached()


class PortDataStore:
    """
    Tabela fato em um banco DuckDB em memória.

    A instância é compartilhada entre as sessões do Streamlit, que rodam em
    threads diferentes; por isso cada consulta abre o próprio cursor.
    """

    def __init__(self, fato: pd.DataFrame):
        self._con = duckdb.connect()
        self._con.register("fato_df", fato)
        self._con.execute("CREATE TABLE fato AS SELECT * FROM fato_df")
        self._con.unregister("fato_df")
        self.mercadorias = [m for m in self.sql("SELECT DISTINCT mercadoria FROM fato ORDER BY 1")["mercadoria"] if m]

    def sql(self, query: str, params: Optional[list] = None) -> pd.DataFrame:
        cursor = self._con.cursor()
        try:
            return cursor.execute(query, params or []).df()
        finally:
            cursor.close()


# =================================================================
# FILTROS
# =================================================================
def _parse_month_bound(value: str, is_end: bool) -> pd.Timestamp:
    """
    Converte 'YYYY', 'YYYY-MM' ou 'YYYY-MM-DD' no primeiro dia do mês. A base é
    mensal, então o dia é ignorado — isso aceita datas como '2026-06-31', que o
    LLM gera com frequência para "fim do mês".
    """
    match = re.match(r"^\s*(\d{4})(?:-(\d{1,2})(?:-\d{1,2})?)?\s*$", str(value))
    if not match:
        raise ValueError(f"Data inválida: '{value}'. Use YYYY, YYYY-MM ou YYYY-MM-DD.")
    year = int(match.group(1))
    month = int(match.group(2)) if match.group(2) else (12 if is_end else 1)
    if not 1 <= month <= 12:
        raise ValueError(f"Mês inválido em '{value}'.")
    return pd.Timestamp(year=year, month=month, day=1)


def _resolve_mercadorias(store: PortDataStore, filtros: List[str]) -> List[str]:
    """Nome exato quando existe; senão todas as mercadorias que contêm o termo ('SOJA')."""
    resolved = []
    for filtro in filtros:
        termo = normalize_text(filtro)
        matches = [termo] if termo in store.mercadorias else [m for m in store.mercadorias if termo and termo in m]
        resolved.extend(m for m in matches if m not in resolved)
    return resolved


def _in_clause(column: str, values: List[str]) -> Tuple[str, list]:
    return f"{column} IN ({', '.join('?' for _ in values)})", list(values)


def _build_where(
    start_date, end_date, naturezas, terminals, cargo_mix_filter, mercadorias, sentidos
) -> Tuple[List[str], list, List[str], list]:
    """
    Monta o WHERE em duas partes: o escopo (período + natureza), usado também
    para medir a completude dos anos no CAGR, e os demais filtros. Os valores
    vão sempre como parâmetros (?), nunca colados no texto do SQL.
    """
    scope, scope_params = [], []
    if start_date:
        bound = _parse_month_bound(start_date, is_end=False)
        scope.append("ano * 100 + mes >= ?")
        scope_params.append(bound.year * 100 + bound.month)
    if end_date:
        bound = _parse_month_bound(end_date, is_end=True)
        scope.append("ano * 100 + mes <= ?")
        scope_params.append(bound.year * 100 + bound.month)
    if naturezas:
        clause, params = _in_clause("natureza_carga", [normalize_text(n) for n in naturezas])
        scope.append(clause)
        scope_params += params

    filters, filter_params = [], []
    if terminals:
        # Aceita sigla ou nome por extenso ("Santos Brasil" -> "SBSA").
        clause, params = _in_clause("terminal", [canonical_terminal(t) for t in terminals])
        filters.append(clause)
        filter_params += params
    if cargo_mix_filter:
        conds = []
        for filtro in [normalize_text(c) for c in cargo_mix_filter]:
            if filtro in FLUXOS_CARGO_MIX:
                conds.append("cargo_mix LIKE ?")
                filter_params.append(filtro + " %")
            else:
                conds.append("cargo_mix = ?")
                filter_params.append(filtro)
        filters.append("(" + " OR ".join(conds) + ")")
    if mercadorias is not None:
        clause, params = _in_clause("mercadoria", mercadorias)
        filters.append(clause)
        filter_params += params
    if sentidos:
        clause, params = _in_clause("sentido", [normalize_text(s) for s in sentidos])
        filters.append(clause)
        filter_params += params

    return scope, scope_params, filters, filter_params


def _where_sql(clauses: List[str]) -> str:
    return ("WHERE " + " AND ".join(clauses)) if clauses else ""


def _add_period_columns(df: pd.DataFrame) -> pd.DataFrame:
    df["ANO_BIMESTRE"] = df["ANO"].astype(str) + "B" + ((df["MES"] - 1) // 2 + 1).astype(str)
    df["ANO_TRIMESTRE"] = df["ANO"].astype(str) + "T" + ((df["MES"] - 1) // 3 + 1).astype(str)
    df["ANO_QUADRIMESTRE"] = df["ANO"].astype(str) + "Q" + ((df["MES"] - 1) // 4 + 1).astype(str)
    df["ANO_SEMESTRE"] = df["ANO"].astype(str) + "S" + ((df["MES"] - 1) // 6 + 1).astype(str)
    df["ANO_MES"] = df["ANO"].astype(str) + "-" + df["MES"].astype(str).str.zfill(2)
    return df


def _reduce(store: PortDataStore, clauses: List[str], params: list, extra_dims: List[str]) -> pd.DataFrame:
    """
    Filtra e soma no DuckDB, devolvendo uma tabela pequena (mês x terminal x
    cargo mix x dimensões pedidas) para os cálculos de share, YoY e CAGR.
    Os nomes de coluna vêm de OPTIONAL_DIMENSIONS, nunca da entrada do usuário.
    """
    extra_select = "".join(f", {col} AS {col.upper()}" for col in extra_dims)
    df = store.sql(
        f"""
        SELECT ano AS ANO, mes AS MES, terminal AS TERMINAL_SIGLA_CANONICA, cargo_mix AS ESTADO{extra_select},
               SUM(teu)::DOUBLE AS TOTAL_TEU,
               SUM(unidades)::DOUBLE AS TOTAL_UNID,
               SUM(toneladas)::DOUBLE AS TOTAL_TON
        FROM fato
        {_where_sql(clauses)}
        GROUP BY ALL
        """,
        params,
    )
    return _add_period_columns(df)


# =================================================================
# CAGR DETERMINÍSTICO (calculado no código, nunca pelo LLM)
# =================================================================
def _detect_complete_years(store: PortDataStore, scope: List[str], scope_params: list) -> set:
    """
    Um ano é 'completo' quando a base tem os 12 meses presentes. Avaliado sobre
    o escopo inteiro (período + natureza), não por terminal, porque completude é
    propriedade da fonte de dados, não da atividade do terminal.
    """
    months = store.sql(
        f"SELECT ano, COUNT(DISTINCT mes) AS n FROM fato {_where_sql(scope)} GROUP BY ano", scope_params
    )
    return {int(row.ano) for row in months.itertuples() if row.n >= 12}


def _compute_cagr(complete_years: set, df_filtered: pd.DataFrame, group_by, val_col: str, label: str) -> str:
    """
    CAGR = (valor_final / valor_inicial) ** (1 / anos) - 1, usando os endpoints
    de ANOS COMPLETOS dentro da janela. Retorna tabela transparente com os anos
    e volumes efetivamente usados, para o bot repassar sem inventar.
    """
    if len(complete_years) < 2:
        return (
            "[AVISO DO SISTEMA] Não há anos completos suficientes (>= 2) para calcular o CAGR "
            f"na janela informada. Anos completos detectados: {sorted(complete_years) or 'nenhum'}. "
            "O bot deve informar isso ao usuário e NÃO deve estimar o valor por conta própria."
        )

    first_year = min(complete_years)
    last_year = max(complete_years)
    n = last_year - first_year

    dfc = df_filtered[df_filtered["ANO"].isin(complete_years)]

    gb = [group_by] if isinstance(group_by, str) else (group_by or [])
    by_terminal = "terminal" in gb

    if by_terminal:
        groups = [(sigla, g) for sigla, g in dfc.groupby("TERMINAL_SIGLA_CANONICA")]
    else:
        groups = [(None, dfc)]

    rows = []
    for terminal, g in groups:
        yearly = g.groupby("ANO")[val_col].sum()
        v0 = yearly.get(first_year)
        vn = yearly.get(last_year)

        row = {}
        if by_terminal:
            row["Terminal"] = terminal if terminal is not None else "N/D"
        row["Ano Inicial"] = first_year
        row[f"{label} Inicial"] = int(round(v0)) if v0 is not None else None
        row["Ano Final"] = last_year
        row[f"{label} Final"] = int(round(vn)) if vn is not None else None
        row["Anos"] = n

        if v0 is None or vn is None or v0 <= 0:
            row["CAGR (%)"] = "N/D (sem movimentação no ano inicial)"
        else:
            cagr = (vn / v0) ** (1.0 / n) - 1.0
            row["CAGR (%)"] = f"{round(cagr * 100, 2)}%"
        rows.append(row)

    result = pd.DataFrame(rows)
    note = (
        f"[CAGR calculado deterministicamente sobre anos completos: {first_year} a {last_year} "
        f"({n} ano{'s' if n != 1 else ''} de variação). Os endpoints definem o CAGR; "
        f"anos intermediários não alteram o cálculo.]\n"
    )
    return note + result.to_string(index=False)


def _compute_yoy(df_filtered: pd.DataFrame, group_by, val_col: str, label: str) -> str:
    """
    YoY determinístico. Sempre separa os anos em COLUNAS (ano_anterior /
    ano_atual / Diferença / YoY %), independentemente de o modelo ter incluído
    'ano' no group_by. Se o ano atual for parcial, faz a comparação acumulada
    (YTD) e DIZ isso explicitamente, em vez de rotular silenciosamente um valor
    parcial como se fosse o ano cheio.
    """
    years = sorted(int(y) for y in df_filtered["ANO"].unique())
    if len(years) < 2:
        return (
            f"[AVISO DO SISTEMA] YoY precisa de dois anos na janela, mas só há: {years}. "
            "O bot deve pedir um período que inclua o ano anterior e NÃO deve inventar valores."
        )
    curr, prev = years[-1], years[-2]

    curr_max_month = int(df_filtered[df_filtered["ANO"] == curr]["MES"].max())
    full_year = curr_max_month >= 12

    # Comparação justa: limita ambos os anos aos meses disponíveis no ano atual.
    dft = df_filtered[
        (df_filtered["ANO"].isin([prev, curr])) & (df_filtered["MES"] <= curr_max_month)
    ]

    gb = [group_by] if isinstance(group_by, str) else (group_by or [])
    # group_by temporais (ano/mes/...) são ignorados: o YoY já é por ano.
    dim_cols = [GROUP_BY_COLUMNS[g] for g in gb if g in NON_TEMPORAL_GROUP_BY]

    if dim_cols:
        grouped = dft.groupby(dim_cols + ["ANO"])[val_col].sum().reset_index()
        pivot = grouped.pivot_table(
            index=dim_cols, columns="ANO", values=val_col, fill_value=0
        ).reset_index()
    else:
        s = dft.groupby("ANO")[val_col].sum()
        pivot = pd.DataFrame([{prev: float(s.get(prev, 0)), curr: float(s.get(curr, 0))}])

    for y in (prev, curr):
        if y not in pivot.columns:
            pivot[y] = 0.0

    pivot["Diferença"] = (pivot[curr] - pivot[prev]).round(0).astype(int)
    pivot["YoY (%)"] = pivot.apply(
        lambda r: f"{round((r[curr] / r[prev] - 1) * 100, 2)}%"
        if r[prev] > 0
        else "N/D (sem base no ano anterior)",
        axis=1,
    )
    pivot[prev] = pivot[prev].round(0).astype(int)
    pivot[curr] = pivot[curr].round(0).astype(int)
    pivot = pivot.rename(columns={prev: f"{label} {prev}", curr: f"{label} {curr}"})

    if dim_cols:
        pivot = pivot.sort_values(by=f"{label} {curr}", ascending=False)

    if full_year:
        note = f"[YoY ano cheio: {prev} vs {curr}.]\n"
    else:
        note = (
            f"[YoY ACUMULADO/YTD: meses 1 a {curr_max_month}. O ano {curr} só tem dados até o mês "
            f"{curr_max_month}, então {prev} também foi limitado aos meses 1 a {curr_max_month} para "
            f"comparação justa. ATENÇÃO: estes NÃO são valores de ano cheio — informe isso ao usuário.]\n"
        )
    return note + pivot.to_string(index=False)


def query_port_data(
    store: PortDataStore,
    metric: str = "teus",
    start_date: str = None,
    end_date: str = None,
    terminals: list = None,
    cargo_mix_filter: list = None,
    group_by: str = None,
    compare_with_previous: bool = False,
    medida: str = None,
    natureza_carga: list = None,
    mercadorias: list = None,
    sentido: list = None,
) -> str:
    """Motor universal de agregação."""
    # 1. Medida e escopo. TEU e unidades de contêiner só fazem sentido para carga
    # conteinerizada (carga geral também conta "unidades", ex.: veículos); em
    # toneladas o padrão é o porto inteiro.
    medida = medida or METRICA_MEDIDA_PADRAO.get(metric, "unidades")
    if medida not in MEDIDAS:
        raise ValueError(f"Medida inválida: '{medida}'. Use {list(MEDIDAS)}.")
    val_col, label = MEDIDAS[medida]
    naturezas = natureza_carga or ([NATUREZA_CONTEINER] if medida in ("teu", "unidades") else None)

    mercadorias_resolvidas = None
    if mercadorias:
        mercadorias_resolvidas = _resolve_mercadorias(store, mercadorias)
        if not mercadorias_resolvidas:
            return (
                f"[AVISO DO SISTEMA] Mercadoria não encontrada: {mercadorias}. "
                f"Valores válidos: {', '.join(store.mercadorias)}. O bot deve informar isso ao usuário."
            )

    # Filtros fora do padrão histórico (só contêiner, em TEU/unidades) são
    # descritos no resultado, para o bot saber exatamente o que foi somado.
    escopo = []
    if medida == "toneladas" or natureza_carga:
        escopo.append(f"naturezas={', '.join(naturezas) if naturezas else 'todas'}")
    if mercadorias_resolvidas:
        escopo.append(f"mercadorias={', '.join(mercadorias_resolvidas)}")
    if sentido:
        escopo.append(f"sentido={', '.join(sentido)}")
    nota_escopo = f"[Escopo: medida={label}; {'; '.join(escopo)}]\n" if escopo else ""

    # 2. Filtragem e soma no DuckDB
    gb = [group_by] if isinstance(group_by, str) else (group_by or [])
    extra_dims = [OPTIONAL_DIMENSIONS[g] for g in gb if g in OPTIONAL_DIMENSIONS]
    scope, scope_params, filters, filter_params = _build_where(
        start_date, end_date, naturezas, terminals, cargo_mix_filter, mercadorias_resolvidas, sentido
    )
    df_filtered = _reduce(store, scope + filters, scope_params + filter_params, extra_dims)

    if df_filtered.empty:
        extras = f" Escopo: {'; '.join(escopo)}." if escopo else ""
        return (
            f"[AVISO DO SISTEMA] Nenhum dado encontrado. Filtros usados: Terminais={terminals}, "
            f"CargoMix={cargo_mix_filter}, Periodo={start_date} a {end_date}.{extras} O bot deve informar isso ao usuário."
        )

    # 2.5 CAGR é determinístico: calcula no código e retorna (não cai no fluxo de soma).
    if metric == "cagr":
        complete_years = _detect_complete_years(store, scope, scope_params)
        return nota_escopo + _compute_cagr(complete_years, df_filtered, gb, val_col, label)

    # 3.5 YoY é determinístico: pivota por ano em colunas e rotula a base
    # (acumulado/YTD vs ano cheio). Retorna sem cair no fluxo genérico de soma.
    if compare_with_previous:
        return nota_escopo + _compute_yoy(df_filtered, gb, val_col, label)

    # 4. Agrupamento Múltiplo (Tabela Dinâmica)
    groupby_cols = [GROUP_BY_COLUMNS[g] for g in gb if g in GROUP_BY_COLUMNS]

    if groupby_cols:
        aggregated = df_filtered.groupby(groupby_cols)[val_col].sum().reset_index()
    else:
        aggregated = pd.DataFrame([{"Total": df_filtered[val_col].sum()}])
        val_col = "Total"

    # 5. Cálculo de Share / Proporção Percentual
    if metric == "market_share":
        total_geral = aggregated[val_col].sum()
        if total_geral > 0:
            aggregated["Share (%)"] = ((aggregated[val_col] / total_geral) * 100).round(2)
            # Ordena pelo volume exato: o share arredondado empata itens pequenos (0.0%).
            aggregated = aggregated.sort_values(by=val_col, ascending=False)
            aggregated["Share (%)"] = aggregated["Share (%)"].astype(str) + "%"

    if val_col in aggregated.columns:
        aggregated[val_col] = aggregated[val_col].round(0).astype(int)

    return nota_escopo + aggregated.to_string(index=False)
