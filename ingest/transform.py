"""
Transformação da base da APS: do CSV bruto para a tabela fato e a dimensão de terminais.

Usado pelo pipeline de ingestão (GitHub Actions) e pelo app como plano B, quando
os arquivos publicados não estão disponíveis. Um único código para os dois
caminhos garante que os números sejam sempre os mesmos.
"""
import unicodedata
from functools import lru_cache
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

# Mude quando a transformação mudar: força o pipeline a republicar os dados
# mesmo que o CSV da APS seja o mesmo.
SCHEMA_VERSION = "1"

TERMINAIS_CSV = Path(__file__).resolve().parent.parent / "reference" / "terminais.csv"

NATUREZA_CONTEINER = "CARGA CONTEINERIZADA"
TERMINAL_NAO_INFORMADO = "N/D"
FLUXOS_CARGO_MIX = ["REMOCAO", "TRANSBORDO", "CABOTAGEM", "IMPORTACAO", "EXPORTACAO"]

RAW_REQUIRED_COLUMNS = [
    "ANO", "MES", "TERMINAIS", "NATUREZA_CARGA", "TIPO_NAVEGACAO", "MOVIMENTO", "SENTIDO",
    "TIPO_INSTALACAO", "CLASSENAVIO", "MERCADORIAS", "TOTAL_TEU", "TOTAL_UNID", "TOTAL_TONELADAS",
]

# Colunas categóricas do CSV -> nome na tabela fato (texto normalizado: maiúsculo, sem acento).
_DIMENSOES_TEXTO = {
    "TERMINAIS": "terminal_original",
    "NATUREZA_CARGA": "natureza_carga",
    "TIPO_NAVEGACAO": "tipo_navegacao",
    "MOVIMENTO": "movimento",
    "SENTIDO": "sentido",
    "TIPO_INSTALACAO": "tipo_instalacao",
    "CLASSENAVIO": "classe_navio",
    "MERCADORIAS": "mercadoria",
}

FATO_DIMENSOES = ["ano", "mes", "terminal", *_DIMENSOES_TEXTO.values(), "cargo_mix"]
FATO_MEDIDAS = ["teu", "unidades", "toneladas"]


def normalize_text(value) -> str:
    if value is None or pd.isna(value):
        return ""
    value = str(value).strip().upper()
    value = unicodedata.normalize("NFKD", value).encode("ASCII", "ignore").decode("ASCII")
    return " ".join(value.split())


@lru_cache(maxsize=1)
def _terminal_reference() -> pd.DataFrame:
    return pd.read_csv(TERMINAIS_CSV, dtype=str).fillna("")


@lru_cache(maxsize=1)
def terminal_aliases() -> Dict[str, str]:
    """Nome normalizado -> sigla oficial, a partir de reference/terminais.csv."""
    mapping = {}
    for row in _terminal_reference().itertuples(index=False):
        for alias in [row.sigla, row.nome, *row.aliases.split("|")]:
            if normalize_text(alias):
                mapping[normalize_text(alias)] = row.sigla
    return mapping


def canonical_terminal(value) -> str:
    """
    Sigla oficial quando o nome é um alias conhecido; senão o próprio nome
    normalizado (ex.: 'LIBRA 35'). A comparação é exata: busca por pedaço de
    nome confundia terminais diferentes ('SANTOS BRASIL (SSZ35.2 SABOO)').
    """
    value_norm = normalize_text(value)
    if not value_norm:
        return TERMINAL_NAO_INFORMADO
    return terminal_aliases().get(value_norm, value_norm)


def parse_br_number(series: pd.Series) -> pd.Series:
    """'1.234,5' -> 1234.5. Valores ausentes ou inválidos viram 0."""
    text = series.astype(str).str.replace(".", "", regex=False).str.replace(",", ".", regex=False)
    return pd.to_numeric(text, errors="coerce").fillna(0.0)


def _map_unique(series: pd.Series, func) -> pd.Series:
    # Poucas dezenas de valores distintos em ~1 milhão de linhas: aplica a função
    # uma vez por valor em vez de uma vez por linha.
    series = series.fillna("")
    return series.map({value: func(value) for value in series.unique()})


def build_fato(raw: pd.DataFrame) -> pd.DataFrame:
    """CSV bruto da APS -> tabela fato agregada por mês e por todas as dimensões."""
    missing = [c for c in RAW_REQUIRED_COLUMNS if c not in raw.columns]
    if missing:
        raise ValueError(f"Colunas ausentes no CSV da APS: {missing}")

    df = pd.DataFrame(index=raw.index)
    df["ano"] = pd.to_numeric(raw["ANO"], errors="coerce").fillna(0).astype(int)
    df["mes"] = pd.to_numeric(raw["MES"], errors="coerce").fillna(0).astype(int)
    for source, target in _DIMENSOES_TEXTO.items():
        df[target] = _map_unique(raw[source], normalize_text)
    df["terminal"] = _map_unique(raw["TERMINAIS"], canonical_terminal)

    # --- REPLICAÇÃO DA LÓGICA DO POWER QUERY (CARGO MIX) ---
    # Cargo mix só existe para contêineres; nas demais naturezas fica nulo.
    estado_cv = np.where(df["mercadoria"] == "SEM CARGAS", "VAZIO", "CHEIO")
    condicoes = [
        df["movimento"] == "REMOCAO",
        df["movimento"] == "TRANSBORDO",
        df["tipo_navegacao"] == "CABOTAGEM",
        df["sentido"] == "DESEMBARQUE",
    ]
    fluxo = np.select(condicoes, FLUXOS_CARGO_MIX[:4], default="EXPORTACAO")
    cargo_mix = pd.Series(fluxo, index=df.index) + " " + pd.Series(estado_cv, index=df.index)
    df["cargo_mix"] = cargo_mix.where(df["natureza_carga"] == NATUREZA_CONTEINER)

    df["teu"] = parse_br_number(raw["TOTAL_TEU"])
    df["unidades"] = parse_br_number(raw["TOTAL_UNID"])
    df["toneladas"] = parse_br_number(raw["TOTAL_TONELADAS"])

    df = df[(df["ano"] > 0) & (df["mes"].between(1, 12))]

    fato = df.groupby(FATO_DIMENSOES, dropna=False)[FATO_MEDIDAS].sum().reset_index()
    return fato.sort_values(FATO_DIMENSOES, na_position="first").reset_index(drop=True)


def build_dim_terminal(fato: pd.DataFrame) -> pd.DataFrame:
    """Uma linha por terminal: nome, período de atividade, naturezas e volumes totais."""
    nomes = dict(zip(_terminal_reference()["sigla"], _terminal_reference()["nome"]))
    periodo = fato["ano"] * 100 + fato["mes"]
    grouped = fato.assign(periodo=periodo).groupby("terminal")
    dim = pd.DataFrame(
        {
            "primeiro_mes": grouped["periodo"].min(),
            "ultimo_mes": grouped["periodo"].max(),
            "naturezas": grouped["natureza_carga"].agg(lambda s: "|".join(sorted(s.unique()))),
            "teu": grouped["teu"].sum(),
            "toneladas": grouped["toneladas"].sum(),
        }
    ).reset_index()
    for col in ["primeiro_mes", "ultimo_mes"]:
        dim[col] = dim[col].map(lambda p: f"{p // 100}-{p % 100:02d}")
    dim.insert(1, "nome", dim["terminal"].map(nomes).fillna(dim["terminal"]))
    return dim.sort_values(["teu", "toneladas"], ascending=False).reset_index(drop=True)
