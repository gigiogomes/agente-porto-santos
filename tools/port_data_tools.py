import os
import time
import unicodedata
from dataclasses import dataclass, asdict
from datetime import datetime
from io import StringIO
from typing import Dict, List, Optional, Tuple
import streamlit as st

import pandas as pd
import numpy as np
import requests

URL_MENSARIO = "http://mensario.portodesantos.com.br/exportarcargas/csv"
BACKUP_CSV_URL = "https://github.com/gigiogomes/agente-porto-santos/releases/download/v1.0/exportacao_cargas.csv"

TERMINAL_MAPPING = {
    "SBSA": "Santos Brasil",
    "DPW": "DP World",
    "BTP": "Brasil Terminal Portuário",
    "ECOPORTO": "Ecoporto Santos",
    "BTE": "BTE - Base de Transporte e Exportação",
    "TEV": "TEV",
    "MARIMEX": "Marimex",
    "TERMARES": "Termares",
    "TRANSBRASA": "Transbrasa",
    "LOCALFRIO": "Localfrio",
    "BANDEIRANTES-DEICMAR": "Bandeirantes-Deicmar",
}

TERMINAL_ALIASES = {
    "SBSA": ["SBSA", "SANTOS BRASIL", "SANTOS BRASIL PARTICIPACOES"],
    "DPW": ["DPW", "DP WORLD", "DPWORLD", "EMBRAPORT"],
    "BTP": ["BTP", "BRASIL TERMINAL PORTUARIO"],
    "ECOPORTO": ["ECOPORTO", "ECOPORTO SANTOS"],
    "BTE": ["BTE", "BASE DE TRANSPORTE E EXPORTACAO"],
    "TEV": ["TEV"],
    "MARIMEX": ["MARIMEX"],
    "TERMARES": ["TERMARES"],
    "TRANSBRASA": ["TRANSBRASA"],
    "LOCALFRIO": ["LOCALFRIO"],
    "BANDEIRANTES-DEICMAR": ["BANDEIRANTES-DEICMAR", "DEICMAR"],
}

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

def _normalize_text(value: str) -> str:
    if pd.isna(value) or value is None:
        return ""
    value = str(value).strip().upper()
    value = unicodedata.normalize("NFKD", value).encode("ASCII", "ignore").decode("ASCII")
    return " ".join(value.split())

def _read_csv_from_url(url: str, timeout: int = 30, retries: int = 2) -> pd.DataFrame:
    for attempt in range(retries + 1):
        try:
            response = requests.get(url, timeout=timeout, headers={"Cache-Control": "no-cache"})
            response.raise_for_status()
            return pd.read_csv(StringIO(response.text), sep=",", encoding="utf-8")
        except Exception as exc:
            if attempt == retries:
                raise RuntimeError(f"Falha ao ler CSV de {url}: {exc}")
            time.sleep(1.0)

def _canonical_terminal_sigla_from_norm(value_norm: str) -> Optional[str]:
    if not value_norm:
        return None
    for sigla, aliases in TERMINAL_ALIASES.items():
        candidates = {_normalize_text(sigla), _normalize_text(TERMINAL_MAPPING.get(sigla, sigla))}
        candidates.update({_normalize_text(alias) for alias in aliases})
        if any(c in value_norm for c in candidates if c):
            return sigla
    return None

def _normalize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["ANO"] = pd.to_numeric(df["ANO"], errors="coerce").fillna(0).astype(int)
    df["MES"] = pd.to_numeric(df["MES"], errors="coerce").fillna(0).astype(int)
    df["TERMINAIS_NORM"] = df["TERMINAIS"].apply(_normalize_text)
    df["TERMINAL_SIGLA_CANONICA"] = df["TERMINAIS_NORM"].apply(_canonical_terminal_sigla_from_norm)
    
    for col in ["TOTAL_TEU", "TOTAL_UNID"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.replace(".", "", regex=False).str.replace(",", ".", regex=False)
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    # =================================================================
    # --- REPLICAÇÃO DA LÓGICA DO POWER QUERY (CARGO MIX) ---
    colunas_base = ["MERCADORIAS", "MOVIMENTO", "TIPO_NAVEGACAO", "SENTIDO"]
    
    # Verifica se o CSV tem todas as colunas necessárias antes de aplicar
    if set(colunas_base).issubset(df.columns):
        # Limpa espaços e tira acentos das colunas base para a lógica funcionar
        for col in colunas_base:
            df[col] = df[col].apply(_normalize_text)

        # 1. Cheio vs Vazio
        cond_vazio = df["MERCADORIAS"] == "SEM CARGAS"
        estado_cv = np.where(cond_vazio, "VAZIO", "CHEIO")

        # 2. Fluxo em Cascata (if... else if...) do Power Query
        condicoes = [
            df["MOVIMENTO"] == "REMOCAO",
            df["MOVIMENTO"] == "TRANSBORDO",
            df["TIPO_NAVEGACAO"] == "CABOTAGEM",
            df["SENTIDO"] == "DESEMBARQUE"
        ]
        escolhas = ["REMOCAO", "TRANSBORDO", "CABOTAGEM", "IMPORTACAO"]
        fluxo = np.select(condicoes, escolhas, default="EXPORTACAO")

        # 3. Mescla e salva na coluna "ESTADO" (substituindo a antiga)
        df["ESTADO"] = pd.Series(fluxo, index=df.index) + " " + pd.Series(estado_cv, index=df.index)
    # =================================================================

    # Filtra apenas anos válidos e meses de 1 a 12
    df = df[(df["ANO"] > 0) & (df["MES"].between(1, 12))].copy()
    
    # JEITO SEGURO DE CRIAR DATA: string "YYYY-MM-01"
    date_strings = df["ANO"].astype(str) + "-" + df["MES"].astype(str).str.zfill(2) + "-01"
    df["DATE"] = pd.to_datetime(date_strings, errors="coerce")
    
    df = df.dropna(subset=["DATE"]).copy()
    
    # --- AGREGAÇÕES TEMPORAIS ---
    df["BIMESTRE"] = ((df["MES"] - 1) // 2 + 1).astype(int)
    df["ANO_BIMESTRE"] = df["ANO"].astype(str) + "B" + df["BIMESTRE"].astype(str)
    
    df["TRIMESTRE"] = ((df["MES"] - 1) // 3 + 1).astype(int)
    df["ANO_TRIMESTRE"] = df["ANO"].astype(str) + "T" + df["TRIMESTRE"].astype(str)
    
    df["QUADRIMESTRE"] = ((df["MES"] - 1) // 4 + 1).astype(int)
    df["ANO_QUADRIMESTRE"] = df["ANO"].astype(str) + "Q" + df["QUADRIMESTRE"].astype(str)
    
    df["SEMESTRE"] = ((df["MES"] - 1) // 6 + 1).astype(int)
    df["ANO_SEMESTRE"] = df["ANO"].astype(str) + "S" + df["SEMESTRE"].astype(str)
    
    df["ANO_MES"] = df["ANO"].astype(str) + "-" + df["MES"].astype(str).str.zfill(2)
    
    return df

def _extract_coverage(df: pd.DataFrame, source_name: str) -> DataCoverage:
    anos = df.loc[df["ANO"] > 0, "ANO"]
    meses_max = df.loc[df["ANO"] == anos.max(), "MES"] if not anos.empty else pd.Series(dtype=int)
    
    # Extrai os tipos únicos de Cargo Mix (Coluna ESTADO)
    cm_list = df["ESTADO"].dropna().unique().tolist() if "ESTADO" in df.columns else []
    
    return DataCoverage(
        source_name=source_name,
        min_year=int(anos.min()) if not anos.empty else 0,
        min_month=1,
        max_year=int(anos.max()) if not anos.empty else 0,
        max_month=int(meses_max.max()) if not meses_max.empty else 0,
        row_count=len(df),
        loaded_at=datetime.now().isoformat(timespec="seconds"),
        cargo_mix_list=cm_list  # <-- SALVA A LISTA AQUI
    )

@st.cache_data(show_spinner=False)
def load_port_data():
    diagnostics = []
    candidates = []
    for source_name, url in [("APS", URL_MENSARIO), ("BACKUP", BACKUP_CSV_URL)]:
        try:
            raw_df = _read_csv_from_url(url)
            
            # Aplica regras de negócio essenciais (só contêiner)
            if "NATUREZA_CARGA" in raw_df.columns:
                raw_df = raw_df[raw_df["NATUREZA_CARGA"].astype(str).str.strip().eq("CARGA CONTEINERIZADA")]
                
            df = _normalize_dataframe(raw_df)
            coverage = _extract_coverage(df, source_name)
            diagnostics.append({"source": source_name, "status": "success"})
            candidates.append((df, coverage))
        except Exception as exc:
            diagnostics.append({"source": source_name, "status": "error", "error": str(exc)})

    if not candidates:
        # Pega os erros reais armazenados no log de diagnósticos
        error_details = "\n".join([f"- {d['source']}: {d.get('error', 'Erro desconhecido')}" for d in diagnostics if d['status'] == 'error'])
        raise RuntimeError(f"Falha fatal ao carregar dados. Detalhes dos erros:\n{error_details}")

    selected_df, selected_coverage = max(candidates, key=lambda c: (c[1].max_year, c[1].max_month))
    return selected_df, selected_coverage, diagnostics

def query_port_data(
    df: pd.DataFrame,
    metric: str = "teus",
    start_date: str = None, 
    end_date: str = None, 
    terminals: list = None,
    cargo_mix_filter: list = None, # <-- NOVO FILTRO AQUI
    group_by: str = None,
    compare_with_previous: bool = False
) -> str:
    """Motor universal de agregação."""
    df_filtered = df.copy()

    # 1. Filtros de Período
    if start_date:
        df_filtered = df_filtered[df_filtered['DATE'] >= pd.to_datetime(start_date)]
    if end_date:
        df_filtered = df_filtered[df_filtered['DATE'] <= pd.to_datetime(end_date)]
        
    # 2. Filtros de Terminal e Cargo Mix
    if terminals and len(terminals) > 0:
        df_filtered = df_filtered[df_filtered['TERMINAL_SIGLA_CANONICA'].isin(terminals)]
        
    if cargo_mix_filter and len(cargo_mix_filter) > 0:
        filtros_norm = [_normalize_text(c) for c in cargo_mix_filter]
        estado_norm = df_filtered["ESTADO"].apply(_normalize_text)
        mask = pd.Series(False, index=df_filtered.index)
        for filtro in filtros_norm:
            if filtro in ["REMOCAO", "TRANSBORDO", "CABOTAGEM", "IMPORTACAO", "EXPORTACAO"]:
                mask |= estado_norm.str.startswith(filtro + " ")
            else:
                mask |= estado_norm.eq(filtro)
        df_filtered = df_filtered[mask]
        
    if df_filtered.empty:
        return f"[AVISO DO SISTEMA] Nenhum dado encontrado. Filtros usados: Terminais={terminals}, CargoMix={cargo_mix_filter}, Periodo={start_date} a {end_date}. O bot deve informar isso ao usuário."

    # 3. Definição da Coluna de Métrica Base
    val_col = "TOTAL_TEU" if metric in ["teus", "market_share", "cagr"] else "TOTAL_UNID"

    if compare_with_previous and not df_filtered.empty:
        max_year = df_filtered["ANO"].max()
        # Descobre qual é o último mês disponível no ano mais recente
        max_month = df_filtered[df_filtered["ANO"] == max_year]["MES"].max()
        
        # Corta os anos anteriores no mesmo mês limite
        if pd.notna(max_month):
            df_filtered = df_filtered[df_filtered["MES"] <= max_month]

# 4. Agrupamento Múltiplo (Tabela Dinâmica)
    groupby_cols = []
    if group_by:
        # Se vier só uma string, transforma em lista para não quebrar
        if isinstance(group_by, str): 
            group_by = [group_by]
            
        for gb in group_by:
            if gb == "ano": groupby_cols.append("ANO")
            elif gb == "semestre": groupby_cols.append("ANO_SEMESTRE")
            elif gb == "quadrimestre": groupby_cols.append("ANO_QUADRIMESTRE")
            elif gb == "trimestre": groupby_cols.append("ANO_TRIMESTRE")
            elif gb == "bimestre": groupby_cols.append("ANO_BIMESTRE")
            elif gb == "mes": groupby_cols.append("ANO_MES")
            elif gb == "terminal": groupby_cols.append("TERMINAL_SIGLA_CANONICA")
            elif gb == "cargo_mix": groupby_cols.append("ESTADO")

    if groupby_cols:
        aggregated = df_filtered.groupby(groupby_cols)[val_col].sum().reset_index()
    else:
        aggregated = pd.DataFrame([{"Total": df_filtered[val_col].sum()}])
        val_col = "Total"

# 5. Cálculo de Crescimento / Variação Percentual
    if compare_with_previous and not aggregated.empty:
        aggregated = aggregated.sort_values(by=groupby_cols)
        
        # Se agrupou por mais de uma coluna (ex: ["terminal", "ano"]), calcula a variação DENTRO de cada terminal
        if len(groupby_cols) > 1 and "ANO" in groupby_cols:
            cols_except_year = [c for c in groupby_cols if c != "ANO"]
            aggregated['Crescimento (%)'] = aggregated.groupby(cols_except_year)[val_col].pct_change() * 100
        else:
            aggregated['Crescimento (%)'] = aggregated[val_col].pct_change() * 100
            
        # Limpa os resultados vazios do primeiro ano e formata com %
        aggregated['Crescimento (%)'] = aggregated['Crescimento (%)'].round(2).fillna(0).astype(str) + "%"

    # 6. Cálculo de Share / Proporção Percentual
    if metric == "market_share":
        total_geral = aggregated[val_col].sum()
        if total_geral > 0:
            # Calcula a porcentagem
            aggregated['Share (%)'] = ((aggregated[val_col] / total_geral) * 100).round(2)
            # Ordena do maior para o menor para facilitar a leitura
            aggregated = aggregated.sort_values(by='Share (%)', ascending=False)
            # Formata como string com símbolo % para o bot entender direto
            aggregated['Share (%)'] = aggregated['Share (%)'].astype(str) + "%"

    if val_col in aggregated.columns:
        aggregated[val_col] = aggregated[val_col].round(0).astype(int)

    return aggregated.to_string(index=False)
