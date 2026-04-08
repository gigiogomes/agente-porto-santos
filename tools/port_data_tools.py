import os
import uuid
import unicodedata
from dataclasses import dataclass, asdict
from datetime import datetime
from io import StringIO
from typing import Optional, List, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import requests


URL_MENSARIO = "http://mensario.portodesantos.com.br/exportarcargas/csv"
BACKUP_CSV_URL = "https://github.com/gigiogomes/agente-porto-santos/releases/download/v1.0/exportacao_cargas.csv"

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEMP_GRAPH_DIR = os.path.join(BASE_DIR, "temp_graphs")
os.makedirs(TEMP_GRAPH_DIR, exist_ok=True)

TERMINAL_MAPPING = {
    "SBSA": "Santos Brasil",
    "DPW": "DP World",
    "BTP": "Brasil Terminal Portuário",
    "ECOPORTO": "Ecoporto Santos",
    "BTE": "BTE - Base de Transporte e Exportação"
}

TERMINAL_ALIASES = {
    "SBSA": ["SBSA", "SANTOS BRASIL", "SANTOS BRASIL PARTICIPACOES"],
    "DPW": ["DPW", "DP WORLD", "DPWORLD", "EMBRAPORT"],
    "BTP": ["BTP", "BRASIL TERMINAL PORTUARIO", "BRASIL TERMINAL PORTUÁRIO"],
    "ECOPORTO": ["ECOPORTO", "ECOPORTO SANTOS"],
    "BTE": ["BTE", "BASE DE TRANSPORTE E EXPORTACAO", "BASE DE TRANSPORTE E EXPORTAÇÃO"],
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


def _normalize_text(value: str) -> str:
    if value is None:
        return ""
    value = str(value).strip().upper()
    value = unicodedata.normalize("NFKD", value).encode("ASCII", "ignore").decode("ASCII")
    return " ".join(value.split())


def _read_csv_from_url(url: str, timeout: int = 20) -> pd.DataFrame:
    response = requests.get(
        url,
        timeout=timeout,
        headers={"Cache-Control": "no-cache"}
    )
    response.raise_for_status()
    return pd.read_csv(StringIO(response.text), sep=",", encoding="utf-8")


def _apply_business_rules(df: pd.DataFrame) -> pd.DataFrame:
    print(f"🔍 [DEBUG] Colunas encontradas no CSV: {list(df.columns)}")

    if "NATUREZA_CARGA" in df.columns:
        linhas_antes = len(df)
        df = df[df["NATUREZA_CARGA"] == "CARGA CONTEINERIZADA"].copy()
        linhas_depois = len(df)
        print(
            f"✅ Filtro de Carga Conteinerizada aplicado: "
            f"de {linhas_antes} para {linhas_depois} linhas."
        )
    else:
        print("⚠️ AVISO: Coluna 'NATUREZA_CARGA' não encontrada. O filtro não pôde ser aplicado.")

    return _normalize_dataframe(df)


def _normalize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    if "ANO" in df.columns:
        df["ANO"] = pd.to_numeric(df["ANO"], errors="coerce").fillna(0).astype(int)

    if "MES" in df.columns:
        df["MES"] = pd.to_numeric(df["MES"], errors="coerce").fillna(0).astype(int)

    if "TERMINAIS" in df.columns:
        df["TERMINAIS"] = df["TERMINAIS"].astype(str).str.strip()
        df["TERMINAIS_NORM"] = df["TERMINAIS"].apply(_normalize_text)

        print(f"🔍 [DEBUG] Alguns terminais encontrados no CSV: {df['TERMINAIS'].unique()[:15]}")

        df["Nome_Terminal_Completo"] = df["TERMINAIS_NORM"].map(
            {_normalize_text(k): v for k, v in TERMINAL_MAPPING.items()}
        ).fillna(df["TERMINAIS"])

        df["Nome_Terminal_Completo_NORM"] = df["Nome_Terminal_Completo"].apply(_normalize_text)

    for col in ["TOTAL_TEU", "TOTAL_UNID", "TOTAL_TONELADAS"]:
        if col in df.columns:
            df[col] = (
                df[col]
                .astype(str)
                .str.replace(".", "", regex=False)
                .str.replace(",", ".", regex=False)
            )
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    return df


def _extract_coverage(df: pd.DataFrame, source_name: str) -> DataCoverage:
    anos_validos = df.loc[df["ANO"] > 0, "ANO"]

    min_year = int(anos_validos.min()) if not anos_validos.empty else 0
    max_year = int(anos_validos.max()) if not anos_validos.empty else 0

    meses_min_year = df.loc[df["ANO"] == min_year, "MES"] if min_year > 0 else pd.Series(dtype=int)
    meses_max_year = df.loc[df["ANO"] == max_year, "MES"] if max_year > 0 else pd.Series(dtype=int)

    min_month = int(meses_min_year.min()) if not meses_min_year.empty else 0
    max_month = int(meses_max_year.max()) if not meses_max_year.empty else 0

    return DataCoverage(
        source_name=source_name,
        min_year=min_year,
        min_month=min_month,
        max_year=max_year,
        max_month=max_month,
        row_count=len(df),
        loaded_at=datetime.now().isoformat(timespec="seconds")
    )


def _coverage_sort_key(item: Tuple[pd.DataFrame, DataCoverage]):
    _, coverage = item
    return (coverage.max_year, coverage.max_month, coverage.row_count)


def load_port_data():
    diagnostics = []
    candidates = []

    sources = [
        ("APS", URL_MENSARIO),
        ("BACKUP", BACKUP_CSV_URL),
    ]

    for source_name, url in sources:
        try:
            raw_df = _read_csv_from_url(url)
            df = _apply_business_rules(raw_df)
            coverage = _extract_coverage(df, source_name)

            diagnostics.append({
                "source": source_name,
                "status": "success",
                "coverage": asdict(coverage)
            })

            print(
                f"✅ Fonte {source_name} carregada com cobertura até "
                f"{coverage.max_month:02d}/{coverage.max_year}"
            )
            candidates.append((df, coverage))

        except Exception as e:
            diagnostics.append({
                "source": source_name,
                "status": "error",
                "error": str(e)
            })
            print(f"⚠️ Falha ao carregar {source_name}: {e}")

    if not candidates:
        raise RuntimeError("Não foi possível carregar nem a APS nem o backup.")

    selected_df, selected_coverage = max(candidates, key=_coverage_sort_key)

    diagnostics.append({
        "selected_source": selected_coverage.source_name,
        "selected_max_year": selected_coverage.max_year,
        "selected_max_month": selected_coverage.max_month
    })

    print(
        f"✅ Fonte selecionada: {selected_coverage.source_name} "
        f"({selected_coverage.max_month:02d}/{selected_coverage.max_year})"
    )

    return selected_df, selected_coverage, diagnostics


def get_data_coverage_text(coverage: DataCoverage, diagnostics: list) -> str:
    status_fontes = []
    for item in diagnostics:
        if item.get("status") == "success":
            cov = item.get("coverage", {})
            status_fontes.append(
                f"- {item['source']}: ok, até {cov.get('max_month', 0):02d}/{cov.get('max_year', 0)}"
            )
        elif item.get("status") == "error":
            status_fontes.append(f"- {item['source']}: erro ({item.get('error', 'sem detalhe')})")

    status_fontes_str = "\n".join(status_fontes) if status_fontes else "- Sem diagnóstico disponível"

    return (
        f"Fonte ativa: {coverage.source_name}\n"
        f"Cobertura: {coverage.min_month:02d}/{coverage.min_year} até {coverage.max_month:02d}/{coverage.max_year}\n"
        f"Linhas: {coverage.row_count:,}\n"
        f"Carregado em: {coverage.loaded_at}\n"
        f"Diagnóstico das fontes:\n{status_fontes_str}"
    )


def _resolve_terminal_aliases(termo_busca: str) -> List[str]:
    termo_norm = _normalize_text(termo_busca)

    aliases = {termo_norm}

    for key, values in TERMINAL_ALIASES.items():
        key_norm = _normalize_text(key)
        values_norm = {_normalize_text(v) for v in values}

        if termo_norm == key_norm or termo_norm in values_norm:
            aliases.update(values_norm)
            aliases.add(key_norm)

            nome_completo = TERMINAL_MAPPING.get(key, key)
            aliases.add(_normalize_text(nome_completo))

    nome_extenso = TERMINAL_MAPPING.get(termo_busca.upper(), termo_busca)
    aliases.add(_normalize_text(nome_extenso))

    return list(aliases)


def _encontrar_terminal(df: pd.DataFrame, termo_busca: str) -> pd.Series:
    if "TERMINAIS_NORM" not in df.columns:
        raise ValueError("Coluna 'TERMINAIS_NORM' não encontrada no DataFrame.")

    aliases = _resolve_terminal_aliases(termo_busca)

    mask = pd.Series(False, index=df.index)

    for alias in aliases:
        mask = (
            mask
            | (df["TERMINAIS_NORM"] == alias)
            | df["TERMINAIS_NORM"].str.contains(alias, case=False, na=False)
        )

        if "Nome_Terminal_Completo_NORM" in df.columns:
            mask = (
                mask
                | (df["Nome_Terminal_Completo_NORM"] == alias)
                | df["Nome_Terminal_Completo_NORM"].str.contains(alias, case=False, na=False)
            )

    return mask


def get_volume(df: pd.DataFrame, terminal_sigla: str, ano: int, meses: list = None) -> str:
    mask = df["ANO"] == ano

    if meses and len(meses) > 0:
        mask &= df["MES"].isin(meses)

    if terminal_sigla.upper() != "PORTO":
        mask &= _encontrar_terminal(df, terminal_sigla)

    df_filtrado = df[mask]

    if df_filtrado.empty:
        mes_str = f" nos meses {meses}" if meses else ""
        return f"Sem dados para {terminal_sigla} no ano de {ano}{mes_str}."

    total_teus = df_filtrado["TOTAL_TEU"].sum()

    coluna_unidades = "TOTAL_UNID"
    if coluna_unidades in df_filtrado.columns:
        total_conteineres = df_filtrado[coluna_unidades].sum()
        teu_ratio = (total_teus / total_conteineres) if total_conteineres > 0 else 0
    else:
        total_conteineres = 0
        teu_ratio = 0
        return f"Erro: Coluna de unidades não encontrada. Total TEU: {total_teus:,.0f}"

    nome_terminal = "Porto de Santos" if terminal_sigla.upper() == "PORTO" else TERMINAL_MAPPING.get(terminal_sigla.upper(), terminal_sigla)
    mes_str = f" (Meses: {meses})" if meses else ""

    return (
        f"Resultados de movimentação para {nome_terminal} em {ano}{mes_str}:\n"
        f"- Total TEUs: {total_teus:,.0f}\n"
        f"- Total Contêineres (Unidades): {total_conteineres:,.0f}\n"
        f"- TEU Ratio: {teu_ratio:.2f} TEUs/Contêiner\n"
    )


def get_market_share(df: pd.DataFrame, ano: int, terminal_sigla: str, meses: list = None) -> str:
    df_filtrado = df[df["ANO"] == ano].copy()

    if meses:
        df_filtrado = df_filtrado[df_filtrado["MES"].isin(meses)]
        periodo = f"nos meses {meses} de {ano}"
    else:
        periodo = f"no ano {ano}"

    total_porto = df_filtrado["TOTAL_TEU"].sum()

    if total_porto == 0:
        return f"Não há dados de movimentação no porto para o período solicitado ({periodo})."

    if terminal_sigla.upper() == "PORTO":
        return (
            f"O volume total do Porto de Santos {periodo} foi de "
            f"{total_porto:,.0f} TEUs."
        )

    mask_terminal = _encontrar_terminal(df_filtrado, terminal_sigla)
    total_terminal = df_filtrado[mask_terminal]["TOTAL_TEU"].sum()
    market_share = (total_terminal / total_porto) * 100 if total_porto > 0 else 0

    nome_terminal = TERMINAL_MAPPING.get(terminal_sigla.upper(), terminal_sigla)

    return (
        f"O market share da {nome_terminal} {periodo} foi de {market_share:.2f}% "
        f"(Volume do terminal: {total_terminal:,.0f} TEUs. "
        f"Volume do porto: {total_porto:,.0f} TEUs)."
    )


def get_cargo_mix(df: pd.DataFrame, terminal_sigla: str, ano: int, meses: list = None) -> str:
    mask = df["ANO"] == ano

    if meses and len(meses) > 0:
        mask &= df["MES"].isin(meses)

    if terminal_sigla.upper() != "PORTO":
        mask &= _encontrar_terminal(df, terminal_sigla)

    df_filtrado = df[mask].copy()

    if df_filtrado.empty:
        mes_str = f" nos meses {meses}" if meses else ""
        return f"Sem dados para {terminal_sigla} no ano de {ano}{mes_str}."

    df_filtrado["ESTADO"] = "Cheio"
    if "MERCADORIAS" in df_filtrado.columns:
        df_filtrado.loc[df_filtrado["MERCADORIAS"] == "SEM CARGAS", "ESTADO"] = "Vazio"

    df_filtrado["CATEGORIA"] = "Outros"

    if "SENTIDO" in df_filtrado.columns:
        df_filtrado.loc[df_filtrado["SENTIDO"] == "EMBARQUE", "CATEGORIA"] = "Exportação"
        df_filtrado.loc[df_filtrado["SENTIDO"] == "DESEMBARQUE", "CATEGORIA"] = "Importação"

    if "TIPO_NAVEGACAO" in df_filtrado.columns:
        df_filtrado.loc[df_filtrado["TIPO_NAVEGACAO"] == "CABOTAGEM", "CATEGORIA"] = "Cabotagem"

    if "MOVIMENTO" in df_filtrado.columns:
        df_filtrado.loc[df_filtrado["MOVIMENTO"] == "TRANSBORDO", "CATEGORIA"] = "Transbordo"
        df_filtrado.loc[df_filtrado["MOVIMENTO"] == "REMOÇÃO", "CATEGORIA"] = "Remoção"

    df_filtrado["MIX_FINAL"] = df_filtrado["CATEGORIA"] + " " + df_filtrado["ESTADO"]

    resumo_mix = (
        df_filtrado.groupby("MIX_FINAL")["TOTAL_TEU"]
        .sum()
        .sort_values(ascending=False)
    )

    total_teus = resumo_mix.sum()
    mes_str = f" (Meses: {meses})" if meses else ""
    nome_terminal = "Porto de Santos" if terminal_sigla.upper() == "PORTO" else TERMINAL_MAPPING.get(terminal_sigla.upper(), terminal_sigla)

    resultado = f"Mix de Cargas de {nome_terminal} em {ano}{mes_str} - Total: {total_teus:,.0f} TEUs:\n"

    for categoria, volume in resumo_mix.items():
        percentual = (volume / total_teus) * 100 if total_teus > 0 else 0
        resultado += f"- {categoria}: {volume:,.0f} TEUs ({percentual:.2f}%)\n"

    return resultado


def get_growth(
    df: pd.DataFrame,
    terminal_sigla: str,
    ano_base: int,
    ano_comparacao: int,
    meses_base: list = None,
    meses_comparacao: list = None,
    cargo_mix: str = None
) -> str:
    df_filtrado = df.copy()

    if terminal_sigla.upper() != "PORTO":
        mask_terminal = _encontrar_terminal(df_filtrado, terminal_sigla)
        df_filtrado = df_filtrado[mask_terminal]

    if cargo_mix:
        df_filtrado["ESTADO"] = "Cheio"
        if "MERCADORIAS" in df_filtrado.columns:
            df_filtrado.loc[df_filtrado["MERCADORIAS"] == "SEM CARGAS", "ESTADO"] = "Vazio"

        df_filtrado["CATEGORIA"] = "Outros"

        if "SENTIDO" in df_filtrado.columns:
            df_filtrado.loc[df_filtrado["SENTIDO"] == "EMBARQUE", "CATEGORIA"] = "Exportação"
            df_filtrado.loc[df_filtrado["SENTIDO"] == "DESEMBARQUE", "CATEGORIA"] = "Importação"

        if "TIPO_NAVEGACAO" in df_filtrado.columns:
            df_filtrado.loc[df_filtrado["TIPO_NAVEGACAO"] == "CABOTAGEM", "CATEGORIA"] = "Cabotagem"

        if "MOVIMENTO" in df_filtrado.columns:
            df_filtrado.loc[df_FILTRADO["MOVIMENTO"] == "TRANSBORDO", "CATEGORIA"] = "Transbordo"
            df_filtrado.loc[df_filtrado["MOVIMENTO"] == "REMOÇÃO", "CATEGORIA"] = "Remoção"

        df_filtrado["MIX_FINAL"] = df_filtrado["CATEGORIA"] + " " + df_filtrado["ESTADO"]
        df_filtrado = df_filtrado[df_filtrado["MIX_FINAL"].str.lower() == cargo_mix.lower()]

    mask_base = df_filtrado["ANO"] == ano_base
    if meses_base and len(meses_base) > 0:
        mask_base &= df_filtrado["MES"].isin(meses_base)
    vol_base = df_filtrado[mask_base]["TOTAL_TEU"].sum()

    mask_comp = df_filtrado["ANO"] == ano_comparacao
    if meses_comparacao and len(meses_comparacao) > 0:
        mask_comp &= df_filtrado["MES"].isin(meses_comparacao)
    vol_comp = df_filtrado[mask_comp]["TOTAL_TEU"].sum()

    if vol_base == 0:
        if vol_comp == 0:
            return f"Não houve movimentação em ambos os períodos para {terminal_sigla}."
        return f"Crescimento de 100% (volume base era zero, e agora é {vol_comp:,.0f} TEUs)."

    crescimento = ((vol_comp - vol_base) / vol_base) * 100

    mix_texto = f" na categoria '{cargo_mix}'" if cargo_mix else ""
    meses_base_str = f" nos meses {meses_base}" if meses_base else ""
    meses_comp_str = f" nos meses {meses_comparacao}" if meses_comparacao else ""
    nome_terminal = "Porto de Santos" if terminal_sigla.upper() == "PORTO" else TERMINAL_MAPPING.get(terminal_sigla.upper(), terminal_sigla)

    resultado = (
        f"Análise de Crescimento para {nome_terminal}{mix_texto}:\n"
        f"- Período Base ({ano_base}{meses_base_str}): {vol_base:,.0f} TEUs\n"
        f"- Período Atual ({ano_comparacao}{meses_comp_str}): {vol_comp:,.0f} TEUs\n"
        f"- Variação: {crescimento:.2f}%\n"
    )

    return resultado


def plotar_grafico(
    df: pd.DataFrame,
    tipo_grafico: str,
    tema: str = "market_share",
    anos: list = None,
    terminal_sigla: str = "PORTO"
) -> str:
    if anos is None or len(anos) == 0:
        anos = [int(df["ANO"].max())]

    anos = sorted(anos)
    df_filtrado = df[df["ANO"].isin(anos)].copy()

    if df_filtrado.empty:
        return "Não há dados para gerar o gráfico no período solicitado."

    plt.figure(figsize=(10, 5))
    titulo_anos = f"{anos[0]} a {anos[-1]}" if len(anos) > 1 else str(anos[0])
    plt.title(f"{tema.replace('_', ' ').title()} - {terminal_sigla} ({titulo_anos})")

    if tema == "market_share":
        resumo = df_filtrado.groupby("TERMINAIS")["TOTAL_TEU"].sum().sort_values(ascending=False)
        top3 = resumo.head(3)
        outros = pd.Series({"OUTROS": resumo.iloc[3:].sum()}) if len(resumo) > 3 else pd.Series(dtype=float)
        dados_finais = pd.concat([top3, outros])

        if tipo_grafico in ["pizza", "rosca"]:
            plt.pie(dados_finais, labels=dados_finais.index, autopct="%1.1f%%", startangle=90)
            if tipo_grafico == "rosca":
                centro = plt.Circle((0, 0), 0.70, fc="white")
                plt.gca().add_artist(centro)
        else:
            plt.bar(dados_finais.index, dados_finais.values)
            plt.ylabel("TEUs")

    elif tema == "evolucao_mensal":
        if terminal_sigla.upper() != "PORTO":
            mask = _encontrar_terminal(df_filtrado, terminal_sigla)
            df_filtrado = df_filtrado[mask]

        if df_filtrado.empty:
            plt.close()
            return "Não há dados para gerar o gráfico do terminal solicitado."

        df_filtrado["ANO_MES"] = (
            df_filtrado["ANO"].astype(str)
            + "-"
            + df_filtrado["MES"].astype(str).str.zfill(2)
        )

        resumo_mensal = (
            df_filtrado.groupby("ANO_MES")["TOTAL_TEU"]
            .sum()
            .sort_index()
        )

        if tipo_grafico == "linha":
            plt.plot(resumo_mensal.index, resumo_mensal.values, marker="o", linestyle="-")
            plt.grid(True, linestyle="--", alpha=0.6)
        else:
            plt.bar(resumo_mensal.index, resumo_mensal.values)

        plt.xticks(rotation=60 if len(resumo_mensal) > 12 else 0)
        plt.xlabel("Mês/Ano")
        plt.ylabel("TEUs")

    else:
        plt.close()
        return f"Tema de gráfico '{tema}' não suportado."

    plt.tight_layout()

    filename = f"grafico_{uuid.uuid4().hex}.png"
    caminho_arquivo = os.path.join(TEMP_GRAPH_DIR, filename)

    plt.savefig(caminho_arquivo, format="png", bbox_inches="tight")
    plt.close()

    return f"Gráfico gerado com sucesso em: {caminho_arquivo}"