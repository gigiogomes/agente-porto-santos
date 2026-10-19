"""
Visualização automática dos resultados das consultas.

A forma vem da estrutura do resultado (quais dimensões e medidas ele tem),
nunca do LLM: um número vira cartão; série no tempo vira linha; ranking e
share viram barras horizontais; comparação entre anos vira barras em dois tons.

Cores: paleta de referência validada para daltonismo, sem alterações. Uma cor
para magnitudes; cores categóricas em ordem fixa só quando as séries são o
assunto (no máximo 4, o resto vira "Demais", em cinza).

Tema claro/escuro: fundo, textos, eixos e grade NÃO são definidos aqui. O
Streamlit aplica o tema ativo ao gráfico no navegador, inclusive quando o
usuário troca de tema sem a página rodar de novo. Por isso as cores das séries
são um conjunto único, legível nos dois fundos.
"""
from typing import Optional, Tuple

import pandas as pd
import plotly.graph_objects as go

from tools.port_data_tools import QueryResult, formatar_numero, formatar_percentual

TEMPORAIS = ["ANO", "ANO_SEMESTRE", "ANO_QUADRIMESTRE", "ANO_TRIMESTRE", "ANO_BIMESTRE", "ANO_MES"]
MAX_BARRAS = 10
MAX_SERIES = 4
FRACAO_MINIMA = 0.01  # itens abaixo de 1% do total entram em "Demais"
DEMAIS = "Demais"

CORES = {
    "series": ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"],
    "negativo": "#e34948",
    "contexto": "#898781",  # cinza neutro, igual nos dois temas
}
# O ano anterior usa o mesmo azul com transparência: mistura-se ao fundo e
# recua tanto no tema claro quanto no escuro (um azul claro fixo se
# destacaria no fundo escuro, o contrário do pretendido).
OPACIDADE_ANTERIOR = 0.45


def _layout(
    fig: go.Figure, horizontal: bool, n_categorias: int = 0, legenda: bool = False, rotulos_diretos: bool = False
) -> go.Figure:
    # Com o valor escrito na ponta de cada barra, o eixo de valor é redundante.
    eixo_valor = dict(
        showgrid=not rotulos_diretos, zeroline=True, showticklabels=not rotulos_diretos, tickformat=",.0f", automargin=True,
    )
    eixo_categoria = dict(showgrid=False, automargin=True)
    fig.update_layout(
        separators=",.",  # vírgula decimal, ponto de milhar
        margin=dict(l=8, r=8, t=8, b=8),
        height=max(260, 34 * n_categorias + 60) if horizontal else 320,
        barcornerradius=4,
        bargap=0.25,
        showlegend=legenda,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        xaxis=eixo_valor if horizontal else eixo_categoria,
        yaxis={**(eixo_categoria if horizontal else eixo_valor), **({"autorange": "reversed"} if horizontal else {})},
        hovermode="closest" if horizontal else "x unified",
    )
    return fig


def _top_com_demais(df: pd.DataFrame, dim: str, valor: str, n: int = MAX_BARRAS) -> pd.DataFrame:
    """
    Os maiores itens (até n, cada um com pelo menos 1% do total) e uma linha
    "Demais" com a soma do resto, quando o resto tem mais de um item.
    """
    df = df.sort_values(valor, ascending=False).reset_index(drop=True)
    relevantes = (df[valor] >= df[valor].sum() * FRACAO_MINIMA) & (df.index < n - 1)
    resto = df[~relevantes]
    if len(resto) <= 1:
        return df.head(n)
    df = df[relevantes]
    linha = {dim: DEMAIS, valor: resto[valor].sum()}
    if "Share (%)" in df.columns:
        linha["Share (%)"] = round(resto["Share (%)"].sum(), 2)
    return pd.concat([df, pd.DataFrame([linha])], ignore_index=True)


def _barras_horizontais(df, dim, valor, rotulo, texto, hover) -> go.Figure:
    df = df.copy()
    df[dim] = df[dim].astype(str)
    fig = go.Figure(
        go.Bar(
            x=df[valor], y=df[dim], orientation="h",
            marker=dict(color=[CORES["contexto"] if d == DEMAIS else CORES["series"][0] for d in df[dim]]),
            text=texto, textposition="outside", cliponaxis=False,
            hovertemplate=hover, name=rotulo,
        )
    )
    # Folga à direita para o rótulo da maior barra não ser cortado.
    fig.update_xaxes(range=[min(0, df[valor].min() * 1.25), df[valor].max() * 1.25])
    return _layout(fig, horizontal=True, n_categorias=len(df), rotulos_diretos=True)


def _linhas(df, dim_tempo, valor, rotulo, dim_serie=None) -> go.Figure:
    df = df.copy()
    df[dim_tempo] = df[dim_tempo].astype(str)
    fig = go.Figure()
    if dim_serie is None:
        series = [(rotulo, df, CORES["series"][0])]
    else:
        totais = df.groupby(dim_serie)[valor].sum().sort_values(ascending=False)
        principais = list(totais.index[:MAX_SERIES])
        if len(totais) > MAX_SERIES:
            df[dim_serie] = df[dim_serie].where(df[dim_serie].isin(principais), DEMAIS)
            df = df.groupby([dim_tempo, dim_serie], as_index=False)[valor].sum()
            principais.append(DEMAIS)
        cor = dict(zip(principais, CORES["series"]))
        cor[DEMAIS] = CORES["contexto"]
        series = [(str(s), df[df[dim_serie] == s], cor[s]) for s in principais]

    marcadores = df[dim_tempo].nunique() <= 24
    for nome, dados, cor in series:
        dados = dados.sort_values(dim_tempo)
        fig.add_trace(
            go.Scatter(
                x=dados[dim_tempo], y=dados[valor], name=nome,
                mode="lines+markers" if marcadores else "lines",
                line=dict(color=cor, width=2), marker=dict(size=8, color=cor),
                hovertemplate=f"{nome}: %{{y:,.0f}}<extra></extra>",
            )
        )
    fig.update_xaxes(type="category")
    return _layout(fig, horizontal=False, legenda=dim_serie is not None)


def visualizacao(resultado: QueryResult) -> Tuple[str, Optional[object]]:
    """
    ("cartao", kwargs de st.metric), ("figura", go.Figure) ou ("nenhuma", None).
    A tabela é sempre exibida também: o gráfico complementa, não substitui.
    """
    df = resultado.tabela
    if df is None or df.empty:
        return "nenhuma", None
    rotulo = resultado.medida
    valor = resultado.colunas_valor[0] if resultado.colunas_valor else None
    dims = resultado.colunas_dim
    tempo = [d for d in dims if d in TEMPORAIS]
    categorias = [d for d in dims if d not in TEMPORAIS]

    if resultado.tipo == "total":
        return "cartao", {"label": f"Total ({rotulo})", "value": formatar_numero(df["Total"].iloc[0])}

    if resultado.tipo == "yoy":
        anterior, atual = resultado.colunas_valor[0], resultado.colunas_valor[1]
        if not dims:
            pct = df["YoY (%)"].iloc[0]
            return "cartao", {
                "label": atual,
                "value": formatar_numero(df[atual].iloc[0]),
                "delta": None if pd.isna(pct) else f"{formatar_percentual(pct)} vs {anterior.split()[-1]}",
            }
        if len(dims) != 1:
            return "nenhuma", None
        dim = dims[0]
        maior = df[[anterior, atual]].max(axis=1)
        top = df[maior >= maior.max() * FRACAO_MINIMA].sort_values(atual, ascending=False).head(MAX_BARRAS).copy()
        top[dim] = top[dim].astype(str)
        fig = go.Figure()
        for coluna, opacidade in [(anterior, OPACIDADE_ANTERIOR), (atual, 1.0)]:
            fig.add_trace(
                go.Bar(
                    x=top[coluna], y=top[dim], orientation="h", name=coluna.split()[-1],
                    marker=dict(color=CORES["series"][0], opacity=opacidade),
                    hovertemplate=f"{coluna.split()[-1]}: %{{x:,.0f}} {rotulo}<extra></extra>",
                )
            )
        fig.update_layout(barmode="group")
        return "figura", _layout(fig, horizontal=True, n_categorias=2 * len(top), legenda=True)

    if resultado.tipo == "cagr":
        if "Terminal" not in df.columns:
            pct = df["CAGR (%)"].iloc[0]
            if pd.isna(pct):
                return "nenhuma", None
            anos = f"{df['Ano Inicial'].iloc[0]}–{df['Ano Final'].iloc[0]}"
            return "cartao", {"label": f"CAGR {anos} ({rotulo})", "value": f"{formatar_percentual(pct)} a.a."}
        dados = df.dropna(subset=["CAGR (%)"]).sort_values("CAGR (%)", ascending=False).head(MAX_BARRAS)
        fig = go.Figure(
            go.Bar(
                x=dados["CAGR (%)"], y=dados["Terminal"].astype(str), orientation="h",
                marker=dict(color=[CORES["series"][0] if v >= 0 else CORES["negativo"] for v in dados["CAGR (%)"]]),
                text=[formatar_percentual(v) for v in dados["CAGR (%)"]], textposition="outside", cliponaxis=False,
                    hovertemplate="%{y}: %{x:.2f}% a.a.<extra></extra>",
            )
        )
        extremo = dados["CAGR (%)"].abs().max() * 1.3
        fig.update_xaxes(range=[min(0, dados["CAGR (%)"].min() * 1.3), max(0, dados["CAGR (%)"].max()) + extremo * 0.1])
        return "figura", _layout(fig, horizontal=True, n_categorias=len(dados), rotulos_diretos=True)

    # tipo "tabela" ou "share"
    if tempo and len(dims) == 1:
        return "figura", _linhas(df, tempo[0], valor, rotulo)
    if len(tempo) == 1 and len(categorias) == 1:
        return "figura", _linhas(df, tempo[0], valor, rotulo, dim_serie=categorias[0])
    if len(categorias) == 1 and not tempo:
        dim = categorias[0]
        top = _top_com_demais(df, dim, valor)
        if resultado.tipo == "share":
            texto = [formatar_percentual(v) for v in top["Share (%)"]]
            hover = "%{y}: %{customdata:.2f}% (%{x:,.0f} " + rotulo + ")<extra></extra>"
            fig = _barras_horizontais(top, dim, valor, rotulo, texto, hover)
            fig.update_traces(customdata=top["Share (%)"])
        else:
            texto = [formatar_numero(v) for v in top[valor]]
            hover = "%{y}: %{x:,.0f} " + rotulo + "<extra></extra>"
            fig = _barras_horizontais(top, dim, valor, rotulo, texto, hover)
        return "figura", fig
    return "nenhuma", None
