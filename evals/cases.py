"""
Perguntas de referência da avaliação.

Cada caso traz a conversa (a última mensagem é a avaliada) e o que se espera:
números calculados pelo oráculo, argumentos da chamada da ferramenta de dados
e menções obrigatórias. `onde="resposta"` aceita números na tabela exibida ao
usuário, além do texto, para perguntas cuja resposta é uma lista.
"""
from dataclasses import dataclass, field
from typing import Callable, List, Optional

from evals.scoring import (
    agrupa,
    cargo_mix,
    compara,
    mercadoria,
    metrica,
    navegacao,
    sem_filtro_de_terminal,
    sentido,
    terminal,
    toneladas,
)

NAO = ["não"]
MESES = ["", "janeiro", "fevereiro", "marco", "abril", "maio", "junho", "julho", "agosto",
         "setembro", "outubro", "novembro", "dezembro"]
EXPORTACAO = {"sentido": "EMBARQUE", "navegacao": "LONGO CURSO"}
IMPORTACAO = {"sentido": "DESEMBARQUE", "navegacao": "LONGO CURSO"}


@dataclass
class Case:
    id: str
    categoria: str
    turnos: List[str]
    esperado: Optional[Callable] = None
    args: List[tuple] = field(default_factory=list)
    mencoes: List[str] = field(default_factory=list)
    alguma_mencao: object = field(default_factory=list)
    sem_numeros_grandes: bool = False
    onde: str = "texto"


def _top(d: dict, n: int) -> list:
    return list(d.values())[:n]


CASES = [
    # --- Contêineres (TEU / unidades) ---------------------------------
    Case("teu_2025", "conteiner", ["Quantos TEUs o porto movimentou em 2025?"],
         lambda o: {"valores": [o.total(ano=2025)]}, [metrica("teus", "market_share")]),
    Case("teu_jun_2026", "conteiner", ["Qual foi o volume de TEUs em junho de 2026?"],
         lambda o: {"valores": [o.total(de=202606, ate=202606)]}),
    Case("sbsa_2025", "conteiner", ["Qual o volume da Santos Brasil em 2025?"],
         lambda o: {"valores": [o.total(ano=2025, terminais=["SBSA"])]}, [terminal("SBSA")]),
    Case("libra_2010_2019", "conteiner", ["Quanto a Libra movimentou em TEUs entre 2010 e 2019?"],
         lambda o: {"valores": [o.total(de=201001, ate=201912, terminais=["LIBRA 35", "LIBRA 37"])]},
         [terminal("LIBRA 35"), terminal("LIBRA 37")]),
    Case("btp_unidades_2024", "conteiner", ["Quantos contêineres (em unidades) a BTP movimentou em 2024?"],
         lambda o: {"valores": [o.total("unidades", ano=2024, terminais=["BTP"])]}, [metrica("conteineres")]),

    # --- Market share --------------------------------------------------
    Case("share_terminal_2015", "share", ["Qual o market share por terminal em 2015?"],
         lambda o: {"percentuais": [o.shares("terminal", ano=2015)[t] for t in ("SBSA", "LIBRA 35")]},
         [metrica("market_share"), agrupa("terminal")], onde="resposta"),
    Case("share_terminal_2025", "share", ["Qual o market share por terminal em 2025?"],
         lambda o: {"percentuais": _top(o.shares("terminal", ano=2025), 3)},
         [metrica("market_share"), agrupa("terminal")]),
    Case("share_cargo_mix_2025", "share", ["Qual a participação de cada cargo mix em 2025?"],
         lambda o: {"percentuais": _top(o.shares("cargo_mix", ano=2025), 2)},
         [metrica("market_share"), agrupa("cargo_mix")], onde="resposta"),
    Case("share_dpw_1s2026", "share", ["Qual foi o market share da DP World no primeiro semestre de 2026?"],
         lambda o: {"percentuais": [o.shares("terminal", de=202601, ate=202606)["DPW"]]},
         [metrica("market_share"), agrupa("terminal"), sem_filtro_de_terminal()]),
    Case("mix_btp_2025", "share", ["Qual o mix de carga da BTP em 2025?"],
         lambda o: {"percentuais": _top(o.shares("cargo_mix", ano=2025, terminais=["BTP"]), 2)},
         [terminal("BTP"), agrupa("cargo_mix")], onde="resposta"),

    # --- Crescimento (YoY) ---------------------------------------------
    Case("yoy_teu_2025", "yoy", ["Qual foi o crescimento de TEUs de 2024 para 2025?"],
         lambda o: {"percentuais": [o.yoy_pct(2024, 2025)]}, [compara()]),
    Case("yoy_terminal_2025", "yoy", ["Qual foi o crescimento YoY de cada terminal em 2025?"],
         lambda o: {"percentuais": [o.yoy_pct(2024, 2025, terminais=[t]) for t in ("SBSA", "BTP", "DPW")]},
         [compara(), agrupa("terminal")], onde="resposta"),
    # A resposta precisa deixar claro que 2026 está incompleto (comparação do
    # mesmo período), qualquer que seja a formulação.
    Case("ytd_2026", "yoy", ["Como está a movimentação de TEUs de 2026 em relação a 2025?"],
         lambda o: {"percentuais": [o.yoy_pct(2025, 2026, ate_mes=o.max_month)]}, [compara()],
         alguma_mencao=lambda o: [[
             "acumulad", "ytd", "parcia", "incomplet", "mesmo periodo", "ano completo", "ano cheio",
             "primeiros", f"ate {MESES[o.max_month]}", f"janeiro a {MESES[o.max_month]}",
         ]]),
    # Julho, e não agosto: em agosto a variação do mês coincide com a do acumulado
    # (3,88%), e o caso não distinguiria uma resposta errada de uma certa.
    Case("yoy_julho", "yoy", ["Compare julho de 2026 com julho de 2025 em TEUs."],
         lambda o: {"percentuais": [o.variacao_pct(o.total(de=202507, ate=202507), o.total(de=202607, ate=202607))]}),
    Case("yoy_acucar", "yoy", ["Qual foi o crescimento das exportações de açúcar de 2024 para 2025?"],
         lambda o: {"percentuais": [o.yoy_pct(2024, 2025, "toneladas", mercadorias=["ACUCAR"], **EXPORTACAO)]},
         [compara(), mercadoria("ACUCAR")]),
    Case("continuacao_acucar", "yoy",
         ["Quantas toneladas de açúcar foram exportadas em 2025?", "Compare com o ano anterior"],
         lambda o: {"percentuais": [o.yoy_pct(2024, 2025, "toneladas", mercadorias=["ACUCAR"], **EXPORTACAO)]},
         [compara(), mercadoria("ACUCAR")]),

    # --- CAGR ----------------------------------------------------------
    Case("cagr_2015_2025", "cagr", ["Qual o CAGR de TEUs do porto entre 2015 e 2025?"],
         lambda o: {"percentuais": [o.cagr_pct(2015, 2025)]}, [metrica("cagr")]),
    Case("cagr_btp_2014_2024", "cagr", ["Qual o CAGR da BTP entre 2014 e 2024?"],
         lambda o: {"percentuais": [o.cagr_pct(2014, 2024, terminais=["BTP"])]}, [metrica("cagr"), terminal("BTP")]),
    # Pergunta mal formulada: a resposta precisa avisar que 2026 está incompleto
    # (pode, além disso, oferecer o CAGR de outro período).
    Case("cagr_2026", "cagr", ["Qual o CAGR de TEUs em 2026?"],
         alguma_mencao=[["anos completos", "incomplet", "parcia", "nao e possivel", "nao ha", "insuficient", "pelo menos dois"]]),

    # --- Toneladas, granéis e mercadorias ------------------------------
    Case("toneladas_2025", "toneladas", ["Qual foi a movimentação total do porto em toneladas em 2025?"],
         lambda o: {"valores": [o.total("toneladas", ano=2025)]}, [toneladas()]),
    Case("toneladas_natureza_2025", "toneladas", ["Quantas toneladas de cada natureza de carga o porto movimentou em 2025?"],
         lambda o: {"valores": list(o.por("natureza_carga", "toneladas", ano=2025).values())},
         [agrupa("natureza_carga")], onde="resposta"),
    Case("soja_exportacao_2025", "toneladas", ["Quantas toneladas de soja em grãos foram exportadas em 2025?"],
         lambda o: {"valores": [o.total("toneladas", ano=2025, mercadorias=["SOJA EM GRAOS"], **EXPORTACAO)]},
         [mercadoria("SOJA"), navegacao("LONGO CURSO")]),
    Case("milho_exportacao_2024", "toneladas", ["Quanto de milho o porto exportou em 2024?"],
         lambda o: {"valores": [o.total("toneladas", ano=2024, mercadorias=["MILHO"], **EXPORTACAO)]},
         [mercadoria("MILHO"), sentido("EMBARQUE")]),
    Case("top3_mercadorias_2025", "toneladas", ["Quais foram as 3 mercadorias mais movimentadas em toneladas em 2025?"],
         args=[agrupa("mercadoria")],
         alguma_mencao=lambda o: [[m] for m in list(o.por("mercadoria", "toneladas", ano=2025))[:3]]),
    Case("granel_liquido_desembarque_2025", "toneladas", ["Quantas toneladas de granel líquido foram desembarcadas em 2025?"],
         lambda o: {"valores": [o.total("toneladas", ano=2025, naturezas=["GRANEL LIQUIDO"], sentido="DESEMBARQUE")]},
         [sentido("DESEMBARQUE")]),
    Case("share_acucar_terminal_2025", "toneladas", ["Qual a participação de cada terminal na exportação de açúcar em 2025?"],
         lambda o: {"percentuais": _top(o.shares("terminal", "toneladas", ano=2025, mercadorias=["ACUCAR"], **EXPORTACAO), 1)},
         [metrica("market_share"), toneladas(), mercadoria("ACUCAR")]),
    Case("fertilizantes_importacao_2025", "toneladas", ["Quantas toneladas de fertilizantes foram importadas em 2025?"],
         lambda o: {"valores": [o.total("toneladas", ano=2025, mercadorias=["FERTILIZANTES"], **IMPORTACAO)]},
         [mercadoria("FERTILIZANTES"), sentido("DESEMBARQUE")]),

    # --- Cargo mix -----------------------------------------------------
    Case("remocao_2025", "cargo_mix", ["Quantos TEUs de remoção o porto teve em 2025?"],
         lambda o: {"valores": [o.total(ano=2025, cargo_mix="REMOCAO")]}, [cargo_mix("REMOCAO")]),
    Case("transbordo_sbsa_2025", "cargo_mix", ["Quantos TEUs de transbordo a Santos Brasil fez em 2025?"],
         lambda o: {"valores": [o.total(ano=2025, cargo_mix="TRANSBORDO", terminais=["SBSA"])]},
         [cargo_mix("TRANSBORDO"), terminal("SBSA")]),
    Case("exportacao_cheio_2025", "cargo_mix", ["Quantos TEUs de exportação cheios foram movimentados em 2025?"],
         lambda o: {"valores": [o.total(ano=2025, cargo_mix="EXPORTACAO CHEIO")]}, [cargo_mix("EXPORTACAO")]),

    # --- Recortes de tempo ---------------------------------------------
    Case("trimestres_2025", "tempo", ["Quantos TEUs foram movimentados em cada trimestre de 2025?"],
         lambda o: {"valores": [o.total(de=2025 * 100 + m, ate=2025 * 100 + m + 2) for m in (1, 4, 7, 10)]},
         [agrupa("trimestre")], onde="resposta"),
    Case("mes_recorde_2025", "tempo", ["Qual foi o mês com maior movimentação de TEUs em 2025?"],
         lambda o: {"valores": [max(o.total(de=2025 * 100 + m, ate=2025 * 100 + m) for m in range(1, 13))]},
         [agrupa("mes")]),
    Case("evolucao_2020_2025", "tempo", ["Mostre a evolução anual de TEUs de 2020 a 2025."],
         lambda o: {"valores": [o.total(ano=a) for a in range(2020, 2026)]}, [agrupa("ano")], onde="resposta"),

    # --- Conhecimento (documentos) -------------------------------------
    Case("rag_1892", "documentos", ["Quando o porto organizado de Santos entrou em operação?"], mencoes=["1892"]),
    Case("rag_landlord", "documentos", ["O que é o modelo landlord port adotado em Santos?"],
         alguma_mencao=[["autoridade portuaria"], ["arrend", "privad", "operadores"]]),
    Case("rag_expansao", "documentos", ["Quais são os principais projetos de expansão e infraestrutura do Porto de Santos?"],
         alguma_mencao=[["tecon", "tunel", "dragagem", "barnabe"]]),

    # --- Robustez: não inventar ----------------------------------------
    Case("ano_2030", "robustez", ["Quantos TEUs o porto movimentou em 2030?"],
         alguma_mencao=[NAO], sem_numeros_grandes=True),
    Case("terminal_inexistente", "robustez", ["Qual o volume do terminal XYZ em 2025?"],
         alguma_mencao=[NAO], sem_numeros_grandes=True),
    Case("mercadoria_inexistente", "robustez", ["Quantas toneladas de ouro foram exportadas em 2025?"],
         alguma_mencao=[NAO], sem_numeros_grandes=True),
    Case("continuacao_share", "robustez", ["Qual o market share por terminal em 2025?", "E em 2024?"],
         lambda o: {"percentuais": _top(o.shares("terminal", ano=2024), 2)}, [metrica("market_share")]),
]
