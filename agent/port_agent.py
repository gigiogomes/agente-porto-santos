"""
Agente do Porto de Santos: um único agente com ferramentas.

Recebe a conversa inteira (sem paráfrase de um coordenador) e roda um loop:
o modelo responde ou pede ferramentas; os resultados voltam para ele; repete
até haver resposta final ou atingir o limite de passos.
"""
import json
import logging
import os
from dataclasses import dataclass, field
from typing import Iterator, List, Optional, Tuple

from dotenv import load_dotenv

from agent.knowledge_agent import KnowledgeAgent
from tools.port_data_tools import PortDataStore, QueryResult, load_port_data, run_query

load_dotenv()

logger = logging.getLogger(__name__)

# Mensagem genérica exibida ao usuário. O detalhe técnico vai apenas para o log.
USER_FACING_ERROR = "Ocorreu um erro ao processar a sua solicitação. Tente novamente em instantes."

MAX_PASSOS = 5  # chamadas ao modelo por pergunta (cada uma pode pedir várias ferramentas)
MAX_HISTORICO = 12  # mensagens anteriores enviadas ao modelo (6 trocas)

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "consultar_dados",
            "description": (
                "Consulta a movimentação do Porto de Santos (base mensal da APS): volumes em TEU, contêineres "
                "ou toneladas, market share, crescimento (YoY), CAGR, cargo mix, granéis e mercadorias. "
                "Pode ser chamada várias vezes."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "metric": {
                        "type": "string",
                        "enum": ["teus", "conteineres", "toneladas", "market_share", "cagr"],
                        "description": "Qual métrica extrair.",
                    },
                    "medida": {
                        "type": "string",
                        "enum": ["teu", "unidades", "toneladas"],
                        "description": "Base de cálculo para market_share, cagr e comparações. Padrão: teu (ou toneladas quando metric='toneladas').",
                    },
                    "start_date": {"type": "string", "description": "Mês inicial, YYYY-MM. Ex: 2021-01"},
                    "end_date": {"type": "string", "description": "Mês final (inclusivo), YYYY-MM. Ex: 2023-12"},
                    "terminals": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Terminais (siglas oficiais ou nomes da base). Vazio = porto inteiro.",
                    },
                    "natureza_carga": {
                        "type": "array",
                        "items": {"type": "string", "enum": ["CARGA CONTEINERIZADA", "GRANEL SOLIDO", "GRANEL LIQUIDO", "CARGA GERAL"]},
                        "description": "Padrão: só contêineres quando a medida é teu/unidades; todas quando é toneladas.",
                    },
                    "mercadorias": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Busca por trecho do nome: 'FERTILIZANTES' traz todos os tipos de fertilizante; 'SOJA EM GRAOS' só os grãos; 'SOJA' traz grãos e farelo.",
                    },
                    "sentido": {"type": "array", "items": {"type": "string", "enum": ["EMBARQUE", "DESEMBARQUE"]}},
                    "navegacao": {
                        "type": "array",
                        "items": {"type": "string", "enum": ["LONGO CURSO", "CABOTAGEM"]},
                        "description": "LONGO CURSO = comércio exterior; CABOTAGEM = entre portos brasileiros.",
                    },
                    "cargo_mix_filter": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Cargo mix de contêineres: fluxo (EXPORTACAO, IMPORTACAO, CABOTAGEM, TRANSBORDO, REMOCAO) ou valor exato (ex: 'EXPORTACAO CHEIO').",
                    },
                    "group_by": {
                        "type": "array",
                        "items": {
                            "type": "string",
                            "enum": [
                                "ano", "semestre", "quadrimestre", "trimestre", "bimestre", "mes",
                                "terminal", "cargo_mix", "natureza_carga", "mercadoria", "sentido", "navegacao",
                            ],
                        },
                        "description": "Dimensões para agrupar; pode cruzar (ex: ['mes', 'terminal']).",
                    },
                    "compare_with_previous": {
                        "type": "boolean",
                        "description": "True para crescimento/comparação com o ano anterior (YoY).",
                    },
                },
                "required": ["metric"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "buscar_documentos",
            "description": "Busca trechos de documentos sobre o Porto de Santos: história, infraestrutura, modelo de gestão, projetos e expansão.",
            "parameters": {
                "type": "object",
                "properties": {"pergunta": {"type": "string", "description": "O que buscar."}},
                "required": ["pergunta"],
            },
        },
    },
]

SYSTEM_PROMPT = """
Você é o assistente de dados do Porto de Santos. Responda em português do Brasil, em primeira pessoa, de forma direta, sem anunciar que vai consultar algo.

[FERRAMENTAS]
- consultar_dados: qualquer número sobre a movimentação do porto. Todo número da resposta deve vir desta ferramenta: nunca calcule, estime ou recalcule (nem somas, diferenças ou percentuais). Se precisar de outro número, faça outra consulta.
- buscar_documentos: perguntas qualitativas (história, infraestrutura, modelo de gestão, projetos). Responda só com o que os trechos trazem; se não houver, diga que não encontrou.
Se a pergunta for ambígua, escolha a interpretação mais provável e diga qual usou.

[COMO APRESENTAR]
1. As tabelas retornadas por consultar_dados são exibidas automaticamente ao usuário logo abaixo da sua resposta. Não copie a tabela inteira: destaque os números principais (até cerca de 5) e explique o que significam.
2. Use os números exatamente como vêm da ferramenta, já no formato brasileiro (1.234.567 e 43,91%). Sempre diga a unidade (TEU, unidades ou toneladas) e o período.
3. Notas entre colchetes no resultado ([YoY ACUMULADO...], [Escopo: ...], [AVISO DO SISTEMA] etc.) são obrigatórias: siga-as e informe ao usuário o que dizem (ex.: que a comparação é acumulada até um mês).
4. Sem dados para o pedido: diga isso claramente e não invente valores.

[MEDIDA PADRÃO]
- Sem unidade na pergunta, "volume", "movimentação" e "market share" de terminais, cargo mix, crescimento e CAGR são em TEU (contêineres): é a convenção do setor. Diga que é TEU.
- Use toneladas só quando a pergunta falar em peso/toneladas, granel, carga geral, mercadorias ou "movimentação total do porto".

[PERÍODOS E TOTAIS]
- Sem período na pergunta: use o último mês disponível e diga isso.
- Datas no formato YYYY-MM. "Último ano": o último ano completo, se o atual estiver incompleto; diga qual usou.
- Pergunta por um total ("quantos", "quanto", "total entre 2010 e 2019"): consulte sem group_by e responda o total. Quebras por ano, mês ou tipo são complemento opcional, em outra consulta.

[CRESCIMENTO E COMPARAÇÕES]
- Crescimento, variação, "comparar com", YoY: compare_with_previous=true, com uma janela que cubra os dois períodos. A ferramenta compara só os meses presentes nos dois anos e devolve diferença e percentual prontos.
  • Ano contra o anterior: start_date=ANO_ANTERIOR-01, end_date=ANO-12.
  • Um mês contra o mesmo mês do ano anterior: start_date=ANO_ANTERIOR-MM, end_date=ANO-MM.
  • Por terminal, cargo mix, mercadoria etc.: group_by com a dimensão (não inclua "ano" nem "mes").
- Em continuações ("compare com o ano anterior", "e em 2024?"), mantenha os filtros da consulta anterior, listados em [Consultas usadas nesta resposta] no histórico.
- CAGR (crescimento médio anual composto): metric="cagr". A ferramenta usa só anos completos e informa quais; repita exatamente esses anos e avise se o período pedido incluía ano incompleto.

[SHARE E MIX]
- Share, participação, proporção: metric="market_share" com group_by da dimensão. O share de UM terminal é em relação ao porto: NÃO filtre o terminal; agrupe por terminal e destaque o pedido.
- "Mix de carga" de um terminal: metric="market_share", terminals=[terminal], group_by=["cargo_mix"].
- Junto ao share, mencione o volume absoluto.

[TERMINAIS]
- Prefira a sigla oficial: Santos Brasil -> SBSA; DP World ou Embraport -> DPW; Brasil Terminal Portuário -> BTP; Ecoporto -> ECOPORTO.
- Terminal com mais de um valor na base (ex.: Libra -> LIBRA 35 e LIBRA 37): passe todos.

[CONTÊINERES, TONELADAS E MERCADORIAS]
- TEU e contêineres (unidades) existem só para carga conteinerizada.
- Peso, granéis, carga geral, mercadorias ou "movimentação total do porto": metric="toneladas" (para share, CAGR ou comparação em peso: medida="toneladas").
- Mercadorias: use os nomes da lista. "Soja" -> "SOJA EM GRAOS" (farelo de soja é outra mercadoria; mencione).
- Exportação/importação de mercadorias e granéis: sentido EMBARQUE/DESEMBARQUE e navegacao=["LONGO CURSO"]. Só "embarque"/"desembarque": apenas sentido.
- Contêineres: exportação, importação, cabotagem, transbordo, remoção, cheio/vazio -> cargo_mix_filter (ex.: ["REMOCAO"], ["EXPORTACAO CHEIO"]), não sentido.
"""


@dataclass
class Turno:
    """Resultado de uma pergunta: texto, tabelas para exibir e histórico atualizado."""

    texto: str
    resultados: List[QueryResult] = field(default_factory=list)
    consultas: List[dict] = field(default_factory=list)
    historico: List[dict] = field(default_factory=list)


class PortAgent:
    """
    Instância compartilhada entre sessões do Streamlit: guarda só o que é
    imutável (dados, índice de documentos, cliente). O histórico da conversa
    é recebido e devolvido a cada pergunta.
    """

    def __init__(self, request_timeout: float = 60.0, max_retries: int = 2):
        logger.info("Iniciando o agente...")
        self.model_name = os.getenv("PORT_AGENT_MODEL", "gpt-4o")
        self._aceita_temperatura = True
        fato, self.data_coverage, self.data_diagnostics = load_port_data()
        self.store = PortDataStore(fato)
        self.knowledge_agent = KnowledgeAgent()
        self.system_prompt = SYSTEM_PROMPT + self._contexto_dos_dados()

        self.client = None
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key:
            try:
                from openai import OpenAI

                # timeout e max_retries cobrem falhas transitórias (429/5xx).
                self.client = OpenAI(api_key=api_key, timeout=request_timeout, max_retries=max_retries)
            except Exception:
                logger.exception("Falha ao inicializar o cliente OpenAI.")

    def _contexto_dos_dados(self) -> str:
        """Parte do prompt que vem dos dados. Fica no fim: o começo, fixo, aproveita o cache de prompt."""
        cov = self.data_coverage
        return (
            "\n[DADOS DISPONÍVEIS]\n"
            f"Período: {cov.min_year} até {cov.max_month:02d}/{cov.max_year} (último mês disponível).\n"
            f"Terminais (maior para menor volume, inclui desativados): {', '.join(cov.terminal_list)}\n"
            f"Cargo mix: {', '.join(cov.cargo_mix_list)}\n"
            f"Naturezas: {', '.join(cov.natureza_list)}\n"
            f"Mercadorias: {', '.join(cov.mercadoria_list)}\n"
        )

    def get_status_summary(self) -> str:
        """Resumo da saúde dos dados para a interface do Streamlit."""
        cov = self.data_coverage
        if cov.source_name == "PUBLICADO":
            source = f"dados publicados pelo pipeline (gerados em {cov.generated_at or 'data desconhecida'})"
        elif cov.source_name == "APS":
            source = "APS (endpoint oficial, processado no app)"
        else:
            source = f"{cov.source_name} (cópia estática, pode estar desatualizada)"
        return (
            f"✅ **Base Ativa**\n"
            f"- Fonte: {source}\n"
            f"- Período: {cov.min_year} a {cov.max_year}-{cov.max_month:02d}\n"
            f"- Linhas (agregadas por mês): {cov.row_count:,}\n"
            f"- Carregado em: {cov.loaded_at}"
        )

    def procedencia(self, resultado: QueryResult) -> str:
        """Linha 'de onde veio este número', montada pelo código a partir dos filtros."""
        cov = self.data_coverage
        f = resultado.filtros
        partes = [f"Fonte: Mensário da APS, dados até {cov.max_month:02d}/{cov.max_year}", resultado.medida, f["periodo"]]
        rotulos = [
            ("terminais", "Terminais"), ("naturezas", "Natureza"), ("mercadorias", "Mercadorias"),
            ("sentido", "Sentido"), ("navegacao", "Navegação"), ("cargo_mix", "Cargo mix"),
        ]
        for chave, rotulo in rotulos:
            if f.get(chave) and f[chave] != ["todas"]:
                partes.append(f"{rotulo}: {', '.join(f[chave])}")
        return " · ".join(partes)

    # ------------------------------------------------------------------ #
    # Loop do agente
    # ------------------------------------------------------------------ #
    def _stream(self, messages: List[dict], usar_ferramentas: bool, separar: bool = False):
        """
        Uma chamada ao modelo em streaming. Gera os trechos de texto e retorna
        (texto, chamadas). `separar` insere uma quebra antes do primeiro trecho,
        quando já houve texto em um passo anterior.
        """
        kwargs = {
            "model": self.model_name,
            "messages": messages,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if self._aceita_temperatura:
            kwargs["temperature"] = 0.0
        if usar_ferramentas:
            kwargs["tools"] = TOOLS
        try:
            stream = self.client.chat.completions.create(**kwargs)
        except Exception as exc:
            # Alguns modelos (ex.: de raciocínio) só aceitam a temperatura padrão.
            if "temperature" not in kwargs or "temperature" not in str(exc):
                raise
            logger.info("O modelo %s não aceita temperature; seguindo sem ela.", self.model_name)
            self._aceita_temperatura = False
            kwargs.pop("temperature")
            stream = self.client.chat.completions.create(**kwargs)

        partes, chamadas = [], {}
        for chunk in stream:
            if not chunk.choices:  # o último pedaço traz só o uso de tokens
                continue
            delta = chunk.choices[0].delta
            if delta.content:
                if separar and not partes:
                    yield ("texto", "\n\n")
                partes.append(delta.content)
                yield ("texto", delta.content)
            # Pedidos de ferramenta chegam fatiados; o índice diz a qual chamada cada fatia pertence.
            for tc in delta.tool_calls or []:
                atual = chamadas.setdefault(tc.index, {"id": "", "name": "", "arguments": ""})
                atual["id"] = tc.id or atual["id"]
                if tc.function and tc.function.name:
                    atual["name"] += tc.function.name
                if tc.function and tc.function.arguments:
                    atual["arguments"] += tc.function.arguments
        return "".join(partes), [chamadas[i] for i in sorted(chamadas)]

    def _executar(self, chamada: dict, resultados: List[QueryResult], consultas: List[dict]) -> str:
        try:
            args = json.loads(chamada["arguments"] or "{}")
        except json.JSONDecodeError:
            return "Erro: argumentos da ferramenta não são um JSON válido."
        nome = chamada["name"]
        logger.info("[FERRAMENTA] %s %s", nome, json.dumps(args, ensure_ascii=False))

        if nome == "consultar_dados":
            try:
                resultado = run_query(self.store, **args)
            except Exception as exc:
                logger.exception("Erro ao executar consultar_dados.")
                return f"Erro ao executar a consulta: {exc}"
            consultas.append(args)
            if resultado.tabela is not None:
                resultados.append(resultado)
            return resultado.texto("pt")
        if nome == "buscar_documentos":
            return self.knowledge_agent.search(args.get("pergunta", ""))
        return f"Ferramenta desconhecida: {nome}"

    def responder(self, mensagem: str, historico: Optional[List[dict]] = None) -> Iterator[Tuple[str, object]]:
        """
        Gera eventos ("texto", trecho) durante a resposta e, no fim, ("fim", Turno).
        Nunca guarda estado na instância.
        """
        historico = list(historico or [])
        if not self.client:
            aviso = "⚠️ O serviço de IA não está configurado (OPENAI_API_KEY ausente)."
            yield ("texto", aviso)
            yield ("fim", Turno(aviso, historico=historico))
            return

        messages = [{"role": "system", "content": self.system_prompt}, *historico, {"role": "user", "content": mensagem}]
        partes, resultados, consultas = [], [], []

        for passo in range(MAX_PASSOS):
            ultimo_passo = passo == MAX_PASSOS - 1
            try:
                texto, chamadas = yield from self._stream(messages, not ultimo_passo, separar=bool(partes))
            except Exception:
                logger.exception("Erro na chamada ao modelo.")
                texto, chamadas = USER_FACING_ERROR, []
                yield ("texto", texto)
            if texto:
                partes.append(texto)
            if not chamadas:
                break

            messages.append(
                {
                    "role": "assistant",
                    "content": texto or None,
                    "tool_calls": [
                        {"id": c["id"], "type": "function", "function": {"name": c["name"], "arguments": c["arguments"]}}
                        for c in chamadas
                    ],
                }
            )
            for chamada in chamadas:
                conteudo = self._executar(chamada, resultados, consultas)
                messages.append({"role": "tool", "tool_call_id": chamada["id"], "content": conteudo})

        texto_final = "\n\n".join(partes)
        # O histórico guarda os filtros usados, para continuações ("e em 2024?")
        # partirem da consulta exata, e não de uma paráfrase.
        memoria = texto_final
        if consultas:
            memoria += f"\n\n[Consultas usadas nesta resposta: {json.dumps(consultas, ensure_ascii=False)}]"
        novo_historico = [*historico, {"role": "user", "content": mensagem}, {"role": "assistant", "content": memoria}]
        yield ("fim", Turno(texto_final, resultados, consultas, novo_historico[-MAX_HISTORICO:]))

    def responder_completo(self, mensagem: str, historico: Optional[List[dict]] = None) -> Turno:
        """Versão sem streaming de `responder` (usada pela avaliação)."""
        for evento, valor in self.responder(mensagem, historico):
            if evento == "fim":
                return valor
