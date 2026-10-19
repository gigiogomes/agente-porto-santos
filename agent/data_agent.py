import json
import logging
import os
import sys
from typing import Any, Dict, Optional, Tuple

from dotenv import load_dotenv

# ROTAS
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)

if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

try:
    from tools.port_data_tools import PortDataStore, load_port_data, query_port_data
except ImportError:
    from port_data_tools import PortDataStore, load_port_data, query_port_data

load_dotenv()

logger = logging.getLogger(__name__)


def default_filters() -> Dict[str, Any]:
    """Estado inicial dos filtros de conversa do analista (por sessão)."""
    return {
        "start_date": None,
        "end_date": None,
        "terminals": [],
        "metric": "teus",
    }


class PortDataAgent:
    """
    Analista de dados.

    A instância é compartilhada entre sessões (cache global), portanto guarda
    apenas o que é imutável: a base de consulta (DuckDB) e a cobertura dos dados.
    A "memória de curto prazo" (filtros) NÃO vive aqui — é recebida e devolvida
    a cada chamada de `ask`, evitando vazamento entre usuários sob concorrência.
    """

    def __init__(self, request_timeout: float = 60.0, max_retries: int = 2):
        logger.info("Iniciando Agente Analista de Dados...")
        fato, self.data_coverage, self.data_diagnostics = load_port_data()
        self.store = PortDataStore(fato)

        api_key = os.getenv("OPENAI_API_KEY")
        self.client = None
        self.model_name = os.getenv("PORT_DATA_AGENT_MODEL", "gpt-4o")
        if api_key:
            try:
                from openai import OpenAI

                self.client = OpenAI(
                    api_key=api_key,
                    timeout=request_timeout,
                    max_retries=max_retries,
                )
            except Exception:
                logger.exception("Falha ao inicializar o cliente OpenAI no data agent.")
                self.client = None

        self.tools = [
            {
                "type": "function",
                "function": {
                    "name": "query_port_data",
                    "description": "Faz qualquer consulta de dados, agregação ou comparação de períodos no formato YYYY-MM-DD. Use para volumes (TEU, contêineres ou toneladas), market share, crescimento, cargo mix, granéis e mercadorias.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "metric": {
                                "type": "string",
                                "enum": ["teus", "conteineres", "toneladas", "market_share", "cagr"],
                                "description": "Qual métrica extrair",
                            },
                            "medida": {
                                "type": "string",
                                "enum": ["teu", "unidades", "toneladas"],
                                "description": "Base de cálculo para market_share, cagr e comparações. Padrão: teu (ou toneladas quando metric='toneladas'). Use 'toneladas' para granéis, carga geral, mercadorias ou o porto inteiro em peso.",
                            },
                            "natureza_carga": {
                                "type": "array",
                                "items": {
                                    "type": "string",
                                    "enum": ["CARGA CONTEINERIZADA", "GRANEL SOLIDO", "GRANEL LIQUIDO", "CARGA GERAL"],
                                },
                                "description": "Filtra por natureza de carga. Padrão: só contêineres quando a medida é teu/unidades; todas quando é toneladas.",
                            },
                            "mercadorias": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Mercadorias pelo nome da lista disponível (ex: ['SOJA EM GRAOS']). Nome exato; se não houver, busca parcial (ex: 'SOJA' traz soja em grãos e farelo).",
                            },
                            "sentido": {
                                "type": "array",
                                "items": {"type": "string", "enum": ["EMBARQUE", "DESEMBARQUE"]},
                                "description": "EMBARQUE = carga saindo por navio; DESEMBARQUE = carga chegando. Exportação/importação também exigem navegacao=['LONGO CURSO'].",
                            },
                            "navegacao": {
                                "type": "array",
                                "items": {"type": "string", "enum": ["LONGO CURSO", "CABOTAGEM"]},
                                "description": "LONGO CURSO = comércio exterior; CABOTAGEM = entre portos brasileiros.",
                            },
                            "start_date": {
                                "type": "string",
                                "description": "Mês inicial em YYYY-MM (YYYY-MM-DD também é aceito; o dia é ignorado). Ex: 2021-01",
                            },
                            "end_date": {
                                "type": "string",
                                "description": "Mês final (inclusivo) em YYYY-MM (YYYY-MM-DD também é aceito; o dia é ignorado). Ex: 2023-12",
                            },
                            "terminals": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Lista de terminais (siglas oficiais ou nomes da base). Deixe vazio para o porto inteiro.",
                            },
                            "cargo_mix_filter": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Filtra por tipos específicos de Cargo Mix (ex: ['REMOÇÃO', 'EXPORTAÇÃO']).",
                            },
                            "group_by": {
                                "type": "array",
                                "items": {
                                    "type": "string",
                                    "enum": [
                                        "ano",
                                        "semestre",
                                        "quadrimestre",
                                        "trimestre",
                                        "bimestre",
                                        "mes",
                                        "terminal",
                                        "cargo_mix",
                                        "natureza_carga",
                                        "mercadoria",
                                        "sentido",
                                        "navegacao",
                                    ],
                                },
                                "description": "Como agrupar os dados. Pode cruzar informações passando mais de um (ex: ['mes', 'terminal']).",
                            },
                            "compare_with_previous": {
                                "type": "boolean",
                                "description": "True se o usuário pedir crescimento (YoY/MoM/Crescimento) para calcular a %",
                            },
                        },
                        "required": ["metric"],
                    },
                },
            }
        ]

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _update_context_from_args(
        filters: Dict[str, Any], args: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Devolve uma cópia dos filtros atualizada pela última ferramenta chamada."""
        updated = dict(filters)
        if args.get("start_date"):
            updated["start_date"] = args["start_date"]
        if args.get("end_date"):
            updated["end_date"] = args["end_date"]
        if args.get("terminals"):
            updated["terminals"] = args["terminals"]
        if args.get("metric"):
            updated["metric"] = args["metric"]
        return updated

    def _create_completion(self, messages, use_tools: bool):
        kwargs: Dict[str, Any] = {
            "model": self.model_name,
            "messages": messages,
            "temperature": 0.0,
        }
        if use_tools:
            kwargs["tools"] = self.tools
        return self.client.chat.completions.create(**kwargs)

    def get_status_summary(self) -> str:
        """Resumo da saúde dos dados para a interface do Streamlit."""
        if not getattr(self, "data_coverage", None):
            return "Nenhum dado carregado."

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

    # ------------------------------------------------------------------ #
    # API principal
    # ------------------------------------------------------------------ #
    def ask(
        self, question: str, filters: Optional[Dict[str, Any]] = None
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Responde a uma pergunta de dados.

        Recebe os filtros da sessão e devolve (resposta, filtros_atualizados).
        Nunca guarda filtros na instância.
        """
        filters = dict(filters or default_filters())

        if not self.client:
            return (
                "Erro: cliente OpenAI não configurado para o Agente de Dados.",
                filters,
            )

        max_y = self.data_coverage.max_year
        max_m = self.data_coverage.max_month
        mix_list = (
            ", ".join(str(x) for x in self.data_coverage.cargo_mix_list)
            if self.data_coverage.cargo_mix_list
            else "Não identificados"
        )
        terminal_list = ", ".join(self.data_coverage.terminal_list) or "Não identificados"
        mercadoria_list = ", ".join(self.data_coverage.mercadoria_list) or "Não identificadas"
        natureza_list = ", ".join(self.data_coverage.natureza_list) or "Não identificadas"

        system_prompt = f"""
        Você é o Analista de Dados do Porto.

        [COBERTURA DE DADOS]
        Você tem dados carregados até: {max_y}-{max_m:02d} (Este é o "mês atual" ou "último mês disponível").
        Cargo Mix disponíveis nesta base: {mix_list}
        Naturezas de carga: {natureza_list}
        Mercadorias: {mercadoria_list}

        [REGRAS DE TEMPO RELATIVO - EXTREMAMENTE IMPORTANTE]
        1. Omissão de data: Se o usuário pedir algo genérico (ex: "Qual o volume da BTP?"), assuma automaticamente o mês mais recente ({max_y}-{max_m:02d}) configurando `start_date` e `end_date` para este mês.
        2. "Último mês": Use `start_date` e `end_date` cobrindo o mês {max_y}-{max_m:02d}.
        3. "Último ano": Avalie o contexto. Pode ser o ano cheio de {max_y} ou, se estivermos no começo do ano, o ano consolidado de {max_y - 1}.

        [REGRAS DE COMPARAÇÃO - YoY / MoM]
        Para calcular o crescimento ou comparar períodos, você DEVE buscar os dois períodos na ferramenta definindo o `start_date` e `end_date` de forma abrangente e ativando `compare_with_previous=True`.
        - Exemplo YoY (Year-over-Year): Para comparar o último mês com o mesmo mês do ano anterior, defina start_date='{max_y-1}-{max_m:02d}-01', end_date='{max_y}-{max_m:02d}-31' e group_by='mes' ou 'ano'.
        - Não tente fazer a matemática sozinho; deixe a ferramenta agregar agrupando pelo período.

        [REGRA DE YoY - OBRIGATÓRIO - LEIA COM ATENÇÃO]
        1. Sempre que o usuário pedir comparação entre um ano e o anterior (YoY, "crescimento de X para Y", "variação ano a ano"), ative `compare_with_previous=True` e inclua AMBOS os anos na janela: start_date no dia 01-01 do ano ANTERIOR e end_date no dia 31-12 do ano ATUAL.
        2. Para YoY por cargo mix, passe group_by=["cargo_mix"]. Para YoY por terminal, group_by=["terminal"].
        3. A ferramenta JÁ retorna as colunas prontas: "<medida> <ano_anterior>", "<medida> <ano_atual>" (ex.: "TEU 2025", "Toneladas 2025"), "Diferença" e "YoY (%)". Você DEVE apenas LER e repassar esses valores. NUNCA recalcule a diferença ou o percentual por conta própria.
        4. Se a resposta da ferramenta começar com "[YoY ACUMULADO/YTD: meses 1 a N...]", o ano atual está incompleto. Você é OBRIGADO a avisar o usuário que a comparação é acumulada até o mês N (não é o ano cheio) e a citar exatamente o período. Não apresente esses números como totais anuais.
        5. Se a coluna YoY (%) vier como "N/D", explique que aquele item não tinha movimentação no ano anterior, e não invente um número.

        [CONTEXTO DA CONVERSA ATUAL]
        Data Inicial foco: {filters['start_date'] or 'Não definida'}
        Data Final foco: {filters['end_date'] or 'Não definida'}
        Terminais: {', '.join(filters['terminals']) if filters['terminals'] else 'Todos'}
        Métrica Ativa: {filters['metric']}

        [REGRAS DE CARGO MIX - OBRIGATÓRIO]
        1. A coluna de Cargo Mix chama-se "ESTADO". Os valores válidos exatos são: {mix_list}.
        2. Se o usuário perguntar "quais os cargo mix", cite a lista acima diretamente sem usar ferramentas.
        3. Se pedir sobre "remoção" ou "exportação", você DEVE acionar a ferramenta usando a propriedade `cargo_mix_filter` com o valor exato em MAIÚSCULO. Ex: `cargo_mix_filter=["REMOÇÃO"]`.
        4. Se perguntar "Qual o share por cargo mix" ou "proporção", você DEVE acionar a ferramenta passando `group_by="cargo_mix"`.
        5. Se o usuário pedir "mix de carga" de um terminal, responda os volumes detalhados usando group_by=["cargo_mix"].
        6. CRUZAMENTOS: Se o usuário pedir "por terminal mês a mês", use group_by=["mes", "terminal"] juntos.

        [REGRAS DE TERMINAIS - OBRIGATÓRIO]
        O usuário pode digitar o nome completo, mas prefira a SIGLA OFICIAL ao usar o filtro 'terminals' da ferramenta:
        - "Santos Brasil" -> "SBSA"
        - "DP World" ou "Embraport" -> "DPW"
        - "Brasil Terminal Portuário" -> "BTP"
        - "Ecoporto" -> "ECOPORTO"
        Valores existentes na base, do maior para o menor volume (inclui terminais históricos já desativados): {terminal_list}
        Se o usuário citar um terminal que tem mais de um valor na lista (ex.: "Libra" -> "LIBRA 35", "LIBRA 37"), passe todos eles.

        [REGRAS DE MÉTRICAS E SHARE]
        1. Se o usuário usar as palavras "SHARE", "PARTICIPAÇÃO", "PROPORÇÃO" ou "MIX", você DEVE obrigatoriamente usar metric="market_share".
        2. O resultado voltará com uma coluna 'Share (%)'. Você deve priorizar a exibição deste valor percentual na sua resposta final.
        3. Sempre que mostrar o Share, mencione também o volume absoluto (TEUs) logo ao lado para contexto.
        4. Para "Cargo Mix", use sempre group_by="cargo_mix" e metric="market_share".
        5. CÁLCULO DE YOY (Year-over-Year): Se o usuário pedir "Crescimento YoY por terminal" para um ano específico (ex: 2026), defina compare_with_previous=True, start_date="2025-01-01" (ano anterior) e use group_by=["terminal", "ano"]. A ferramenta já ajustará os meses automaticamente para uma comparação justa.

        [REGRA DE CAGR - OBRIGATÓRIO]
        1. Para "CAGR", "taxa composta de crescimento", "crescimento médio anual composto" ou similar, você DEVE chamar a ferramenta com metric="cagr".
        2. NUNCA calcule o CAGR você mesmo, nem estime. A ferramenta calcula a partir dos dados reais e já usa SOMENTE anos completos (12 meses).
        3. Para CAGR de um terminal específico, passe esse terminal em 'terminals' (sigla oficial). Para comparar vários terminais, use group_by=["terminal"].
        4. Defina start_date no início do período pedido. Ex.: "últimos dez anos" -> start_date dez anos antes do último ano disponível ({max_y}); end_date pode cobrir {max_y}-12-31.
        5. A ferramenta retorna EXATAMENTE quais anos completos usou (ex.: "2016 a 2024") e os volumes inicial/final. Você DEVE repassar esses anos na resposta e NUNCA afirmar um intervalo diferente do que a ferramenta reportou.
        6. Se a ferramenta retornar "N/D" para um terminal, explique que ele não tinha movimentação no ano inicial e por isso o CAGR daquele período não é calculável — não invente um número.

        [REGRAS DE TONELADAS, GRANÉIS E MERCADORIAS - OBRIGATÓRIO]
        1. TEU e contêineres (unidades) existem só para carga conteinerizada; sem outro filtro, as métricas teus, conteineres, market_share e cagr consideram apenas contêineres.
        2. Perguntas em toneladas/peso, sobre granéis, carga geral, mercadorias (soja, açúcar, milho, combustíveis...) ou "movimentação total do porto": use metric="toneladas". Para share, CAGR ou comparação em peso, mantenha a métrica pedida e passe medida="toneladas".
        3. Mercadorias: use os nomes exatos da lista de mercadorias em `mercadorias`. Para "soja", prefira "SOJA EM GRAOS" e cite que farelo de soja é outra mercadoria, a menos que o usuário peça ambos.
        4. Exportação de granel ou mercadoria -> sentido=["EMBARQUE"] e navegacao=["LONGO CURSO"]; importação -> sentido=["DESEMBARQUE"] e navegacao=["LONGO CURSO"] (cabotagem é transporte entre portos brasileiros, não comércio exterior). "Embarque"/"desembarque" sem falar em exportação/importação -> só sentido. Cargo mix (cargo_mix_filter) só existe para contêineres.
        5. Para quebrar por natureza, mercadoria, sentido ou tipo de navegação, use group_by com "natureza_carga", "mercadoria", "sentido" ou "navegacao".
        6. Se a resposta da ferramenta começar com "[Escopo: ...]", respeite e informe ao usuário a medida e os filtros ali descritos. Sempre diga a unidade (TEU, unidades ou toneladas).
"""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ]

        try:
            response = self._create_completion(messages, use_tools=True)
        except Exception:
            logger.exception("Erro na chamada de roteamento do data agent.")
            return (
                "Não foi possível consultar os dados no momento. Tente novamente em instantes.",
                filters,
            )

        response_message = response.choices[0].message

        if not response_message.tool_calls:
            return (response_message.content or "", filters)

        messages.append(response_message)
        for tool_call in response_message.tool_calls:
            function_name = tool_call.function.name
            try:
                raw_args = json.loads(tool_call.function.arguments)
            except (json.JSONDecodeError, TypeError):
                raw_args = {}

            logger.info("[DATA AGENT] Executando: %s", function_name)
            logger.info(
                "[FILTROS] Argumentos: %s",
                json.dumps(raw_args, ensure_ascii=False),
            )

            # Atualiza a memória de filtros para a próxima interação (sem mutar a instância).
            filters = self._update_context_from_args(filters, raw_args)

            try:
                if function_name == "query_port_data":
                    tool_result = query_port_data(self.store, **raw_args)
                else:
                    tool_result = f"Ferramenta {function_name} não encontrada."
            except Exception as exc:
                logger.exception("Erro ao executar query_port_data.")
                tool_result = f"Erro ao executar a consulta: {exc}"

            messages.append(
                {
                    "tool_call_id": tool_call.id,
                    "role": "tool",
                    "name": function_name,
                    "content": str(tool_result),
                }
            )

        try:
            final_response = self._create_completion(messages, use_tools=False)
            return (final_response.choices[0].message.content or "", filters)
        except Exception:
            logger.exception("Erro na chamada de síntese do data agent.")
            return (
                "Não foi possível resumir os dados no momento. Tente novamente em instantes.",
                filters,
            )
