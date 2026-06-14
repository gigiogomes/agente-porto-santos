import json
import logging
import os
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv

try:
    from agent.data_agent import PortDataAgent, default_filters
except ImportError:
    from data_agent import PortDataAgent, default_filters

try:
    from agent.knowledge_agent import KnowledgeAgent
except ImportError:
    from knowledge_agent import KnowledgeAgent

load_dotenv()

logger = logging.getLogger(__name__)

# Mensagem genérica exibida ao usuário. O detalhe técnico vai apenas para o log.
USER_FACING_ERROR = (
    "Ocorreu um erro ao processar a sua solicitação. "
    "Tente novamente em instantes."
)


class CoordinatorAgent:
    """
    Orquestrador dos sub-agentes.

    IMPORTANTE: esta instância é compartilhada entre todas as sessões (cache
    global do Streamlit). Por isso ela NÃO guarda memória de conversa nem
    filtros mutáveis. O histórico e os filtros vivem na sessão do usuário e são
    passados/retornados a cada chamada de `chat`, mantendo a instância segura
    sob concorrência.
    """

    def __init__(
        self,
        session_id: Optional[str] = None,
        request_timeout: float = 60.0,
        max_retries: int = 2,
    ):
        self.session_id = session_id
        self.model_name = os.getenv("COORDINATOR_AGENT_MODEL", "gpt-4o")
        self.client = None

        api_key = os.getenv("OPENAI_API_KEY")
        if api_key:
            try:
                from openai import OpenAI

                # timeout e max_retries cobrem falhas transitórias (429/5xx).
                self.client = OpenAI(
                    api_key=api_key,
                    timeout=request_timeout,
                    max_retries=max_retries,
                )
            except Exception:
                logger.exception("Falha ao inicializar o cliente OpenAI no coordinator.")
                self.client = None

        logger.info("Iniciando Agente Coordenador...")
        self.data_agent = PortDataAgent(
            request_timeout=request_timeout, max_retries=max_retries
        )
        self.knowledge_agent = KnowledgeAgent()

        self.max_history_messages = 8

        self.system_message = {
            "role": "system",
            "content": """
            Você é o Assistente Virtual Principal do Porto de Santos.
            Sua função é atender o usuário, classificar a intenção e buscar os dados necessários usando suas ferramentas (agentes subordinados).

            [REGRA DE OURO - COMPORTAMENTO]
            Você DEVE agir como um assistente único. NUNCA diga coisas como "O analista de dados informou", "Encaminhei sua pergunta", ou "Consultei o especialista".
            Apenas receba a resposta da ferramenta e entregue a informação diretamente ao usuário de forma natural, direta e em primeira pessoa.

            [FERRAMENTAS]
            - consult_data_analyst: use para perguntas quantitativas, dados, volume, market share, crescimento, prazos, comparações e também perguntas sobre MIX DE CARGA (cargo mix), tipo de carga, cheio/vazio, cabotagem, etc.
            - consult_knowledge_specialist: use para perguntas qualitativas e textuais sobre as regras do porto.
""",
        }

        self.tools = [
            {
                "type": "function",
                "function": {
                    "name": "consult_data_analyst",
                    "description": "Encaminha a pergunta de dados ou estatísticas para o analista de dados.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "A requisição original do usuário",
                            }
                        },
                        "required": ["query"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "consult_knowledge_specialist",
                    "description": "Encaminha a pergunta conceitual/qualitativa para o especialista de documentos.",
                    "parameters": {
                        "type": "object",
                        "properties": {"query": {"type": "string"}},
                        "required": ["query"],
                    },
                },
            },
        ]

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def _create_completion(self, messages: List[Dict[str, Any]], use_tools: bool):
        kwargs: Dict[str, Any] = {
            "model": self.model_name,
            "messages": messages,
            "temperature": 0.0,
        }
        if use_tools:
            kwargs["tools"] = self.tools
        return self.client.chat.completions.create(**kwargs)

    def _append_turn(
        self,
        history: List[Dict[str, str]],
        user_message: str,
        assistant_text: str,
    ) -> List[Dict[str, str]]:
        history = history + [
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": assistant_text},
        ]
        return history[-self.max_history_messages :]

    # ------------------------------------------------------------------ #
    # API principal
    # ------------------------------------------------------------------ #
    def chat(
        self,
        user_message: str,
        history: Optional[List[Dict[str, str]]] = None,
        data_filters: Optional[Dict[str, Any]] = None,
    ) -> Tuple[str, List[Dict[str, str]], Dict[str, Any]]:
        """
        Processa uma mensagem do usuário.

        Recebe e devolve o estado da sessão (histórico e filtros do analista),
        sem nunca guardá-lo na instância. Retorna:
            (resposta_final, novo_historico, novos_filtros)
        """
        # Cópias defensivas: nunca mutar o estado recebido da sessão in-place.
        history = list(history or [])
        data_filters = dict(data_filters or default_filters())

        if not self.client:
            return (
                "⚠️ O serviço de IA não está configurado (OPENAI_API_KEY ausente).",
                history,
                data_filters,
            )

        model_messages = (
            [self.system_message]
            + history
            + [{"role": "user", "content": user_message}]
        )

        # 1ª chamada: o coordenador decide se usa ferramenta.
        try:
            response = self._create_completion(model_messages, use_tools=True)
        except Exception:
            logger.exception("Erro na chamada de roteamento do coordinator.")
            return (USER_FACING_ERROR, history, data_filters)

        response_message = response.choices[0].message

        # Sem ferramenta: resposta direta.
        if not response_message.tool_calls:
            final_text = response_message.content or ""
            history = self._append_turn(history, user_message, final_text)
            return (final_text, history, data_filters)

        # Com ferramenta: executa cada chamada e devolve o resultado ao modelo.
        model_messages.append(response_message)
        recent_context = "\n".join(
            f"{msg['role']}: {msg['content']}" for msg in history[-4:]
        )

        for tool_call in response_message.tool_calls:
            function_name = tool_call.function.name
            try:
                args = json.loads(tool_call.function.arguments)
            except (json.JSONDecodeError, TypeError):
                args = {}
            worker_query = args.get("query", user_message)

            logger.info("[COORDENADOR] Encaminhando para: %s", function_name)
            logger.info("[RACIOCINIO] Tradução da pergunta: %s", worker_query)

            enriched_query = (
                f"Contexto da conversa:\n{recent_context}\n\nPergunta atual: {worker_query}"
                if recent_context
                else worker_query
            )

            try:
                if function_name == "consult_data_analyst":
                    # O analista é stateless: recebe e devolve os filtros da sessão.
                    worker_response, data_filters = self.data_agent.ask(
                        enriched_query, data_filters
                    )
                elif function_name == "consult_knowledge_specialist":
                    worker_response = self.knowledge_agent.ask(enriched_query)
                else:
                    worker_response = "Erro: ferramenta desconhecida."
            except Exception:
                logger.exception("Falha ao executar a ferramenta %s.", function_name)
                worker_response = (
                    f"Não foi possível obter os dados solicitados ({function_name})."
                )

            model_messages.append(
                {
                    "tool_call_id": tool_call.id,
                    "role": "tool",
                    "name": function_name,
                    "content": str(worker_response),
                }
            )

        # 2ª chamada: o coordenador sintetiza a resposta final (temperature=0).
        try:
            final_response = self._create_completion(model_messages, use_tools=False)
            final_text = final_response.choices[0].message.content or ""
        except Exception:
            logger.exception("Erro na chamada de síntese do coordinator.")
            return (USER_FACING_ERROR, history, data_filters)

        history = self._append_turn(history, user_message, final_text)
        return (final_text, history, data_filters)
