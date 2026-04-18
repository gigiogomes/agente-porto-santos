import json
import os
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

try:
    from agent.data_agent import PortDataAgent
    from agent.knowledge_agent import KnowledgeAgent
except ImportError:
    from data_agent import PortDataAgent
    from knowledge_agent import KnowledgeAgent

load_dotenv()

class CoordinatorAgent:
    def __init__(self, session_id: Optional[str] = None):
        self.session_id = session_id
        self.model_name = os.getenv("COORDINATOR_AGENT_MODEL", "gpt-4o")
        self.client = None

        api_key = os.getenv("OPENAI_API_KEY")
        if api_key:
            try:
                from openai import OpenAI
                self.client = OpenAI(api_key=api_key)
            except Exception:
                self.client = None

        print("Iniciando Agente Coordenador...")
        self.data_agent = PortDataAgent()
        self.knowledge_agent = KnowledgeAgent()
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
"""
        }

        self.history: List[Dict[str, str]] = []
        self.max_history_messages = 8

        self.tools = [
            {
                "type": "function",
                "function": {
                    "name": "consult_data_analyst",
                    "description": "Encaminha a pergunta de dados ou estatísticas para o analista de dados.",
                    "parameters": {
                        "type": "object",
                        "properties": {"query": {"type": "string", "description": "A requisição original do usuário"}},
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

    def chat(self, user_message: str) -> str:
        if not self.client:
            return "Erro: OpenAI API key não configurada no Agente Coordenador."

        recent_context = "\n".join(f"{msg['role']}: {msg['content']}" for msg in self.history[-4:])
        
        model_messages = [self.system_message] + self.history + [{"role": "user", "content": user_message}]

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=model_messages,
            tools=self.tools,
            temperature=0.0
        )

        response_message = response.choices[0].message

        if not response_message.tool_calls:
            final_text = response_message.content or ""
            self.history.extend([
                {"role": "user", "content": user_message},
                {"role": "assistant", "content": final_text},
            ])
            self.history = self.history[-self.max_history_messages :]
            return final_text

        model_messages.append(response_message)
        for tool_call in response_message.tool_calls:
            function_name = tool_call.function.name
            args = json.loads(tool_call.function.arguments)
            worker_query = args.get("query", user_message)

            print(f"\n🤖 [COORDENADOR] Encaminhando para: {function_name}")
            print(f"🧠 [RACIOCÍNIO] Tradução da pergunta: {worker_query}")

            # GARANTA QUE O CONTEXTO SEJA EXTRAÍDO ASSIM:
            recent_context = "\n".join([f"{msg['role']}: {msg['content']}" for msg in self.history[-4:]])
            
            # REFORCE A PERGUNTA COM O CONTEXTO PARA O SUB-AGENTE:
            enriched_query = f"Contexto da conversa:\n{recent_context}\n\nPergunta atual: {worker_query}"

            try:
                if function_name == "consult_data_analyst":
                    worker_response = self.data_agent.ask(enriched_query) # Passando a pergunta enriquecida
                elif function_name == "consult_knowledge_specialist":
                    worker_response = self.knowledge_agent.ask(enriched_query)
                else:
                    worker_response = "Erro: ferramenta desconhecida."
            except Exception as exc:
                worker_response = f"Falha na execução: {exc}"

            model_messages.append({
                "tool_call_id": tool_call.id,
                "role": "tool",
                "name": function_name,
                "content": str(worker_response),
            })

        final_response = self.client.chat.completions.create(model=self.model_name, messages=model_messages)
        final_text = final_response.choices[0].message.content or ""
        
        self.history.extend([
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": final_text},
        ])
        self.history = self.history[-self.max_history_messages :]
        return final_text