import os
import json
from dotenv import load_dotenv
from openai import OpenAI

from agent.data_agent import PortDataAgent
from agent.knowledge_agent import KnowledgeAgent

load_dotenv()


class CoordinatorAgent:
    def __init__(self):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        print("Iniciando Agente Coordenador...")

        self.data_agent = PortDataAgent()
        self.knowledge_agent = KnowledgeAgent()

        self.memory = [
            {
                "role": "system",
                "content": """
Você é o Agente Coordenador Especialista do Porto de Santos.

Sua função é atender o usuário, manter o contexto da conversa e coordenar agentes especialistas.

Você possui acesso aos seguintes agentes subordinados:
- data_analyst: especialista em dados quantitativos (TEUs, market share, crescimento, mix de carga, volumes, comparações e séries).
- knowledge_specialist: especialista em conhecimento documental e qualitativo (investimentos, expansão, regras, contexto institucional, explicações baseadas em documentos).

Regras de roteamento:
1. Se a pergunta for um cumprimento simples, responda diretamente.
2. Se a pergunta for quantitativa, chame consult_data_analyst.
3. Se a pergunta for qualitativa, documental, regulatória ou explicativa, chame consult_knowledge_specialist.
4. Se a pergunta exigir números + contexto, você pode chamar os dois agentes e consolidar a resposta.
5. Preserve o contexto recente da conversa para resolver follow-ups como:
   - "e a BTP?"
   - "agora compare com 2025"
   - "e no porto como um todo?"
6. Ao responder ao usuário, seja claro, objetivo e natural.
"""
            }
        ]

        self.tools = [
            {
                "type": "function",
                "function": {
                    "name": "consult_data_analyst",
                    "description": "Encaminha a pergunta para o agente analista de dados. Use para perguntas numéricas, séries, market share, crescimento, mix, TEUs, volumes e comparações.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Pergunta reformulada com contexto suficiente para o analista de dados."
                            }
                        },
                        "required": ["query"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "consult_knowledge_specialist",
                    "description": "Encaminha a pergunta para o agente de conhecimento. Use para perguntas qualitativas, regulatórias, documentais, institucionais e contextuais.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Pergunta reformulada com contexto suficiente para o agente de conhecimento."
                            }
                        },
                        "required": ["query"]
                    }
                }
            }
        ]

    def _build_recent_context(self, max_messages: int = 6) -> str:
        recent = []

        for msg in self.memory[-max_messages:]:
            if isinstance(msg, dict) and msg.get("role") in {"user", "assistant"}:
                role = msg.get("role", "")
                content = msg.get("content", "")
                recent.append(f"{role}: {content}")

        return "\n".join(recent)

    def chat(self, user_message: str) -> str:
        self.memory.append({"role": "user", "content": user_message})

        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=self.memory,
            tools=self.tools,
            tool_choice="auto"
        )

        response_message = response.choices[0].message
        final_text = response_message.content or ""

        if response_message.tool_calls:
            self.memory.append(response_message)
            recent_context = self._build_recent_context()

            for tool_call in response_message.tool_calls:
                function_name = tool_call.function.name
                args = json.loads(tool_call.function.arguments or "{}")
                worker_query = args.get("query", user_message)

                print(f"🔄 [Coordenador] Tool acionada: {function_name}")
                print(f"🔄 [Coordenador] Query delegada: {worker_query}")

                if function_name == "consult_data_analyst":
                    worker_response = self.data_agent.ask(
                        worker_query,
                        context=recent_context
                    )

                elif function_name == "consult_knowledge_specialist":
                    worker_response = self.knowledge_agent.ask(
                        worker_query,
                        context=recent_context
                    )

                else:
                    worker_response = "Erro: ferramenta de coordenação não reconhecida."

                self.memory.append({
                    "tool_call_id": tool_call.id,
                    "role": "tool",
                    "name": function_name,
                    "content": str(worker_response)
                })

            final_response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=self.memory
            )

            final_text = final_response.choices[0].message.content or ""

        self.memory.append({"role": "assistant", "content": final_text})
        return final_text


if __name__ == "__main__":
    coordenador = CoordinatorAgent()

    print("\n--- Iniciando Chat (Digite 'sair' para encerrar) ---")
    while True:
        pergunta = input("\nVocê: ")
        if pergunta.lower() in ["sair", "exit", "quit"]:
            break

        resposta = coordenador.chat(pergunta)
        print(f"\nCoordenador: {resposta}")