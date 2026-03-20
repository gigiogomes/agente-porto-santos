import os
import json
from dotenv import load_dotenv
from openai import OpenAI

# Importando o agente especialista (Worker)
from agent.data_agent import PortDataAgent

load_dotenv()

class CoordinatorAgent:
    def __init__(self):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        print("Iniciando Agente Coordenador...")
        # Inicializa o agente subordinado
        self.data_agent = PortDataAgent()
        
        # SessionMemory: Memória de curto prazo para manter o contexto da conversa
        self.memory = [
            {
                "role": "system", 
                "content": """
                Você é o Agente Coordenador Especialista do Porto de Santos.
                Sua função é atender o usuário, manter o contexto da conversa e coordenar outros agentes.
                
                No momento, você tem acesso ao seguinte agente subordinado:
                - data_analyst: Especialista em dados quantitativos (TEUs, Market Share, Crescimento, Mix de Carga).
                
                Regras de Raciocínio (Think-Then-Answer):
                1. Se a pergunta for sobre cumprimentos genéricos, responda diretamente.
                2. Se a pergunta envolver números, movimentação, market share ou terminais, SEMPRE chame a ferramenta 'consult_data_analyst'.
                3. Ao receber a resposta do data_analyst, repasse a informação ao usuário de forma amigável, clara e formatada.
                """
            }
        ]
        
        # Registrando o Worker Agent como uma Tool (A2A Messaging)
        self.tools = [
            {
                "type": "function",
                "function": {
                    "name": "consult_data_analyst",
                    "description": "Delega a pergunta para o Agente Analista de Dados. Use isso para qualquer pergunta sobre números, TEUs, terminais ou market share.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string", 
                                "description": "A pergunta exata ou instrução reformulada com todo o contexto necessário para o analista de dados."
                            }
                        },
                        "required": ["query"]
                    }
                }
            }
        ]

    def chat(self, user_message: str) -> str:
        """Gerencia o fluxo da conversa, memória e roteamento para sub-agentes."""
        
        # Adiciona a mensagem do usuário à memória da sessão
        self.memory.append({"role": "user", "content": user_message})
        
        # Passo 1: O Coordenador pensa e decide se responde ou delega
        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=self.memory,
            tools=self.tools,
            tool_choice="auto"
        )
        
        response_message = response.choices[0].message
        
        # Se o Coordenador decidiu chamar o Agente de Dados (A2A Protocol)
        if response_message.tool_calls:
            # Salva a intenção de chamada na memória
            self.memory.append(response_message)
            
            for tool_call in response_message.tool_calls:
                function_name = tool_call.function.name
                
                if function_name == "consult_data_analyst":
                    args = json.loads(tool_call.function.arguments)
                    worker_query = args.get("query")
                    
                    print(f"🔄 [Coordenador] Delegando para DataAgent: '{worker_query}'")
                    
                    # Comunicação A2A: Chama o agente de dados e aguarda a resposta
                    worker_response = self.data_agent.ask(worker_query)
                    
                    # Devolve a resposta do worker para o Coordenador refletir
                    self.memory.append({
                        "tool_call_id": tool_call.id,
                        "role": "tool",
                        "name": function_name,
                        "content": worker_response
                    })
            
            # Passo 2: O Coordenador lê a resposta do worker e formula a resposta final
            final_response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=self.memory
            )
            
            final_text = final_response.choices[0].message.content
            
        else:
            # O Coordenador respondeu sozinho (ex: "Olá, bom dia!")
            final_text = response_message.content
            
        # Salva a resposta final na memória para o próximo turno da conversa
        self.memory.append({"role": "assistant", "content": final_text})
        
        return final_text

# ==========================================
# TESTE DO FLUXO MULTIAGENTE COM MEMÓRIA
# ==========================================
if __name__ == "__main__":
    coordenador = CoordinatorAgent()
    
    print("\n--- Iniciando Chat (Digite 'sair' para encerrar) ---")
    while True:
        pergunta = input("\nVocê: ")
        if pergunta.lower() in ['sair', 'exit', 'quit']:
            break
            
        resposta = coordenador.chat(pergunta)
        print(f"\nCoordenador: {resposta}")