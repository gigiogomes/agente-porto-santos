import os
import sys
import json
from dotenv import load_dotenv
from openai import OpenAI

# Garante que o Python encontre a pasta tools se rodar o arquivo isoladamente
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Importando as ferramentas atualizadas
from tools.port_data_tools import (
    load_port_data, 
    get_market_share, 
    get_cargo_mix, 
    get_growth,
    get_volume,
    plotar_grafico
)

# Carrega as variáveis do .env
load_dotenv()

class PortDataAgent:
    def __init__(self):
        # Inicializa o cliente da OpenAI
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        # Carrega o DataFrame na memória do Agente (URL ou Fallback)
        print("Iniciando Agente Analista de Dados...")
        self.df = load_port_data()
        
        # --- LÓGICA DINÂMICA DE TEMPO ---
        # Descobre automaticamente qual o ano e os meses mais recentes na base de dados
        max_ano = int(self.df['ANO'].max())
        meses_disponiveis = self.df[self.df['ANO'] == max_ano]['MES'].unique().tolist()
        meses_str = str(meses_disponiveis) # Vai gerar algo como "[1]" ou "[1, 2, 3]"
        
        # Define o Prompt de Sistema com f-string (injeta as variáveis reais)
        # Define o Prompt de Sistema com f-string (injeta as variáveis reais)
        self.system_prompt = f"""
        Você é o Agente Especialista em Dados do Porto de Santos.
        Seu objetivo é responder perguntas precisas sobre movimentação de carga, volumes (TEUs), unidades, market share e crescimento dos terminais.
        
        CONTEXTO TEMPORAL DINÂMICO:
        O ano atual é 2026. A sua base de dados possui dados consolidados do ano de {max_ano} APENAS para os meses: {meses_str}. 
        Trate o ano de 2025 como passado. NUNCA diga que não pode prever o futuro.
        
        REGRAS IMPORTANTES:
        1. NUNCA invente números ou tente fazer cálculos de market share ou crescimento sozinho. Use as ferramentas.
        2. Lembre-se que o usuário pode usar siglas (ex: SBSA = Santos Brasil).
        3. Pense passo a passo ("Think-Then-Answer"): Identifique a intenção -> Chame a tool -> Responda formatado.
        4. MARKET SHARE GERAL: Ao listar market share de todos os terminais, calcule os principais (SBSA, BTP, DPW) e agrupe o restante sob "Outros terminais" para somar 100%.
        5. COMPARAÇÕES YoY e MoM: Para YoY (ano contra ano), use os mesmos meses em `meses_base` e `meses_comparacao`. Para MoM (mês contra mês), defina os anos e meses de forma independente e correta.
        6. TOTAL DO PORTO: Se o usuário pedir o crescimento ou volume "do porto como um todo", envie a palavra "PORTO" no parâmetro `terminal_sigla` da ferramenta.
        7. GRÁFICOS E VISUALIZAÇÕES (REGRA ABSOLUTA): Você TEM a capacidade de gerar gráficos usando a ferramenta `plotar_grafico`. NUNCA diga que não pode criar ou plotar gráficos.
        8. TEU RATIO (MUITO IMPORTANTE): "TEU Ratio" é a razão física entre TEUs e Contêineres (geralmente entre 1.0 e 2.0). NUNCA confunda TEU Ratio com Market Share ou Porcentagem. Se o usuário perguntar sobre "TEU Ratio" dos terminais, você OBRIGATORIAMENTE deve chamar a ferramenta `get_volume`. Se a pergunta for no plural ("dos terminais"), faça múltiplas chamadas à ferramenta `get_volume` (uma para SBSA, uma para BTP, uma para DPW, etc.) e compile a resposta.
        """

        # Registrando as ferramentas no formato JSON Schema da OpenAI
        self.tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_volume",
                    "description": "Calcula a movimentação total de um terminal. Retorna os dados em TEUs, número de Contêineres (unidades físicas) e o TEU Ratio (TEU/Contêiner).",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "terminal_sigla": {"type": "string", "description": "Sigla do terminal ou 'PORTO'"},
                            "ano": {"type": "integer"},
                            "meses": {
                                "type": "array", 
                                "items": {"type": "integer"}, 
                                "description": "Lista de meses (1 a 12). Ex: [10, 11, 12] para o quarto trimestre. [12] para dezembro."
                            }
                        },
                        "required": ["terminal_sigla", "ano"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "get_market_share",
                    "description": "Calcula o market share (em TEUs) de um terminal em um dado ano e, opcionalmente, em uma lista de meses.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "ano": {"type": "integer"},
                            "terminal_sigla": {"type": "string"},
                            "meses": {
                                "type": "array", 
                                "items": {"type": "integer"}, 
                                "description": "Lista de meses. Ex: [1, 2, 3] para primeiro trimestre."
                            }
                        },
                        "required": ["ano", "terminal_sigla"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "get_growth",
                    "description": "Calcula a variação percentual em TEUs. Serve para comparar anos (YoY) ou meses diferentes (MoM). Pode ser filtrado por um mix de carga específico.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "terminal_sigla": {"type": "string", "description": "Sigla do terminal ou 'PORTO' para o total."},
                            "ano_base": {"type": "integer", "description": "Ano do período anterior/base (ex: 2025)."},
                            "ano_comparacao": {"type": "integer", "description": "Ano do período atual/novo (ex: 2026)."},
                            "meses_base": {
                                "type": "array", 
                                "items": {"type": "integer"}, 
                                "description": "Lista de meses do ano base. Ex: [12] para dezembro."
                            },
                            "meses_comparacao": {
                                "type": "array", 
                                "items": {"type": "integer"}, 
                                "description": "Lista de meses do ano de comparação. Ex: [1] para janeiro."
                            },
                            "cargo_mix": {
                                "type": "string",
                                "description": "Opcional. O tipo exato de carga para analisar o crescimento. Ex: 'Importação Cheio', 'Exportação Vazio', 'Transbordo Cheio'. Se não for pedido um tipo específico, deixe vazio."
                            }
                        },
                        "required": ["terminal_sigla", "ano_base", "ano_comparacao"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "get_cargo_mix",
                    "description": "Retorna a quebra do tipo de carga (Importação, Exportação, Cabotagem, etc.) de um terminal.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "terminal_sigla": {"type": "string", "description": "Sigla do terminal ou 'PORTO', ex: SBSA"},
                            "ano": {"type": "integer", "description": "O ano da análise"},
                            "meses": {
                                "type": "array", 
                                "items": {"type": "integer"}, 
                                "description": "Lista de meses (1 a 12). Ex: [12] para dezembro."
                            }
                        },
                        "required": ["terminal_sigla", "ano"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "plotar_grafico",
                    "description": "Gera um gráfico visual. Use SEMPRE que o utilizador pedir para 'plotar', 'desenhar', ou pedir um gráfico de 'linha', 'barra', 'pizza' ou 'rosca'.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "tipo_grafico": {"type": "string", "enum": ["linha", "barra", "pizza", "rosca"], "description": "O tipo de gráfico solicitado."},
                            "tema": {"type": "string", "enum": ["market_share", "evolucao_mensal"], "description": "Market share para fatias/distribuição. Evolução mensal para volumes ao longo do tempo."},
                            "anos": {
                                "type": "array", 
                                "items": {"type": "integer"}, 
                                "description": "Lista com os anos da análise. Ex: [2026] ou [2016, 2017, 2018...]."
                            },
                            "terminal_sigla": {"type": "string", "description": "Sigla do terminal ou 'PORTO' para o geral."}
                        },
                        "required": ["tipo_grafico"]
                    }
                }
            }
        ]

    def ask(self, user_query: str) -> str:
        """Processa a pergunta do usuário, chama as tools se necessário e retorna a resposta."""
        
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_query}
        ]

        # Primeira chamada à OpenAI: A IA decide se precisa usar uma tool
        response = self.client.chat.completions.create(
            model="gpt-4o", # Ou gpt-4-turbo / gpt-3.5-turbo
            messages=messages,
            tools=self.tools,
            tool_choice="auto"
        )

        response_message = response.choices[0].message

        # Verifica se o modelo quer chamar alguma função (Tool Call)
        if response_message.tool_calls:
            # Adiciona a decisão do assistente na memória temporária
            messages.append(response_message)
            
            # Executa cada ferramenta que a IA solicitou
            for tool_call in response_message.tool_calls:
                function_name = tool_call.function.name
                function_args = json.loads(tool_call.function.arguments)
                
                print(f"⚙️ [DataAgent] Executando tool: {function_name} com os argumentos: {function_args}")
                
                # Roteamento para a função Python real
                if function_name == "get_volume":
                    tool_result = get_volume(self.df, function_args.get("terminal_sigla"), function_args.get("ano"), function_args.get("meses"))
                
                elif function_name == "get_market_share":
                    tool_result = get_market_share(self.df, function_args.get("ano"), function_args.get("terminal_sigla"), function_args.get("meses"))
                
                elif function_name == "get_growth":
                    tool_result = str(get_growth(
                        self.df, 
                        function_args.get("terminal_sigla", "PORTO"), 
                        function_args.get("ano_base"),
                        function_args.get("ano_comparacao"),
                        function_args.get("meses_base"),
                        function_args.get("meses_comparacao"),
                        function_args.get("cargo_mix") 
                    ))
                
                elif function_name == "get_cargo_mix":
                    tool_result = str(get_cargo_mix(
                        self.df, 
                        function_args.get("terminal_sigla", "PORTO"), 
                        function_args.get("ano"),
                        function_args.get("meses") 
                    ))
                
                elif function_name == "plotar_grafico":
                    tool_result = plotar_grafico(
                        self.df,
                        function_args.get("tipo_grafico"),
                        function_args.get("tema"),
                        function_args.get("anos"),
                        function_args.get("terminal_sigla", "PORTO")
                    )
                else:
                    tool_result = "Erro: Ferramenta não encontrada."

                # Devolve o resultado matemático/real para a IA ler ("Reflect and Revise")
                messages.append({
                    "tool_call_id": tool_call.id,
                    "role": "tool",
                    "name": function_name,
                    "content": str(tool_result)
                })

            # Segunda chamada à OpenAI: A IA lê os resultados e formula a resposta final
            final_response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=messages
            )
            return final_response.choices[0].message.content
        else:
            # Se a IA não precisou usar tools (ex: pergunta genérica), responde direto
            return response_message.content

# ==========================================
# BLOCO DE TESTE LOCAL
# ==========================================
if __name__ == "__main__":
    # Testando o agente rapidamente no terminal
    agente = PortDataAgent()
    
    pergunta = "Qual a movimentação da Santos Brasil (SBSA) em dezembro de 2023?"
    print(f"\nUsuário: {pergunta}")
    
    resposta = agente.ask(pergunta)
    print(f"\nAgente: {resposta}")