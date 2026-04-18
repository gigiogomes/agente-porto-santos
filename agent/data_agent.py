import json
import os
import sys
from typing import Any, Dict, Optional

from dotenv import load_dotenv

# ROTAS
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Pega a pasta raiz do projeto (ex: port_agent_g)
ROOT_DIR = os.path.dirname(BASE_DIR)

# Injeta ambas as pastas no "radar" do Python
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

# Agora o Python consegue achar o arquivo, não importa se ele está na pasta tools ou solto
try:
    from tools.port_data_tools import load_port_data, query_port_data
except ImportError:
    from port_data_tools import load_port_data, query_port_data
# --------------------------------------------

load_dotenv()

class PortDataAgent:
    def __init__(self, chart_output_dir: Optional[str] = None):
        self.chart_output_dir = chart_output_dir
        self.last_generated_chart_path: Optional[str] = None
        
        # MEMÓRIA DE CURTO PRAZO: Mantém o contexto fluido sem lotar os tokens
        self.current_filters = {
            "start_date": None,
            "end_date": None,
            "terminals": [],
            "metric": "teus"
        }

        print("Iniciando Agente Analista de Dados...")
        self.df, self.data_coverage, self.data_diagnostics = load_port_data()

        api_key = os.getenv("OPENAI_API_KEY")
        self.client = None
        self.model_name = os.getenv("PORT_DATA_AGENT_MODEL", "gpt-4o")
        if api_key:
            try:
                from openai import OpenAI
                self.client = OpenAI(api_key=api_key)
            except Exception:
                self.client = None

        self.tools = [
            {
                "type": "function",
                "function": {
                    "name": "query_port_data",
                    "description": "Faz qualquer consulta de dados, agregação ou comparação de períodos no formato YYYY-MM-DD. Use para volumes, market share, crescimento, cargo mix.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "metric": {
                                "type": "string", 
                                "enum": ["teus", "conteineres", "market_share", "cagr"],
                                "description": "Qual métrica extrair"
                            },
                            "start_date": {
                                "type": "string",
                                "description": "Data de início em YYYY-MM-DD. Ex: 2021-01-01"
                            },
                            "end_date": {
                                "type": "string",
                                "description": "Data final em YYYY-MM-DD. Ex: 2023-12-31"
                            },
                            "terminals": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Lista de terminais. Deixe vazio para o porto inteiro."
                            },
                            "cargo_mix_filter": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Filtra por tipos específicos de Cargo Mix (ex: ['REMOÇÃO', 'EXPORTAÇÃO'])."
                            },
                            "group_by": {
                                "type": "array",
                                "items": {
                                    "type": "string",
                                    "enum": ["ano", "semestre", "quadrimestre", "trimestre", "bimestre", "mes", "terminal", "cargo_mix"],
                                },
                                "description": "Como agrupar os dados. Pode cruzar informações passando mais de um (ex: ['mes', 'terminal'])."
                            },
                            "compare_with_previous": {
                                "type": "boolean",
                                "description": "True se o usuário pedir crescimento (YoY/MoM/Crescimento) para calcular a %"
                            }
                        },
                        "required": ["metric"]
                    }
                }
            }
        ]

    def _update_context_from_args(self, args: dict):
        """Atualiza a memória baseada na última ferramenta chamada"""
        if args.get("start_date"): self.current_filters["start_date"] = args["start_date"]
        if args.get("end_date"): self.current_filters["end_date"] = args["end_date"]
        if args.get("terminals"): self.current_filters["terminals"] = args["terminals"]
        if args.get("metric"): self.current_filters["metric"] = args["metric"]

    def get_status_summary(self) -> str:
        """Retorna o resumo da saúde dos dados para a interface do Streamlit"""
        if not hasattr(self, 'data_coverage') or not self.data_coverage:
            return "Nenhum dado carregado."
        
        cov = self.data_coverage
        return (
            f"✅ **Base Ativa**\n"
            f"- Período: {cov.min_year} a {cov.max_year}-{cov.max_month:02d}\n"
            f"- Registros: {cov.row_count:,}\n"
            f"- Atualizado em: {cov.loaded_at}"
        )

    def ask(self, question: str, context: str = "") -> str:
        if not self.client:
            return "Erro: cliente OpenAI não configurado para o Agente de Dados."

        # Injete as variáveis de contexto ao invés de mensagens inteiras
        max_y = self.data_coverage.max_year
        max_m = self.data_coverage.max_month
        mix_list = ", ".join([str(x) for x in self.data_coverage.cargo_mix_list]) if self.data_coverage.cargo_mix_list else "Não identificados"

        system_prompt = f"""
        Você é o Analista de Dados do Porto.
        
        [COBERTURA DE DADOS]
        Você tem dados carregados até: {max_y}-{max_m:02d} (Este é o "mês atual" ou "último mês disponível").
        Cargo Mix disponíveis nesta base: {mix_list}
        
        [REGRAS DE TEMPO RELATIVO - EXTREMAMENTE IMPORTANTE]
        1. Omissão de data: Se o usuário pedir algo genérico (ex: "Qual o volume da BTP?"), assuma automaticamente o mês mais recente ({max_y}-{max_m:02d}) configurando `start_date` e `end_date` para este mês.
        2. "Último mês": Use `start_date` e `end_date` cobrindo o mês {max_y}-{max_m:02d}.
        3. "Último ano": Avalie o contexto. Pode ser o ano cheio de {max_y} ou, se estivermos no começo do ano, o ano consolidado de {max_y - 1}.
        
        [REGRAS DE COMPARAÇÃO - YoY / MoM]
        Para calcular o crescimento ou comparar períodos, você DEVE buscar os dois períodos na ferramenta definindo o `start_date` e `end_date` de forma abrangente e ativando `compare_with_previous=True`.
        - Exemplo YoY (Year-over-Year): Para comparar o último mês com o mesmo mês do ano anterior, defina start_date='{max_y-1}-{max_m:02d}-01', end_date='{max_y}-{max_m:02d}-31' e group_by='mes' ou 'ano'.
        - Não tente fazer a matemática sozinho; deixe a ferramenta agregar agrupando pelo período.
        
        [CONTEXTO DA CONVERSA ATUAL]
        Data Inicial foco: {self.current_filters['start_date'] or 'Não definida'}
        Data Final foco: {self.current_filters['end_date'] or 'Não definida'}
        Terminais: {', '.join(self.current_filters['terminals']) if self.current_filters['terminals'] else 'Todos'}
        Métrica Ativa: {self.current_filters['metric']}

        [REGRAS DE CARGO MIX - OBRIGATÓRIO]
        1. A coluna de Cargo Mix chama-se "ESTADO". Os valores válidos exatos são: {mix_list}.
        2. Se o usuário perguntar "quais os cargo mix", cite a lista acima diretamente sem usar ferramentas.
        3. Se pedir sobre "remoção" ou "exportação", você DEVE acionar a ferramenta usando a propriedade `cargo_mix_filter` com o valor exato em MAIÚSCULO. Ex: `cargo_mix_filter=["REMOÇÃO"]`.
        4. Se perguntar "Qual o share por cargo mix" ou "proporção", você DEVE acionar a ferramenta passando `group_by="cargo_mix"`.
        5. Se o usuário pedir "mix de carga" de um terminal, responda os volumes detalhados usando group_by=["cargo_mix"].
        6. CRUZAMENTOS: Se o usuário pedir "por terminal mês a mês", use group_by=["mes", "terminal"] juntos.
        
        [REGRAS DE TERMINAIS - OBRIGATÓRIO]
        O usuário pode digitar o nome completo, mas você DEVE converter para a SIGLA OFICIAL ao usar o filtro 'terminals' da ferramenta:
        - "Santos Brasil" -> "SBSA"
        - "DP World" ou "Embraport" -> "DPW"
        - "Brasil Terminal Portuário" -> "BTP"
        - "Ecoporto" -> "ECOPORTO"
        Se você enviar o nome por extenso (ex: "Santos Brasil") a ferramenta falhará.

        [REGRAS DE MÉTRICAS E SHARE]
        1. Se o usuário usar as palavras "SHARE", "PARTICIPAÇÃO", "PROPORÇÃO" ou "MIX", você DEVE obrigatoriamente usar metric="market_share".
        2. O resultado voltará com uma coluna 'Share (%)'. Você deve priorizar a exibição deste valor percentual na sua resposta final.
        3. Sempre que mostrar o Share, mencione também o volume absoluto (TEUs) logo ao lado para contexto.
        4. Para "Cargo Mix", use sempre group_by="cargo_mix" e metric="market_share".
        5. CÁLCULO DE YOY (Year-over-Year): Se o usuário pedir "Crescimento YoY por terminal" para um ano específico (ex: 2026), defina compare_with_previous=True, start_date="2025-01-01" (ano anterior) e use group_by=["terminal", "ano"]. A ferramenta já ajustará os meses automaticamente para uma comparação justa.
"""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question}
        ]

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            tools=self.tools,
            temperature=0.0
        )
        response_message = response.choices[0].message

        if not response_message.tool_calls:
            return response_message.content or ""

        messages.append(response_message)
        for tool_call in response_message.tool_calls:
            function_name = tool_call.function.name
            raw_args = json.loads(tool_call.function.arguments)

            print(f"\n📊 [DATA AGENT] Executando Pandas: {function_name}")
            print(f"⚙️ [FILTROS] Argumentos: {json.dumps(raw_args, indent=2, ensure_ascii=False)}")
            
            # Atualiza a memória para futuras interações!
            self._update_context_from_args(raw_args)

            try:
                if function_name == "query_port_data":
                    tool_result = query_port_data(self.df, **raw_args)
                else:
                    tool_result = f"Ferramenta {function_name} não encontrada."
            except Exception as exc:
                tool_result = f"Erro ao executar a consulta: {exc}"

            messages.append({
                "tool_call_id": tool_call.id,
                "role": "tool",
                "name": function_name,
                "content": str(tool_result),
            })

        final_response = self.client.chat.completions.create(model=self.model_name, messages=messages)
        return final_response.choices[0].message.content or ""

    def clear_last_generated_chart(self):
        self.last_generated_chart_path = None