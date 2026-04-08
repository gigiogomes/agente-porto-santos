import json
import os
import sys
import re

from dotenv import load_dotenv
from openai import OpenAI

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.port_data_tools import (
    load_port_data,
    get_market_share,
    get_cargo_mix,
    get_growth,
    get_volume,
    plotar_grafico,
    get_data_coverage_text
)

load_dotenv()


class PortDataAgent:
    def __init__(self):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        print("Iniciando Agente Analista de Dados...")

        self.df, self.data_coverage, self.data_diagnostics = load_port_data()

        self.max_year = self.data_coverage.max_year
        self.latest_months = sorted(
            self.df.loc[self.df["ANO"] == self.max_year, "MES"].dropna().unique().tolist()
        )
        self.latest_complete_year = self._get_latest_complete_year()

        meses_str = str(self.latest_months)

        self.system_prompt = f"""
Você é o Agente Especialista em Dados do Porto de Santos.
Seu objetivo é responder perguntas precisas sobre movimentação de carga, volumes (TEUs), unidades, market share, crescimento e mix dos terminais.

CONTEXTO DA BASE ATIVA:
- Fonte ativa: {self.data_coverage.source_name}
- Cobertura temporal: até {self.data_coverage.max_month:02d}/{self.data_coverage.max_year}
- Meses disponíveis no ano mais recente ({self.max_year}): {meses_str}

REGRAS IMPORTANTES:
1. NUNCA invente números. Sempre use as ferramentas.
2. O usuário pode usar siglas como SBSA, BTP e DPW.
3. Sempre explicite o período analisado e o filtro aplicado.
4. Se o usuário perguntar sobre cobertura, atualização da base, origem da base, data mais recente ou status dos dados, use a ferramenta get_data_coverage.
5. Se houver contexto recente fornecido, use esse contexto para resolver follow-ups.
6. Se o usuário pedir TEU Ratio, use get_volume.
7. Se o usuário pedir gráficos, use plotar_grafico.
8. Se o usuário pedir análise do porto como um todo, use "PORTO" como terminal_sigla.
9. Se o usuário pedir uma comparação, organize a resposta de forma clara.
10. SE O USUÁRIO NÃO INFORMAR O ANO em perguntas quantitativas, NÃO escolha um ano arbitrariamente.
11. Se o ano mais recente da base estiver incompleto, use o período mais recente disponível e explicite que a resposta é parcial/YTD.
12. Só use um ano fechado anterior se o usuário pedir explicitamente "ano fechado", "ano completo" ou citar o ano desejado.
"""

        self.tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_volume",
                    "description": "Calcula a movimentação total de um terminal. Retorna TEUs, contêineres (unidades físicas) e TEU Ratio.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "terminal_sigla": {
                                "type": "string",
                                "description": "Sigla do terminal ou 'PORTO'."
                            },
                            "ano": {
                                "type": "integer",
                                "description": "Ano da análise."
                            },
                            "meses": {
                                "type": "array",
                                "items": {"type": "integer"},
                                "description": "Lista de meses, por exemplo [1, 2, 3]."
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
                    "description": "Calcula o market share em TEUs de um terminal em um ano e, opcionalmente, em meses específicos.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "ano": {
                                "type": "integer",
                                "description": "Ano da análise."
                            },
                            "terminal_sigla": {
                                "type": "string",
                                "description": "Sigla do terminal."
                            },
                            "meses": {
                                "type": "array",
                                "items": {"type": "integer"},
                                "description": "Lista de meses."
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
                    "description": "Calcula a variação percentual em TEUs entre dois períodos. Suporta comparação anual, mensal ou por mix de carga.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "terminal_sigla": {
                                "type": "string",
                                "description": "Sigla do terminal ou 'PORTO'."
                            },
                            "ano_base": {
                                "type": "integer",
                                "description": "Ano do período base."
                            },
                            "ano_comparacao": {
                                "type": "integer",
                                "description": "Ano do período de comparação."
                            },
                            "meses_base": {
                                "type": "array",
                                "items": {"type": "integer"},
                                "description": "Meses do período base."
                            },
                            "meses_comparacao": {
                                "type": "array",
                                "items": {"type": "integer"},
                                "description": "Meses do período de comparação."
                            },
                            "cargo_mix": {
                                "type": "string",
                                "description": "Opcional. Ex: 'Importação Cheio', 'Exportação Vazio', 'Cabotagem Cheio'."
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
                    "description": "Retorna a quebra do tipo de carga de um terminal.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "terminal_sigla": {
                                "type": "string",
                                "description": "Sigla do terminal ou 'PORTO'."
                            },
                            "ano": {
                                "type": "integer",
                                "description": "Ano da análise."
                            },
                            "meses": {
                                "type": "array",
                                "items": {"type": "integer"},
                                "description": "Lista de meses."
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
                    "description": "Gera um gráfico visual. Use quando o usuário pedir gráfico, plot, linha, barra, pizza ou rosca.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "tipo_grafico": {
                                "type": "string",
                                "enum": ["linha", "barra", "pizza", "rosca"],
                                "description": "Tipo do gráfico."
                            },
                            "tema": {
                                "type": "string",
                                "enum": ["market_share", "evolucao_mensal"],
                                "description": "Tema do gráfico."
                            },
                            "anos": {
                                "type": "array",
                                "items": {"type": "integer"},
                                "description": "Lista de anos da análise."
                            },
                            "terminal_sigla": {
                                "type": "string",
                                "description": "Sigla do terminal ou 'PORTO'."
                            }
                        },
                        "required": ["tipo_grafico"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "get_data_coverage",
                    "description": "Retorna a fonte ativa, a cobertura temporal da base e o diagnóstico do carregamento.",
                    "parameters": {
                        "type": "object",
                        "properties": {}
                    }
                }
            }
        ]

    def _get_latest_complete_year(self):
        meses_por_ano = self.df.groupby("ANO")["MES"].nunique()
        anos_completos = meses_por_ano[meses_por_ano >= 12]
        if anos_completos.empty:
            return None
        return int(anos_completos.index.max())

    def _has_explicit_year(self, text: str) -> bool:
        return bool(re.search(r"\b20\d{2}\b", text))

    def _looks_like_quant_question(self, text: str) -> bool:
        text = text.lower()
        keywords = [
            "market share", "moviment", "teu", "volume", "crescimento",
            "mix", "terminal", "porto", "carga", "dpw", "sbsa", "btp"
        ]
        return any(k in text for k in keywords)

    def _augment_query_with_default_period(self, user_query: str) -> str:
        if self._has_explicit_year(user_query):
            return user_query

        if not self._looks_like_quant_question(user_query):
            return user_query

        if len(self.latest_months) < 12:
            return (
                user_query
                + f"\n\nREGRA DE PERÍODO PADRÃO: o usuário não informou o ano. "
                  f"Use automaticamente o período mais recente disponível na base: "
                  f"{self.max_year}, meses {self.latest_months}. "
                  f"Na resposta final, deixe explícito que se trata de período parcial/YTD."
            )

        return (
            user_query
            + f"\n\nREGRA DE PERÍODO PADRÃO: o usuário não informou o ano. "
              f"Use automaticamente o ano mais recente completo da base: {self.max_year}."
        )

    def get_status_summary(self) -> str:
        return get_data_coverage_text(self.data_coverage, self.data_diagnostics)

    def ask(self, user_query: str, context: str = "") -> str:
        effective_query = self._augment_query_with_default_period(user_query)

        messages = [
            {"role": "system", "content": self.system_prompt}
        ]

        if context:
            messages.append({
                "role": "system",
                "content": f"Contexto recente da conversa:\n{context}"
            })

        messages.append({"role": "user", "content": effective_query})

        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            tools=self.tools,
            tool_choice="auto"
        )

        response_message = response.choices[0].message

        if response_message.tool_calls:
            messages.append(response_message)

            for tool_call in response_message.tool_calls:
                function_name = tool_call.function.name
                function_args = json.loads(tool_call.function.arguments or "{}")

                print(
                    f"⚙️ [DataAgent] Executando tool: {function_name} "
                    f"com argumentos: {function_args}"
                )

                if function_name == "get_volume":
                    tool_result = get_volume(
                        self.df,
                        function_args.get("terminal_sigla"),
                        function_args.get("ano"),
                        function_args.get("meses")
                    )

                elif function_name == "get_market_share":
                    tool_result = get_market_share(
                        self.df,
                        function_args.get("ano"),
                        function_args.get("terminal_sigla"),
                        function_args.get("meses")
                    )

                elif function_name == "get_growth":
                    tool_result = get_growth(
                        self.df,
                        function_args.get("terminal_sigla", "PORTO"),
                        function_args.get("ano_base"),
                        function_args.get("ano_comparacao"),
                        function_args.get("meses_base"),
                        function_args.get("meses_comparacao"),
                        function_args.get("cargo_mix")
                    )

                elif function_name == "get_cargo_mix":
                    tool_result = get_cargo_mix(
                        self.df,
                        function_args.get("terminal_sigla", "PORTO"),
                        function_args.get("ano"),
                        function_args.get("meses")
                    )

                elif function_name == "plotar_grafico":
                    tool_result = plotar_grafico(
                        self.df,
                        function_args.get("tipo_grafico"),
                        function_args.get("tema"),
                        function_args.get("anos"),
                        function_args.get("terminal_sigla", "PORTO")
                    )

                elif function_name == "get_data_coverage":
                    tool_result = get_data_coverage_text(
                        self.data_coverage,
                        self.data_diagnostics
                    )

                else:
                    tool_result = "Erro: ferramenta não encontrada."

                messages.append({
                    "tool_call_id": tool_call.id,
                    "role": "tool",
                    "name": function_name,
                    "content": str(tool_result)
                })

            final_response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=messages
            )

            return final_response.choices[0].message.content or ""

        return response_message.content or ""


if __name__ == "__main__":
    agente = PortDataAgent()

    pergunta = "Qual a movimentação da SBSA em dezembro de 2023?"
    print(f"\nUsuário: {pergunta}")

    resposta = agente.ask(pergunta)
    print(f"\nAgente: {resposta}")