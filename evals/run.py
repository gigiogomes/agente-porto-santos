"""
Avaliação do agente com as perguntas de referência (evals/cases.py).

Uso:
    python -m evals.run --agente coordenador --label baseline
    python -m evals.run --agente unico --only share_terminal_2025,ytd_2026

Faz chamadas reais ao LLM (precisa de OPENAI_API_KEY e custa tokens).
Salva o resultado em evals/results/<data>_<label>.json.
"""
import argparse
import json
import logging
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from evals.cases import CASES
from evals.oracle import Oracle
from evals.scoring import avaliar

RESULTS_DIR = Path(__file__).resolve().parent / "results"


class Uso:
    """Conta chamadas e tokens de todas as chamadas ao LLM feitas pelo SDK da OpenAI."""

    def __init__(self):
        self.zerar()

    def zerar(self):
        self.chamadas, self.entrada, self.saida = 0, 0, 0

    def somar(self, usage):
        self.entrada += usage.prompt_tokens
        self.saida += usage.completion_tokens

    def instrumentar(self):
        from openai.resources.chat import Completions

        original = Completions.create
        uso = self

        def create(self_, *args, **kwargs):
            resposta = original(self_, *args, **kwargs)
            uso.chamadas += 1
            if kwargs.get("stream"):
                def repassar():
                    for chunk in resposta:
                        if getattr(chunk, "usage", None):
                            uso.somar(chunk.usage)
                        yield chunk
                return repassar()
            if getattr(resposta, "usage", None):
                uso.somar(resposta.usage)
            return resposta

        Completions.create = create


def _capturar_chamadas(modulos, chamadas: list):
    """Envolve a ferramenta de dados para registrar os argumentos de cada chamada."""
    for modulo in modulos:
        for nome in ("query_port_data", "run_query"):
            funcao = getattr(modulo, nome, None)
            if funcao is None:
                continue

            def envolvida(store, _funcao=funcao, **kwargs):
                chamadas.append(kwargs)
                return _funcao(store, **kwargs)

            setattr(modulo, nome, envolvida)


def montar_coordenador(chamadas: list):
    """Arquitetura até a Fase 1: coordenador + analista de dados + agente de conhecimento."""
    from agent import coordinator_agent, data_agent

    agente = coordinator_agent.CoordinatorAgent()
    _capturar_chamadas([data_agent], chamadas)

    def responder(turnos, antes_do_ultimo):
        historico, filtros = [], None
        for i, turno in enumerate(turnos):
            if i == len(turnos) - 1:
                antes_do_ultimo()
            texto, historico, filtros = agente.chat(turno, history=historico, data_filters=filtros)
        return texto, []  # esta arquitetura não exibe tabelas além do texto

    return responder, agente.data_agent.store


AGENTES = {"coordenador": montar_coordenador}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Avaliação do agente do Porto de Santos.")
    parser.add_argument("--agente", choices=sorted(AGENTES), required=True)
    parser.add_argument("--label", default="")
    parser.add_argument("--only", default="", help="ids separados por vírgula")
    args = parser.parse_args(argv)

    sys.stdout.reconfigure(encoding="utf-8")
    logging.basicConfig(level=logging.WARNING)
    logging.disable(logging.WARNING)

    uso = Uso()
    uso.instrumentar()
    chamadas: list = []
    responder, store = AGENTES[args.agente](chamadas)
    oracle = Oracle(store)

    casos = [c for c in CASES if not args.only or c.id in args.only.split(",")]
    resultados = []
    for case in casos:
        inicio = {}

        def antes_do_ultimo():
            chamadas.clear()
            uso.zerar()
            inicio["t"] = time.time()

        try:
            texto, tabelas = responder(case.turnos, antes_do_ultimo)
        except Exception as exc:  # um caso com erro não interrompe a avaliação
            texto, tabelas = f"ERRO: {exc!r}", []
        segundos = time.time() - inicio.get("t", time.time())

        r = avaliar(case, oracle, texto, tabelas, list(chamadas))
        r.update(segundos=round(segundos, 2), chamadas_llm=uso.chamadas, tokens_entrada=uso.entrada, tokens_saida=uso.saida)
        resultados.append(r)
        status = "OK   " if r["ok"] else "FALHA"
        print(f"{status} {case.id:34} {segundos:5.1f}s  {uso.chamadas} chamadas  {uso.entrada:6,} tokens")
        for falha in r["falhas"]:
            print(f"        - {falha[:300]}")

    n = len(resultados)
    por_categoria = defaultdict(list)
    for r in resultados:
        por_categoria[r["categoria"]].append(r["ok"])
    resumo = {
        "agente": args.agente,
        "label": args.label,
        "data": datetime.now().isoformat(timespec="seconds"),
        "casos": n,
        "acertos": sum(r["ok"] for r in resultados),
        "por_categoria": {k: f"{sum(v)}/{len(v)}" for k, v in por_categoria.items()},
        "formato_ptbr": sum(r["formato_ptbr"] for r in resultados),
        "segundos_medio": round(sum(r["segundos"] for r in resultados) / n, 2),
        "chamadas_llm_medio": round(sum(r["chamadas_llm"] for r in resultados) / n, 2),
        "tokens_entrada_medio": round(sum(r["tokens_entrada"] for r in resultados) / n),
        "tokens_saida_medio": round(sum(r["tokens_saida"] for r in resultados) / n),
    }
    print("\n" + json.dumps(resumo, ensure_ascii=False, indent=2))

    RESULTS_DIR.mkdir(exist_ok=True)
    nome = f"{datetime.now():%Y%m%d-%H%M}_{args.label or args.agente}.json"
    (RESULTS_DIR / nome).write_text(
        json.dumps({"resumo": resumo, "resultados": resultados}, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Resultado salvo em evals/results/{nome}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
