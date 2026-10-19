"""Pontuação de uma resposta: números, chamadas de ferramenta, menções e formato."""
import json
import re
from typing import Callable, Iterable, List, Set

from ingest.transform import canonical_terminal, normalize_text

_ESCALAS = {"BILHAO": 1e9, "BILHOES": 1e9, "MILHAO": 1e6, "MILHOES": 1e6, "MI": 1e6, "MIL": 1e3}
_NUMERO = re.compile(
    r"(?<![\w])(-?\d{1,3}(?:[.,]\d{3})+(?:[.,]\d+)?|-?\d+(?:[.,]\d+)?)"
    r"(?:\s*(BILHOES|BILHAO|MILHOES|MILHAO|MIL|MI)\b)?"
)
# "2,595,312": separador de milhar americano, fora do padrão brasileiro.
_MILHAR_EN = re.compile(r"(?<![\d.,])\d{1,3}(?:,\d{3})+(?![\d,]|\.\d)")


def _interpretacoes(token: str) -> Set[float]:
    """'2.595' pode ser 2595 (pt-BR) ou 2,595 (en): considera as duas leituras."""
    leituras = {token.replace(".", "").replace(",", "."), token.replace(",", "")}
    if "," not in token and "." not in token:
        leituras = {token}
    valores = set()
    for leitura in leituras:
        try:
            valores.add(float(leitura))
        except ValueError:
            pass
    return valores


def extrair_numeros(texto: str) -> List[float]:
    numeros = []
    for match in _NUMERO.finditer(normalize_text(texto)):
        escala = _ESCALAS.get(match.group(2) or "", 1.0)
        numeros += [v * escala for v in _interpretacoes(match.group(1))]
    return numeros


def tem_valor(numeros: Iterable[float], esperado: float, tolerancia_rel: float = 0.01) -> bool:
    return any(abs(n - esperado) <= abs(esperado) * tolerancia_rel for n in numeros)


def tem_percentual(numeros: Iterable[float], esperado: float, tolerancia: float = 0.06) -> bool:
    # Compara o módulo: "queda de 10,25%" e "-10,25%" dizem a mesma coisa.
    return any(abs(abs(n) - abs(esperado)) <= tolerancia for n in numeros)


def formato_brasileiro(texto: str) -> bool:
    return not _MILHAR_EN.search(texto)


# ------------------------------------------------------------------ #
# Predicados sobre os argumentos das chamadas da ferramenta de dados
# ------------------------------------------------------------------ #
def _lista(call: dict, chave: str) -> List[str]:
    valor = call.get(chave) or []
    return [normalize_text(v) for v in ([valor] if isinstance(valor, str) else valor)]


def metrica(*nomes) -> tuple:
    return (f"metric em {nomes}", lambda c: c.get("metric") in nomes)


def agrupa(dim: str) -> tuple:
    return (f"group_by com {dim}", lambda c: normalize_text(dim) in _lista(c, "group_by"))


def compara() -> tuple:
    return ("compare_with_previous=True", lambda c: c.get("compare_with_previous") is True)


def terminal(nome: str) -> tuple:
    return (f"terminal {nome}", lambda c: nome in [canonical_terminal(t) for t in (c.get("terminals") or [])])


def sem_filtro_de_terminal() -> tuple:
    return ("sem filtro de terminal", lambda c: not c.get("terminals"))


def mercadoria(trecho: str) -> tuple:
    return (f"mercadoria com '{trecho}'", lambda c: any(trecho in m for m in _lista(c, "mercadorias")))


def sentido(valor: str) -> tuple:
    return (f"sentido {valor}", lambda c: valor in _lista(c, "sentido"))


def navegacao(valor: str) -> tuple:
    return (f"navegacao {valor}", lambda c: valor in _lista(c, "navegacao"))


def toneladas() -> tuple:
    return ("medida em toneladas", lambda c: "toneladas" in (c.get("metric"), c.get("medida")))


def cargo_mix(prefixo: str) -> tuple:
    return (f"cargo_mix {prefixo}", lambda c: any(v.startswith(prefixo) for v in _lista(c, "cargo_mix_filter")))


def avaliar(case, oracle, texto: str, tabelas: List[str], chamadas: List[dict]) -> dict:
    falhas = []
    esperado = case.esperado(oracle) if case.esperado else {}
    corpus = texto if case.onde == "texto" else "\n".join([texto, *tabelas])
    numeros = extrair_numeros(corpus)

    for valor in esperado.get("valores", []):
        if not tem_valor(numeros, valor):
            falhas.append(f"valor esperado ausente: {valor:,.0f}")
    for pct in esperado.get("percentuais", []):
        if not tem_percentual(numeros, pct):
            falhas.append(f"percentual esperado ausente: {pct}%")

    if case.args:
        if not any(all(pred(c) for _, pred in case.args) for c in chamadas):
            descricao = ", ".join(desc for desc, _ in case.args)
            falhas.append(f"nenhuma chamada com [{descricao}]; chamadas: {json.dumps(chamadas, ensure_ascii=False)}")

    texto_norm = normalize_text(texto)
    for termo in case.mencoes:
        if normalize_text(termo) not in texto_norm:
            falhas.append(f"não menciona '{termo}'")
    grupos = case.alguma_mencao(oracle) if callable(case.alguma_mencao) else case.alguma_mencao
    for grupo in grupos:
        if not any(normalize_text(t) in texto_norm for t in grupo):
            falhas.append(f"não menciona nenhum de {grupo}")

    if case.sem_numeros_grandes:
        grandes = [m.group(0) for m in _NUMERO.finditer(texto_norm) if len(re.sub(r"\D", "", m.group(1))) >= 6]
        if grandes:
            falhas.append(f"inventou números: {grandes}")

    return {
        "id": case.id,
        "categoria": case.categoria,
        "ok": not falhas,
        "falhas": falhas,
        "formato_ptbr": formato_brasileiro(texto),
        "resposta": texto,
        "chamadas": chamadas,
    }
