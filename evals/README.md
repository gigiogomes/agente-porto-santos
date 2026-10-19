# Avaliação do agente

40 perguntas de referência com respostas calculadas direto nos dados
(`oracle.py`). Para cada pergunta, a avaliação confere os números da resposta,
os filtros usados na consulta e menções obrigatórias (ex.: avisar que um ano
está incompleto).

```bash
python -m evals.run --agente unico --label minha-mudanca
python -m evals.run --agente unico --only ytd_2026,yoy_julho
```

Faz chamadas reais ao LLM: precisa de `OPENAI_API_KEY` e custa tokens. Rode
antes de cada merge e sempre que mudar prompt, ferramentas ou modelo. Os
resultados detalhados ficam em `evals/results/` (fora do git).

## Histórico

| Data | Versão | Acertos | Tempo médio | Chamadas ao LLM | Tokens de entrada | Formato pt-BR |
|---|---|---|---|---|---|---|
| 2026-10-19 | Coordenador + analista (até a Fase 1) | 34/40 | 3,4 s | 3,9 | 7.016 | 35/40 |
| 2026-10-19 | Agente único, 1ª versão | 33/40 | 2,1 s | 2,0 | 5.694 | 40/40 |
| 2026-10-19 | Agente único: TEU como medida padrão, mercadorias por trecho do nome, totais sem quebra | 40/40 | 2,0 s | 2,0 | 5.785 | 40/40 |

## Comparação de modelos (2026-10-19)

Mesmo agente e mesmas 40 perguntas; só muda `PORT_AGENT_MODEL`.

| Modelo | Acertos | Tempo médio | Tokens de saída | Observação |
|---|---|---|---|---|
| gpt-4o | 40/40 | 2,1 s | 112 | padrão mantido; 40/40 em duas rodadas |
| gpt-5.5 | 40/40 | 3,5 s | 213 | mesmo acerto, 70% mais lento |
| gpt-5.4-mini | 38/40 | 2,0 s | 161 | erra share de um terminal e medida do cargo mix |
| gpt-4.1-mini | 33/40 | 2,2 s | 117 | |
| gpt-4o-mini | 26/40 | 2,4 s | 160 | |
| gpt-4.1 | 25/40 | 1,7 s | 111 | responde números sem consultar os dados |

Dois modelos com 40/40 indicam que a suíte saturou: para diferenciar os
melhores, ela precisa de perguntas mais difíceis.

```bash
PORT_AGENT_MODEL=gpt-5.5 python -m evals.run --agente unico --label modelo-gpt-5.5
```
