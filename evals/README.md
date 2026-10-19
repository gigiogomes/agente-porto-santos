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
