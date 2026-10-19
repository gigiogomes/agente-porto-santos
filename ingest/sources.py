"""Endereços das fontes de dados, compartilhados pelo pipeline e pelo app."""

APS_CSV_URL = "https://mensario.portodesantos.com.br/exportarcargas/csv"

# Cópia estática antiga, último recurso caso a APS e a release "dados" falhem.
BACKUP_CSV_URL = "https://github.com/gigiogomes/agente-porto-santos/releases/download/v1.0/exportacao_cargas.csv"

# Release onde o workflow de ingestão publica os arquivos processados.
PUBLISHED_BASE_URL = "https://github.com/gigiogomes/agente-porto-santos/releases/download/dados"

FATO_FILE = "fato_movimentacao.parquet"
DIM_TERMINAL_FILE = "dim_terminal.parquet"
BRUTO_FILE = "bruto.parquet"
MANIFEST_FILE = "manifest.json"
