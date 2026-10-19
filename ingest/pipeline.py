"""
Pipeline de ingestão: CSV da APS -> validação -> Parquet + manifesto.

Uso:
    python -m ingest.pipeline --out build

Só gera arquivos quando o conteúdo mudou em relação à versão publicada. Escreve
`changed=true|false` em $GITHUB_OUTPUT (quando roda no GitHub Actions) e sai com
código 1 se o contrato de dados for violado, sem gerar nada.
"""
import argparse
import hashlib
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from typing import Optional

import pandas as pd
import requests

from ingest import contract, sources
from ingest.transform import SCHEMA_VERSION, TERMINAIS_CSV, build_dim_terminal, build_fato

logger = logging.getLogger(__name__)


def download(url: str, timeout: int = 120, retries: int = 2) -> bytes:
    for attempt in range(retries + 1):
        try:
            response = requests.get(url, timeout=timeout)
            response.raise_for_status()
            return response.content
        except Exception as exc:
            if attempt == retries:
                raise RuntimeError(f"Falha ao baixar {url}: {exc}") from exc
            time.sleep(2.0)


def content_id(source_bytes: bytes) -> str:
    """
    Impressão digital do que gera os arquivos: o CSV da APS, a tabela de
    terminais e a versão da transformação. Mesma impressão = mesmos arquivos.
    """
    digest = hashlib.sha256()
    digest.update(source_bytes)
    digest.update(TERMINAIS_CSV.read_bytes())
    digest.update(SCHEMA_VERSION.encode())
    return digest.hexdigest()


def fetch_previous_manifest(url: str) -> Optional[dict]:
    try:
        response = requests.get(url, timeout=30)
        if response.status_code == 404:
            return None
        response.raise_for_status()
        return response.json()
    except Exception as exc:
        logger.warning("Não foi possível ler o manifesto publicado (%s); seguindo sem ele.", exc)
        return None


def run(out_dir: Path, source_url: str, previous_manifest_url: Optional[str]) -> bool:
    """Executa o pipeline. Retorna True se gerou uma versão nova dos arquivos."""
    logger.info("Lendo %s", source_url)
    # Aceita arquivo local para testes; em produção é sempre a URL da APS.
    source_bytes = Path(source_url).read_bytes() if Path(source_url).is_file() else download(source_url)
    cid = content_id(source_bytes)

    previous = fetch_previous_manifest(previous_manifest_url) if previous_manifest_url else None
    if previous and previous.get("content_id") == cid:
        logger.info("Sem mudanças em relação à versão publicada (%s).", cid[:12])
        return False

    raw = pd.read_csv(BytesIO(source_bytes), low_memory=False)
    fato = build_fato(raw)
    dim_terminal = build_dim_terminal(fato)

    results = contract.validate_fato(fato, previous)
    for r in results:
        log = logger.info if r.ok else (logger.error if r.severidade == "erro" else logger.warning)
        log("[%s] %s: %s", "ok" if r.ok else r.severidade, r.regra, r.detalhe)
    if contract.has_errors(results):
        raise SystemExit("Contrato de dados violado; nada foi publicado.")

    out_dir.mkdir(parents=True, exist_ok=True)
    fato.to_parquet(out_dir / sources.FATO_FILE, compression="zstd", index=False)
    dim_terminal.to_parquet(out_dir / sources.DIM_TERMINAL_FILE, compression="zstd", index=False)
    text_columns = [c for c in raw.columns if not pd.api.types.is_numeric_dtype(raw[c])]
    raw.astype({c: "category" for c in text_columns}).to_parquet(
        out_dir / sources.BRUTO_FILE, compression="zstd", index=False
    )

    manifest = {
        "content_id": cid,
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source_url": source_url,
        "source_sha256": hashlib.sha256(source_bytes).hexdigest(),
        "rows_raw": len(raw),
        "rows_fato": len(fato),
        "coverage": contract.coverage(fato),
        "yearly_totals": contract.yearly_totals(fato),
        "checks": contract.as_dicts(results),
    }
    (out_dir / sources.MANIFEST_FILE).write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Versão nova gerada em %s: %s linhas na tabela fato.", out_dir, f"{len(fato):,}")
    return True


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Ingestão da base de cargas da APS.")
    parser.add_argument("--out", type=Path, default=Path("build"))
    parser.add_argument("--source-url", default=sources.APS_CSV_URL)
    parser.add_argument(
        "--previous-manifest-url",
        default=f"{sources.PUBLISHED_BASE_URL}/{sources.MANIFEST_FILE}",
        help="Manifesto publicado, para comparar. Use '' para ignorar.",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    changed = run(args.out, args.source_url, args.previous_manifest_url or None)

    github_output = os.getenv("GITHUB_OUTPUT")
    if github_output:
        with open(github_output, "a", encoding="utf-8") as fh:
            fh.write(f"changed={'true' if changed else 'false'}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
