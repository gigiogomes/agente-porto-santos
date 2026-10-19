import json

import pytest

from conftest import make_raw
from ingest import pipeline, sources


@pytest.fixture
def csv_file(tmp_path):
    rows = [{"ano": 2024, "mes": m, "terminal": "BTP", "teu": 10, "ton": "5,5"} for m in range(1, 13)]
    path = tmp_path / "aps.csv"
    make_raw(rows).to_csv(path, index=False)
    return path


def test_gera_arquivos_e_manifesto(tmp_path, csv_file):
    out = tmp_path / "build"
    assert pipeline.run(out, str(csv_file), previous_manifest_url=None) is True

    for name in [sources.FATO_FILE, sources.DIM_TERMINAL_FILE, sources.BRUTO_FILE, sources.MANIFEST_FILE]:
        assert (out / name).exists()
    manifest = json.loads((out / sources.MANIFEST_FILE).read_text(encoding="utf-8"))
    assert manifest["coverage"] == {"min_year": 2024, "max_year": 2024, "max_month": 12}
    assert manifest["yearly_totals"]["2024"]["teu"] == 120
    assert manifest["content_id"] == pipeline.content_id(csv_file.read_bytes())


def test_nao_gera_nada_quando_o_conteudo_nao_mudou(tmp_path, csv_file, monkeypatch):
    publicado = {"content_id": pipeline.content_id(csv_file.read_bytes())}
    monkeypatch.setattr(pipeline, "fetch_previous_manifest", lambda url: publicado)

    out = tmp_path / "build"
    assert pipeline.run(out, str(csv_file), previous_manifest_url="https://exemplo") is False
    assert not out.exists()


def test_contrato_violado_interrompe_sem_gerar_arquivos(tmp_path, csv_file, monkeypatch):
    # A versão publicada ia até 2025; a nova "voltou" para 2024.
    publicado = {"content_id": "outro", "coverage": {"min_year": 2024, "max_year": 2025, "max_month": 3}}
    monkeypatch.setattr(pipeline, "fetch_previous_manifest", lambda url: publicado)

    out = tmp_path / "build"
    with pytest.raises(SystemExit):
        pipeline.run(out, str(csv_file), previous_manifest_url="https://exemplo")
    assert not out.exists()
