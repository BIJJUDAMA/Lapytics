import hashlib
from pathlib import Path
from src.config import NOTEBOOK_HASHES, ALL_FEATURE_COLUMNS, BASE_DIR, NOTEBOOKS_DIR

def test_notebook_hashes_unchanged():
    for nb_name, expected_hash in NOTEBOOK_HASHES.items():
        nb_path = NOTEBOOKS_DIR / nb_name if (NOTEBOOKS_DIR / nb_name).exists() else (BASE_DIR / nb_name)
        assert nb_path.exists(), f"Notebook {nb_name} does not exist in {NOTEBOOKS_DIR}"
        content = nb_path.read_bytes()
        actual_hash = hashlib.sha256(content).hexdigest().upper()
        assert actual_hash == expected_hash, f"Hash mismatch for {nb_name}: {actual_hash} != {expected_hash}"

def test_feature_columns_count():
    assert len(ALL_FEATURE_COLUMNS) == 42
    assert "inches" in ALL_FEATURE_COLUMNS
    assert "ppi" in ALL_FEATURE_COLUMNS
    assert "ram" in ALL_FEATURE_COLUMNS
    assert "price" not in ALL_FEATURE_COLUMNS
    assert "price_category" not in ALL_FEATURE_COLUMNS
