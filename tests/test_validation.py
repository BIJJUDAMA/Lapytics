import pandas as pd
import pytest
from src.validation import validate_laptop_input, build_feature_vector
from src.config import ALL_FEATURE_COLUMNS

def test_valid_laptop_input():
    sample = {
        "company": "Dell",
        "typename": "Notebook",
        "opsys": "Windows 10",
        "ram": 16.0,
        "weight": 2.1,
        "inches": 15.6,
        "res_width": 1920,
        "res_height": 1080,
        "storage_gb": 512.0,
        "has_ssd": 1,
        "has_hdd": 0,
        "is_ips": 1,
        "is_touchscreen": 0
    }
    is_valid, errors = validate_laptop_input(sample)
    assert is_valid
    assert len(errors) == 0

    vec = build_feature_vector(sample)
    assert isinstance(vec, pd.DataFrame)
    assert vec.shape == (1, 42)
    assert list(vec.columns) == ALL_FEATURE_COLUMNS
    assert vec.loc[0, "company_Dell"] == 1.0
    assert vec.loc[0, "company_Hp"] == 0.0
    assert vec.loc[0, "ram"] == 16.0
    expected_ppi = (1920**2 + 1080**2)**0.5 / 15.6
    assert abs(vec.loc[0, "ppi"] - expected_ppi) < 1e-4

def test_invalid_laptop_input_bounds():
    sample = {
        "company": "NonExistentBrand",
        "typename": "Notebook",
        "opsys": "Windows 10",
        "ram": -4.0,
        "weight": 2.1,
        "inches": 15.6,
        "res_width": 1920,
        "res_height": 1080,
        "storage_gb": 512.0,
        "has_ssd": 1,
        "has_hdd": 0,
        "is_ips": 1,
        "is_touchscreen": 0
    }
    is_valid, errors = validate_laptop_input(sample)
    assert not is_valid
    assert any("RAM" in err for err in errors)
    assert any("company" in err for err in errors)

    with pytest.raises(ValueError):
        build_feature_vector(sample)

def test_storage_validation():
    # Neither SSD nor HDD selected should fail
    sample = {
        "company": "Dell",
        "typename": "Notebook",
        "opsys": "Windows 10",
        "ram": 8.0,
        "weight": 2.0,
        "inches": 15.6,
        "res_width": 1920,
        "res_height": 1080,
        "storage_gb": 256.0,
        "has_ssd": 0,
        "has_hdd": 0,
        "is_ips": 0,
        "is_touchscreen": 0
    }
    is_valid, errors = validate_laptop_input(sample)
    assert not is_valid
    assert any("storage" in err.lower() for err in errors)
