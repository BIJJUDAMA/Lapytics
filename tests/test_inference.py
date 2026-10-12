import pytest
from src.model_registry import get_available_models, load_model
from src.inference import predict_price, predict_price_category, format_inr

def test_available_models():
    reg_models = get_available_models("regression")
    assert len(reg_models) == 10
    assert "Gradient Boosting Regressor (Tuned)" in reg_models

    clf_models = get_available_models("classification")
    assert len(clf_models) == 10
    assert "SVC" in clf_models
    assert "Random Forest" in clf_models

def test_predict_price_regression():
    sample = {
        "company": "Hp",
        "typename": "Notebook",
        "opsys": "Windows 10",
        "ram": 8.0,
        "weight": 2.0,
        "inches": 15.6,
        "res_width": 1920,
        "res_height": 1080,
        "storage_gb": 256.0,
        "has_ssd": 1,
        "has_hdd": 0,
        "is_ips": 0,
        "is_touchscreen": 0
    }
    models = get_available_models("regression")
    res = predict_price(models[0], sample)
    assert "predicted_price" in res
    assert res["predicted_price"] > 10000
    assert "formatted_price" in res
    assert "₹" in res["formatted_price"]
    assert res["r2"] is not None

def test_predict_price_category_classification():
    sample = {
        "company": "Apple",
        "typename": "Ultrabook",
        "opsys": "Macos",
        "ram": 16.0,
        "weight": 1.37,
        "inches": 13.3,
        "res_width": 2560,
        "res_height": 1600,
        "storage_gb": 512.0,
        "has_ssd": 1,
        "has_hdd": 0,
        "is_ips": 1,
        "is_touchscreen": 0
    }
    res = predict_price_category("Random Forest", sample)
    assert res["predicted_category"] in ["Budget", "Mid-range", "Premium", "Flagship"]
    assert res["probabilities"] is not None
    assert sum(res["probabilities"].values()) > 95.0

def test_missing_model_error():
    with pytest.raises(ValueError):
        load_model("regression", "FakeModelDoesNotExist")

def test_format_inr():
    assert format_inr(500) == "₹500"
    assert format_inr(50000) == "₹50,000"
    assert format_inr(125000) == "₹1,25,000"
    assert format_inr(1500000) == "₹15,00,000"
