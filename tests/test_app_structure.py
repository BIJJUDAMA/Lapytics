from pathlib import Path
from src.config import BASE_DIR
from streamlit.testing.v1 import AppTest

def test_files_and_pages_exist():
    assert (BASE_DIR / "app.py").exists()
    pages_dir = BASE_DIR / "pages"
    assert pages_dir.exists()
    expected = [
        "1_Price_Predictor.py",
        "2_Price_Classification.py",
        "3_Laptop_Clustering.py",
        "4_Model_Laboratory.py"
    ]
    for p in expected:
        assert (pages_dir / p).exists(), f"Missing page {p}"

def test_app_renders():
    at = AppTest.from_file(str(BASE_DIR / "app.py"), default_timeout=30)
    at.run()
    assert not at.exception
    assert len(at.title) > 0

def test_page_1_renders():
    at = AppTest.from_file(str(BASE_DIR / "pages" / "1_Price_Predictor.py"), default_timeout=30)
    at.run()
    assert not at.exception

def test_page_2_renders():
    at = AppTest.from_file(str(BASE_DIR / "pages" / "2_Price_Classification.py"), default_timeout=30)
    at.run()
    assert not at.exception

def test_page_3_renders():
    at = AppTest.from_file(str(BASE_DIR / "pages" / "3_Laptop_Clustering.py"), default_timeout=30)
    at.run()
    assert not at.exception

def test_page_4_renders():
    at = AppTest.from_file(str(BASE_DIR / "pages" / "4_Model_Laboratory.py"), default_timeout=30)
    at.run()
    assert not at.exception
