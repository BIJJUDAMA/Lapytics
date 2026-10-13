from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"
NOTEBOOKS_DIR = BASE_DIR / "notebooks"

# Ensure directories exist
MODELS_DIR.mkdir(parents=True, exist_ok=True)
(MODELS_DIR / "regression").mkdir(parents=True, exist_ok=True)
(MODELS_DIR / "classification").mkdir(parents=True, exist_ok=True)
(MODELS_DIR / "clustering").mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
NOTEBOOKS_DIR.mkdir(parents=True, exist_ok=True)

# SHA-256 Hashes of Existing Notebooks (Strict Read-Only Verification)
NOTEBOOK_HASHES = {
    "classification.ipynb": "767AB5C5D575B6837F1AD61ABC349B63610DA71F246D4A2BB873AF5399AC7C87",
    "pre_processing.ipynb": "F7822DE42521ECC3FA26213FCE98E9564D643D658139A8ACD6A22DB54376CA81",
    "Regression_algorithms_10.ipynb": "2F12499D72172772ECB094B52C180885255BBB012CD7BBFC901C079FD97391B9",
    "regression_first_5_algorithms.ipynb": "C652045DA75EA04DEEBB2C268C29ED025864588F04E3FF50ED73237D1173DF5A",
    "classification_10_algorithms.ipynb": "DA57C91281D871DE94A358235E901B8427C3F17922DA7B49C9D8E94D47FAD2BE",
    "clustering.ipynb": "E31D48E1CB38BB32A8DC7FE2BA26EAB9E63729A82F2CB02C106C42AA6353E015"
}

# The 42 feature columns exactly as preprocessed in data/*.csv
ALL_FEATURE_COLUMNS = [
    "inches",
    "ram",
    "weight",
    "is_ips",
    "is_touchscreen",
    "res_width",
    "res_height",
    "storage_gb",
    "has_ssd",
    "has_hdd",
    "ppi",
    "company_Apple",
    "company_Asus",
    "company_Chuwi",
    "company_Dell",
    "company_Fujitsu",
    "company_Google",
    "company_Hp",
    "company_Huawei",
    "company_Lenovo",
    "company_Lg",
    "company_Mediacom",
    "company_Microsoft",
    "company_Msi",
    "company_Razer",
    "company_Samsung",
    "company_Toshiba",
    "company_Vero",
    "company_Xiaomi",
    "typename_Gaming",
    "typename_Netbook",
    "typename_Notebook",
    "typename_Ultrabook",
    "typename_Workstation",
    "opsys_Chrome Os",
    "opsys_Linux",
    "opsys_Mac Os X",
    "opsys_Macos",
    "opsys_No Os",
    "opsys_Windows 10",
    "opsys_Windows 10 S",
    "opsys_Windows 7"
]

NUMERIC_FEATURES = [
    "inches", "ram", "weight", "res_width", "res_height", "storage_gb", "ppi"
]

BOOLEAN_FEATURES = [
    "is_ips", "is_touchscreen", "has_ssd", "has_hdd"
]

COMPANIES = [
    "Acer", "Apple", "Asus", "Chuwi", "Dell", "Fujitsu", "Google", "Hp",
    "Huawei", "Lenovo", "Lg", "Mediacom", "Microsoft", "Msi", "Razer",
    "Samsung", "Toshiba", "Vero", "Xiaomi"
]

TYPENAMES = [
    "2 in 1 Convertible", "Gaming", "Netbook", "Notebook", "Ultrabook", "Workstation"
]

OPSYS_LIST = [
    "Android", "Chrome Os", "Linux", "Mac Os X", "Macos", "No Os",
    "Windows 10", "Windows 10 S", "Windows 7"
]

PRICE_CATEGORIES = ["Budget", "Mid-range", "Premium", "Flagship"]

# Validation Bounds
FEATURE_BOUNDS = {
    "ram": (2.0, 64.0),
    "weight": (0.5, 5.0),
    "inches": (10.0, 18.5),
    "res_width": (1024, 3840),
    "res_height": (600, 2400),
    "storage_gb": (16.0, 2048.0)
}

RANDOM_STATE = 42
