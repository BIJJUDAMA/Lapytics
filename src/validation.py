import numpy as np
import pandas as pd
from src.config import (
    ALL_FEATURE_COLUMNS,
    COMPANIES,
    TYPENAMES,
    OPSYS_LIST,
    FEATURE_BOUNDS
)

def validate_laptop_input(input_dict: dict) -> tuple[bool, list[str]]:
    """
    Validates user input fields for laptop specifications.
    Returns (is_valid, list_of_error_messages).
    """
    errors = []

    # 1. Required keys
    required_keys = [
        "company", "typename", "opsys", "ram", "weight", "inches",
        "res_width", "res_height", "storage_gb", "has_ssd", "has_hdd",
        "is_ips", "is_touchscreen"
    ]
    for key in required_keys:
        if key not in input_dict or input_dict[key] is None:
            errors.append(f"Missing required field: '{key}'")

    if errors:
        return False, errors

    # 2. Categorical validations
    if input_dict["company"] not in COMPANIES:
        errors.append(f"Invalid company '{input_dict['company']}'. Allowed: {', '.join(COMPANIES)}")
    if input_dict["typename"] not in TYPENAMES:
        errors.append(f"Invalid laptop type '{input_dict['typename']}'. Allowed: {', '.join(TYPENAMES)}")
    if input_dict["opsys"] not in OPSYS_LIST:
        errors.append(f"Invalid operating system '{input_dict['opsys']}'. Allowed: {', '.join(OPSYS_LIST)}")

    # 3. Numeric bounds
    for num_col, (low, high) in FEATURE_BOUNDS.items():
        val = input_dict.get(num_col)
        try:
            val_float = float(val)
            if val_float < low or val_float > high:
                errors.append(f"{num_col.upper()} must be between {low} and {high} (got {val_float})")
        except (ValueError, TypeError):
            errors.append(f"Invalid numeric value for {num_col}: {val}")

    # 4. Storage configuration logic
    has_ssd = int(input_dict.get("has_ssd", 0))
    has_hdd = int(input_dict.get("has_hdd", 0))
    if has_ssd == 0 and has_hdd == 0:
        errors.append("At least one storage medium (SSD or HDD) must be selected.")

    # 5. Booleans
    for bool_col in ["is_ips", "is_touchscreen", "has_ssd", "has_hdd"]:
        val = input_dict.get(bool_col)
        if val not in [0, 1, True, False]:
            errors.append(f"'{bool_col}' must be 0 or 1.")

    return len(errors) == 0, errors


def build_feature_vector(input_dict: dict) -> pd.DataFrame:
    """
    Transforms validated user input into a 1-row DataFrame containing the 42
    preprocessed features expected by the trained ML models.
    """
    is_valid, errors = validate_laptop_input(input_dict)
    if not is_valid:
        raise ValueError(f"Input validation failed: {'; '.join(errors)}")

    # Initialize a row with 0 for all 42 columns
    row = {col: 0.0 for col in ALL_FEATURE_COLUMNS}

    # Numeric features
    inches = float(input_dict["inches"])
    res_w = float(input_dict["res_width"])
    res_h = float(input_dict["res_height"])
    ppi = np.sqrt(res_w**2 + res_h**2) / inches

    row["inches"] = inches
    row["ram"] = float(input_dict["ram"])
    row["weight"] = float(input_dict["weight"])
    row["res_width"] = int(res_w)
    row["res_height"] = int(res_h)
    row["storage_gb"] = float(input_dict["storage_gb"])
    row["ppi"] = ppi

    # Boolean flags (0 or 1)
    row["is_ips"] = int(bool(input_dict["is_ips"]))
    row["is_touchscreen"] = int(bool(input_dict["is_touchscreen"]))
    row["has_ssd"] = int(bool(input_dict["has_ssd"]))
    row["has_hdd"] = int(bool(input_dict["has_hdd"]))

    # Categorical dummy variables (drop_first=True matching pre_processing.ipynb)
    company = input_dict["company"]
    company_col = f"company_{company}"
    if company_col in row:
        row[company_col] = 1.0

    typename = input_dict["typename"]
    typename_col = f"typename_{typename}"
    if typename_col in row:
        row[typename_col] = 1.0

    opsys = input_dict["opsys"]
    opsys_col = f"opsys_{opsys}"
    if opsys_col in row:
        row[opsys_col] = 1.0

    # Create single-row DataFrame with exact column order
    df_vector = pd.DataFrame([row], columns=ALL_FEATURE_COLUMNS)
    return df_vector
