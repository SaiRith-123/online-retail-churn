import pandas as pd


def clean(df, cfg):
    rules = cfg["cleaning"]

    # Standardize types
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")
    df["Customer ID"] = pd.to_numeric(df["Customer ID"], errors="coerce")

    if rules["drop_null_customer_id"]:
        df = df[df["Customer ID"].notna()]

    if rules["drop_cancelled_invoices"]:
        df = df[~df["Invoice"].astype(str).str.startswith("C")]

    if rules["drop_negative_quantity"]:
        df = df[df["Quantity"] > 0]

    if rules["drop_non_positive_price"]:
        df = df[df["Price"] > 0]

    if rules["drop_postage"]:
        df = df[df["StockCode"].astype(str).str.upper() != "POST"]

    df = df.drop_duplicates()
    df["Customer ID"] = df["Customer ID"].astype(int)
    return df.reset_index(drop=True)


# Backwards-compatible alias used elsewhere in the codebase
clean_data = clean
