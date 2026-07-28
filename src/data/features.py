"""Feature engineering module"""

import pandas as pd
import numpy as np


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create and engineer features for modeling.
    
    Args:
        df: Input DataFrame
        
    Returns:
        DataFrame with engineered features
    """
    df = df.copy()
    
    # Add temporal features if date columns exist
    if "InvoiceDate" in df.columns:
        df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
        df["Month"] = df["InvoiceDate"].dt.month
        df["DayOfWeek"] = df["InvoiceDate"].dt.dayofweek
        df["Quarter"] = df["InvoiceDate"].dt.quarter
    
    # Add aggregation features
    # These are examples; adjust based on your actual data
    if "Quantity" in df.columns and "UnitPrice" in df.columns:
        df["TotalValue"] = df["Quantity"] * df["UnitPrice"]
    
    return df


def add_rfm_features(df: pd.DataFrame, customer_id_col: str = "CustomerID",
                     invoice_date_col: str = "InvoiceDate", value_col: str = "TotalValue") -> pd.DataFrame:
    """
    Compute RFM (Recency, Frequency, Monetary) features per customer.

    Args:
        df: Transactions DataFrame with at least customer id, invoice date and value columns
        customer_id_col: Column name for customer identifier
        invoice_date_col: Column name for invoice date (datetime64)
        value_col: Column name for transaction value

    Returns:
        DataFrame with `CustomerID`, `RFM_Recency`, `RFM_Frequency`, `RFM_Monetary`
    """
    if invoice_date_col in df.columns:
        df[invoice_date_col] = pd.to_datetime(df[invoice_date_col])
    else:
        raise ValueError(f"Missing invoice date column: {invoice_date_col}")

    if customer_id_col not in df.columns:
        raise ValueError(f"Missing customer id column: {customer_id_col}")

    # Reference date: one day after the latest invoice
    ref_date = df[invoice_date_col].max() + pd.Timedelta(days=1)

    grouped = df.groupby(customer_id_col)
    rfm = pd.DataFrame()
    rfm["RFM_Recency"] = grouped[invoice_date_col].apply(lambda x: (ref_date - x.max()).days)
    rfm["RFM_Frequency"] = grouped.size()
    rfm["RFM_Monetary"] = grouped[value_col].sum()

    rfm = rfm.reset_index()
    # Optionally, you can add percentile ranks or segments here
    return rfm


def select_features(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Select features based on correlation threshold.
    
    Args:
        df: Input DataFrame
        config: Configuration dictionary
        
    Returns:
        DataFrame with selected features
    """
    threshold = config.get("features", {}).get("correlation_threshold", 0.95)
    
    # Calculate correlation matrix
    numeric_df = df.select_dtypes(include=[np.number])
    corr_matrix = numeric_df.corr().abs()
    
    # Select features based on correlation threshold
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [col for col in upper.columns if any(upper[col] > threshold)]
    
    return df.drop(columns=to_drop, errors="ignore")
