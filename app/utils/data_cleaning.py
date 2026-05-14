import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder


def handle_missing_values(df: pd.DataFrame, strategy: str = 'mean', knn_neighbors: int = 5) -> pd.DataFrame:
    """Imputes missing values in the dataframe."""
    df_cleaned = df.copy()
    num_cols = df_cleaned.select_dtypes(include=['number']).columns.tolist()
    cat_cols = df_cleaned.select_dtypes(include=['object', 'category']).columns.tolist()

    if strategy in ['mean', 'median']:
        if num_cols:
            num_imputer = SimpleImputer(strategy=strategy)
            df_cleaned[num_cols] = num_imputer.fit_transform(df_cleaned[num_cols])
        if cat_cols:
            cat_imputer = SimpleImputer(strategy='most_frequent')
            df_cleaned[cat_cols] = cat_imputer.fit_transform(df_cleaned[cat_cols])

    elif strategy == 'most_frequent':
        # Apply most_frequent to ALL columns (both numeric and categorical)
        if num_cols:
            num_imputer = SimpleImputer(strategy='most_frequent')
            df_cleaned[num_cols] = num_imputer.fit_transform(df_cleaned[num_cols])
        if cat_cols:
            cat_imputer = SimpleImputer(strategy='most_frequent')
            df_cleaned[cat_cols] = cat_imputer.fit_transform(df_cleaned[cat_cols])

    elif strategy == 'knn':
        if num_cols:
            knn_imputer = KNNImputer(n_neighbors=knn_neighbors)
            df_cleaned[num_cols] = knn_imputer.fit_transform(df_cleaned[num_cols])
        if cat_cols:
            cat_imputer = SimpleImputer(strategy='most_frequent')
            df_cleaned[cat_cols] = cat_imputer.fit_transform(df_cleaned[cat_cols])

    elif strategy == 'drop':
        df_cleaned = df_cleaned.dropna().reset_index(drop=True)

    return df_cleaned


def encode_categorical(df: pd.DataFrame, strategy: str = 'label') -> pd.DataFrame:
    """Encodes categorical columns."""
    df_encoded = df.copy()
    cat_cols = df_encoded.select_dtypes(include=['object', 'category']).columns.tolist()

    if not cat_cols:
        return df_encoded

    if strategy == 'label':
        le = LabelEncoder()
        for col in cat_cols:
            df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
    elif strategy == 'onehot':
        df_encoded = pd.get_dummies(df_encoded, columns=cat_cols, drop_first=True)

    return df_encoded


def scale_features(df: pd.DataFrame, strategy: str = 'standard') -> pd.DataFrame:
    """Scales numerical features."""
    df_scaled = df.copy()
    num_cols = df_scaled.select_dtypes(include=['number']).columns.tolist()

    if not num_cols:
        return df_scaled

    if strategy == 'standard':
        scaler = StandardScaler()
    elif strategy == 'minmax':
        scaler = MinMaxScaler()
    else:
        return df_scaled

    df_scaled[num_cols] = scaler.fit_transform(df_scaled[num_cols])
    return df_scaled


def remove_outliers(df: pd.DataFrame, method: str = 'iqr', factor: float = 1.5) -> pd.DataFrame:
    """Removes outliers based on IQR or Z-score."""
    df_out = df.copy()
    num_cols = df_out.select_dtypes(include=['number']).columns.tolist()

    if not num_cols:
        return df_out

    if method == 'iqr':
        mask = pd.Series([True] * len(df_out), index=df_out.index)
        for col in num_cols:
            Q1 = df_out[col].quantile(0.25)
            Q3 = df_out[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - factor * IQR
            upper_bound = Q3 + factor * IQR
            mask = mask & (df_out[col] >= lower_bound) & (df_out[col] <= upper_bound)
        df_out = df_out[mask].reset_index(drop=True)

    elif method == 'zscore':
        from scipy import stats
        filled = df_out[num_cols].fillna(df_out[num_cols].median())
        z_scores = np.abs(stats.zscore(filled))
        df_out = df_out[(z_scores < 3).all(axis=1)].reset_index(drop=True)

    return df_out


def detect_duplicates(df: pd.DataFrame) -> dict:
    """Detects and reports duplicate rows."""
    n_dupes = int(df.duplicated().sum())
    return {
        "duplicate_count": n_dupes,
        "duplicate_percentage": round(float(n_dupes / len(df) * 100), 2) if len(df) > 0 else 0.0
    }
