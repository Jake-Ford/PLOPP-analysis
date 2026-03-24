"""
data_prep.py
============
All data ingestion, cleaning, preprocessing transformers, and train/test
splitting for the PLOPP/FLOPP spectral classification pipeline.

Steps covered
-------------
1.  Raw CSV loading  –  reads per-sample spectral CSV files and pivots them
    so rows = samples and columns = integer wavenumber features.
2.  Label assignment –  attaches PLoPP / FLoPP class labels.
3.  Group extraction –  parses the numeric prefix from each filename so that
    samples from the same physical group stay together during splitting.
4.  Train / test split – group-stratified for PLOPP data, random for FLOPP;
    both combined into a single train and test set.
5.  Custom sklearn transformers – SNVTransformer and DerivativeTransformer,
    both sklearn-compatible (BaseEstimator / TransformerMixin).
6.  External validation loaders – Andrew Turner environmental samples and
    Citadel paint samples.

Usage
-----
    from data_prep import load_training_data, load_validation_data

    X_train, y_train, X_test, y_test = load_training_data()
    X_val, y_val, val_meta           = load_validation_data()
"""

from __future__ import annotations

import os
import re
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split

import config

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _process_csv_directory(
    directory: Path,
    exclude: set[str] | None = None,
    uppercase_only: bool = False,
    lowercase_only: bool = False,
) -> pd.DataFrame:
    """
    Load all CSV files in *directory* and return a wide DataFrame.

    Each CSV is expected to have exactly two columns: Wavelength and Datavalue.
    Rows = samples, columns = integer wavenumber features (matching the notebook
    which casts all wavenumber column names to int after loading).

    Parameters
    ----------
    directory : Path
    exclude : set of filenames to skip (case-insensitive)
    uppercase_only : if True, only load files ending in '.CSV' (uppercase),
                     matching the Andrew Turner notebook logic.
    lowercase_only : if True, only load files ending in '.csv' (lowercase),
                     matching the Citadel notebook logic (excludes stray .CSV files
                     that fail to merge with the sample key).
    """
    directory = Path(directory)
    if not directory.exists():
        raise FileNotFoundError(f"Data directory not found: {directory}")

    exclude = {e.lower() for e in (exclude or set())}

    all_files = os.listdir(directory)
    if uppercase_only:
        csv_files = [f for f in all_files if f.endswith(".CSV") and f.lower() not in exclude]
    elif lowercase_only:
        csv_files = [f for f in all_files if f.endswith(".csv") and f.lower() not in exclude]
    else:
        csv_files = [f for f in all_files if f.lower().endswith(".csv") and f.lower() not in exclude]

    if not csv_files:
        raise ValueError(f"No CSV files found in: {directory}")

    frames = []
    for filename in csv_files:
        file_path = directory / filename
        df = pd.read_csv(file_path, header=None, names=["Wavelength", "Datavalue"])
        df = df.drop_duplicates(subset="Wavelength")
        df_transposed = df.set_index("Wavelength").T
        # Convert wavenumber column names to integers — matches notebook Cell 5/7
        df_transposed.columns = [int(col) for col in df_transposed.columns]
        df_transposed["Sample"] = Path(filename).stem
        frames.append(df_transposed)

    master_df = pd.concat(frames, axis=0, ignore_index=True).fillna(0)
    logger.info("Loaded %d samples from %s", len(master_df), directory)
    return master_df


def _extract_group_prefix(sample_name: str) -> str | None:
    """
    Return the leading integer string from *sample_name*, e.g. '12' from
    '12_scan_001'.  Returns None when no leading integer is present.
    """
    match = re.match(r"^(\d+)", str(sample_name))
    return match.group(1) if match else None


# ---------------------------------------------------------------------------
# sklearn-compatible preprocessing transformers
# ---------------------------------------------------------------------------

class SNVTransformer(BaseEstimator, TransformerMixin):
    """
    Standard Normal Variate (SNV) transformation.

    Mean-centres each spectrum and, optionally, scales it by its standard
    deviation.  Equivalent to row-wise z-score normalisation.

    Parameters
    ----------
    use_scaling : bool, default True
        When True the centred spectrum is divided by its standard deviation.
    """

    def __init__(self, use_scaling: bool = True):
        self.use_scaling = use_scaling

    def fit(self, X, y=None):  # noqa: N803
        return self

    def transform(self, X):  # noqa: N803
        X = np.array(X, dtype=float)
        mean_centred = X - np.mean(X, axis=1, keepdims=True)
        if self.use_scaling:
            std = np.std(mean_centred, axis=1, keepdims=True)
            std = np.where(std == 0, 1.0, std)
            return mean_centred / std
        return mean_centred


class DerivativeTransformer(BaseEstimator, TransformerMixin):
    """
    Savitzky-Golay derivative transformation.

    Applies a Savitzky-Golay filter row-wise to compute the *deriv*-th
    derivative of each spectrum.

    Parameters
    ----------
    window_length : int, default 11
        Length of the filter window (must be odd and > polyorder).
    polyorder : int, default 2
        Order of the polynomial used to fit the samples.
    deriv : int, default 1
        The order of the derivative to compute (1 = first derivative).
    """

    def __init__(
        self,
        window_length: int = config.DERIVATIVE_WINDOW,
        polyorder: int = config.DERIVATIVE_POLYORDER,
        deriv: int = config.DERIVATIVE_ORDER,
    ):
        self.window_length = window_length
        self.polyorder = polyorder
        self.deriv = deriv

    def fit(self, X, y=None):  # noqa: N803
        return self

    def transform(self, X):  # noqa: N803
        X = np.array(X, dtype=float)
        return np.apply_along_axis(
            savgol_filter,
            axis=1,
            arr=X,
            window_length=self.window_length,
            polyorder=self.polyorder,
            deriv=self.deriv,
        )


# ---------------------------------------------------------------------------
# Training data loaders
# ---------------------------------------------------------------------------

def _load_plopp_raw() -> pd.DataFrame:
    """Load raw PLOPP spectra and attach class label + group column."""
    df = _process_csv_directory(config.PLOPP_DATA_DIR)
    df["Target"] = config.LABEL_PLOPP
    df["Group"]  = df["Sample"].apply(_extract_group_prefix)
    return df


def _load_flopp_raw() -> pd.DataFrame:
    """Load raw FLOPP spectra and attach class label + group column."""
    df = _process_csv_directory(config.FLOPP_DATA_DIR)
    df["Target"] = config.LABEL_FLOPP
    df["Group"]  = df["Sample"].apply(_extract_group_prefix)
    return df


def _split_by_group(
    df: pd.DataFrame,
    test_size: float = config.TEST_SIZE,
    random_state: int = config.RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Group-aware train/test split: all samples from the same numeric group
    are kept in the same partition to prevent data leakage.
    """
    groups = df["Group"].unique()
    train_groups, test_groups = train_test_split(
        groups, test_size=test_size, random_state=random_state
    )
    return (
        df[df["Group"].isin(train_groups)].copy(),
        df[df["Group"].isin(test_groups)].copy(),
    )


def load_training_data() -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Load, clean, and split PLOPP + FLOPP data into train and test sets.

    Matches notebook Cells 5-10:
    - PLOPP and FLOPP are loaded separately, columns coerced to int
    - Combined into result_df via concat
    - PLOPP uses a group-stratified split; FLOPP uses a simple random split
    - Both are merged into a single train and test set

    Returns
    -------
    X_train, y_train, X_test, y_test
    """
    logger.info("Loading PLOPP data …")
    plopp_df = _load_plopp_raw()

    logger.info("Loading FLOPP data …")
    flopp_df = _load_flopp_raw()

    # Notebook Cell 7: find shared wavenumber columns across both datasets
    merged_cols = [c for c in plopp_df.columns if c in flopp_df.columns]
    plopp_df = plopp_df[merged_cols]
    flopp_df = flopp_df[merged_cols]

    # Notebook Cell 8: concat into one result_df
    result_df = pd.concat([flopp_df, plopp_df], axis=0, ignore_index=True)
    result_df["Group"] = result_df["Sample"].apply(_extract_group_prefix)

    # Notebook Cell 10: separate back by class for the split
    plopp_df = result_df[result_df["Target"] == config.LABEL_PLOPP]
    flopp_df = result_df[result_df["Target"] == config.LABEL_FLOPP]

    plopp_train, plopp_test = _split_by_group(plopp_df)
    flopp_train, flopp_test = train_test_split(
        flopp_df, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE
    )

    train_df = pd.concat([plopp_train, flopp_train], ignore_index=True)
    test_df  = pd.concat([plopp_test,  flopp_test],  ignore_index=True)

    meta_cols = ["Target", "Sample", "Group"]
    X_train       = train_df.drop(columns=meta_cols)
    y_train       = train_df["Target"]
    X_test        = test_df.drop(columns=meta_cols)
    y_test        = test_df["Target"]
    test_samples  = test_df["Sample"].reset_index(drop=True)

    logger.info(
        "Train: %d samples  |  Test: %d samples  |  Features: %d",
        len(X_train), len(X_test), X_train.shape[1],
    )
    return X_train, y_train, X_test, y_test, test_samples


# ---------------------------------------------------------------------------
# External validation loaders
# ---------------------------------------------------------------------------

def load_andrew_turner_data() -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """
    Load Andrew Turner environmental validation spectra.

    Matches notebook Cells 28-29:
    - Only uppercase .CSV files are loaded (skipping SampleKey and non-spectral files)
    - Sample names have the .CSV extension stripped
    - Merged with SampleKey_correct_sector.csv on Mira's ID
    - MP_ID_5 excluded per project decision (11.11.24)

    Returns
    -------
    X_val, y_val, meta
    """
    directory = config.ANDREW_TURNER_DIR
    key_path  = directory / "SampleKey_correct_sector.csv"

    # Notebook Cell 28: loads only uppercase .CSV files, skips SampleKey.csv
    spectra_df = _process_csv_directory(
        directory,
        exclude={"SampleKey.csv", key_path.name},
        uppercase_only=True,
    )

    # utf-8-sig strips the BOM character from the first column name
    sample_key = pd.read_csv(key_path, encoding="utf-8-sig")

    # Notebook Cell 29: merge on Mira's ID
    merged = pd.merge(
        spectra_df,
        sample_key,
        left_on="Sample",
        right_on="Mira's ID",
        how="left",
    )
    merged.drop(columns=["Mira's ID"], errors="ignore", inplace=True)

    # Notebook Cell 29: two-step label normalisation
    merged["observed"] = merged["Paint or non-paint"].map({"Non-paint": "FLOPP", "Paint": "PLOPP"})
    merged["observed"] = merged["observed"].map({"PLOPP": config.LABEL_PLOPP, "FLOPP": config.LABEL_FLOPP})
    merged.drop(columns=["Paint or non-paint"], errors="ignore", inplace=True)

    # Drop excluded sample and rows without a label
    merged = merged[merged["Sample"] != "MP_ID_5"].copy()
    merged = merged.dropna(subset=["observed"])

    # Standardise sector labels (same as Citadel loader)
    merged["zoie_sector"] = merged["zoie_sector"].replace({
        "Industrial Wood": "Wood",
        "Road marking": "Road Marking",
    })

    meta_cols = ["Sample", "zoie_sector", "observed"]
    feature_cols = [c for c in merged.columns if c not in meta_cols]

    X_val = merged[feature_cols]
    y_val = merged["observed"].rename("Target")
    meta  = merged[["Sample", "zoie_sector", "observed"]]

    logger.info("Andrew Turner validation: %d samples loaded", len(X_val))
    return X_val, y_val, meta


def load_citadel_data(
    at_feature_cols: list | None = None,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """
    Load Citadel paint validation spectra.

    Matches notebook Cells 30-32:
    - If at_feature_cols is provided (Andrew Turner feature column list), the
      Citadel DataFrame's columns are renamed to match — exactly as the notebook
      does by assigning merged_df column names to new_test.
    - Sector labels are standardised (e.g. 'Industrial Wood' → 'Wood').

    Parameters
    ----------
    at_feature_cols : list, optional
        Feature column names from the Andrew Turner dataset.  When provided and
        the column count matches, Citadel columns are renamed to these to ensure
        wavenumber alignment (replicating notebook Cell 30).

    Returns
    -------
    X_val, y_val, meta
    """
    directory = config.CITADEL_DATA_DIR
    key_path  = config.CITADEL_SAMPLE_KEY

    # lowercase_only=True matches notebook behaviour: the notebook's
    # .str.replace('.csv','') only strips lowercase extensions, so any
    # uppercase .CSV files in this directory fail to join with the sample
    # key and are effectively excluded.
    spectra_df = _process_csv_directory(directory, lowercase_only=True)

    # Notebook Cell 30: assign AT column names to Citadel to ensure alignment
    if at_feature_cols is not None:
        cols_to_keep = [c for c in at_feature_cols] + ["Sample"]
        if len(spectra_df.columns) == len(cols_to_keep):
            spectra_df.columns = cols_to_keep
        else:
            logger.warning(
                "Citadel column count (%d) != AT column count (%d); "
                "skipping column name assignment",
                len(spectra_df.columns), len(cols_to_keep),
            )

    # Notebook Cell 32: load sample key and map labels
    sample_key = pd.read_excel(key_path, usecols=["Sample name", "Paint or non-paint", "Sector"])
    sample_key.rename(
        columns={
            "Paint or non-paint": "observed",
            "Sector": "zoie_sector",
            "Sample name": "Sample",
        },
        inplace=True,
    )
    sample_key["observed"] = sample_key["observed"].map(
        {"Non-paint": config.LABEL_FLOPP, "Paint": config.LABEL_PLOPP}
    )

    merged = pd.merge(spectra_df, sample_key, on="Sample", how="left")
    merged = merged.dropna(subset=["observed"])

    # Standardise sector labels (notebook Cell 33)
    merged["zoie_sector"] = merged["zoie_sector"].replace({
        "Industrial Wood": "Wood",
        "Road marking": "Road Marking",
    })

    meta_cols = ["Sample", "zoie_sector", "observed"]
    feature_cols = [c for c in merged.columns if c not in meta_cols]

    X_val = merged[feature_cols]
    y_val = merged["observed"].rename("Target")
    meta  = merged[["Sample", "zoie_sector", "observed"]]

    logger.info("Citadel validation: %d samples loaded", len(X_val))
    return X_val, y_val, meta


def load_plopp_with_sectors() -> pd.DataFrame:
    """
    Load raw PLOPP spectra joined with sector labels.

    Matches notebook Cells 42-45: loads PLoPP-only training data and
    merges with the sector mapping so each sample has a Sector column.
    Samples without a known sector are dropped.

    Returns
    -------
    pd.DataFrame with integer wavenumber columns + Sample, Target, Group, Sector
    """
    plopp_df = _load_plopp_raw()
    sector_map = load_sector_map()
    merged = plopp_df.merge(sector_map, on="Sample", how="left")
    merged = merged.dropna(subset=["Sector"])
    logger.info(
        "PLoPP with sectors: %d samples across %d sectors",
        len(merged), merged["Sector"].nunique(),
    )
    return merged


def load_sector_map() -> pd.DataFrame:
    """
    Load the sector mapping for PLOPP training samples.

    Returns a DataFrame with columns ['Sample', 'Sector'] where Sample
    matches the stem of the original CSV filename (e.g. '54c').
    """
    df = pd.read_csv(config.SECTOR_MAP_PATH, usecols=["Sample", "Sector"])
    # Strip any trailing whitespace / carriage-returns from Windows-formatted CSV
    df["Sector"] = df["Sector"].str.strip()
    df["Sector"] = df["Sector"].replace({"Industrial Wood": "Wood"})
    return df


def load_validation_data() -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """
    Load and combine all external validation datasets (Andrew Turner + Citadel).

    Matches notebook Cell 33:
    - Andrew Turner loaded first; its feature columns are passed to Citadel
      loader so both share the same wavenumber column names (notebook assigns
      merged_df columns to new_test).
    - Concatenated and NaN-observed rows dropped.

    Returns
    -------
    X_val, y_val, meta
    """
    X_at, y_at, meta_at = load_andrew_turner_data()

    # Pass AT feature columns so Citadel is aligned to the same wavenumber grid
    X_ct, y_ct, meta_ct = load_citadel_data(at_feature_cols=list(X_at.columns))

    X_val = pd.concat([X_at, X_ct], axis=0, ignore_index=True)
    y_val = pd.concat([y_at, y_ct], ignore_index=True)
    meta  = pd.concat([meta_at, meta_ct], ignore_index=True)

    logger.info("Combined validation: %d samples", len(X_val))
    return X_val, y_val, meta
