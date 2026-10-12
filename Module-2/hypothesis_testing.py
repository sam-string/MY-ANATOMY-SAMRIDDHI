"""
A/B Test Analysis for an E-Commerce Website
Run with: python hypothesis_testing.py

Loads ab_data.csv and countries.csv from GitHub, with optional local fallback.
The script computes results from the actual data and does not hardcode report results.
"""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from scipy import stats

ALPHA = 0.05
AB_URL = "https://raw.githubusercontent.com/aamir-gk2539-byte/DSML-Training/main/ab_data.csv"
COUNTRIES_URL = "https://raw.githubusercontent.com/aamir-gk2539-byte/DSML-Training/main/countries.csv"
# Optional: set to a folder containing both CSVs, e.g. Path(r"C:/Users/YourName/Downloads")
LOCAL_DATA_DIR = None


def load_csv(url, filename, local_dir=None):
    """Load a CSV from URL, falling back to a local file if configured."""
    errors = []
    try:
        df = pd.read_csv(url)
        print(f"Loaded {filename} from GitHub.")
        return df
    except Exception as exc:
        errors.append(f"GitHub download failed: {type(exc).__name__}: {exc}")
    if local_dir is not None:
        path = Path(local_dir) / filename
        try:
            df = pd.read_csv(path)
            print(f"Loaded {filename} from local file: {path}")
            return df
        except Exception as exc:
            errors.append(f"Local-file load failed ({path}): {type(exc).__name__}: {exc}")
    raise RuntimeError(
        f"Could not load {filename}.\n" + "\n".join(errors) +
        "\nDownload the CSV, set LOCAL_DATA_DIR in this script, and rerun."
    )


def normalize_columns(df):
    df = df.copy()
    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]
    return df


def main():
    try:
        ab = normalize_columns(load_csv(AB_URL, "ab_data.csv", LOCAL_DATA_DIR))
        countries = normalize_columns(load_csv(COUNTRIES_URL, "countries.csv", LOCAL_DATA_DIR))
    except RuntimeError as exc:
        print(exc, file=sys.stderr)
        sys.exit(1)

    required_ab = {"user_id", "group", "landing_page", "converted"}
    required_countries = {"user_id", "country"}
    if not required_ab.issubset(ab.columns):
        raise ValueError(f"A/B data missing {sorted(required_ab - set(ab.columns))}; found {ab.columns.tolist()}")
    if not required_countries.issubset(countries.columns):
        raise ValueError(f"Countries data missing {sorted(required_countries - set(countries.columns))}; found {countries.columns.tolist()}")

    print("\nDATASET INSPECTION")
    for name, df in [("A/B data", ab), ("Countries", countries)]:
        print(f"\n{name}: shape={df.shape}; columns={df.columns.tolist()}")
        print("Missing values:", df.isna().sum().to_dict())
        if "user_id" in df:
            print("Unique user IDs:", df["user_id"].nunique())

    if "timestamp" in ab.columns:
        ab["timestamp"] = pd.to_datetime(ab["timestamp"], errors="coerce")
    ab["group"] = ab["group"].astype("string").str.strip().str.lower()
    ab["landing_page"] = ab["landing_page"].astype("string").str.strip().str.lower()
    ab["converted"] = pd.to_numeric(ab["converted"], errors="coerce")

    print("\nBEFORE CLEANING")
    print("Rows:", len(ab))
    print("Duplicate user_id rows beyond first:", int(ab["user_id"].duplicated().sum()))
    print("Rows with missing values:", int(ab.isna().any(axis=1).sum()))

    is_control = ab["group"].isin({"control"})
    is_treatment = ab["group"].isin({"treatment"})
    is_old = ab["landing_page"].isin({"old_page", "old page", "old"})
    is_new = ab["landing_page"].isin({"new_page", "new page", "new"})
    matches = (is_control & is_old) | (is_treatment & is_new)
    print("Group/page mismatches (including unrecognized labels):", int((~matches).sum()))
    ab = ab.loc[matches].copy()

    if "timestamp" in ab.columns:
        ab = ab.sort_values("timestamp", kind="stable", na_position="last")
    ab = ab.drop_duplicates(subset="user_id", keep="first").copy()
    before_outcome_filter = len(ab)
    ab = ab.loc[ab["converted"].isin([0, 1])].copy()
    print("Rows removed for missing/non-binary converted:", before_outcome_filter - len(ab))

    countries["country"] = countries["country"].astype("string").str.strip()
    duplicate_country_keys = int(countries["user_id"].duplicated().sum())
    print("Duplicate country lookup keys:", duplicate_country_keys)
    if duplicate_country_keys:
        countries = countries.drop_duplicates(subset="user_id", keep="first").copy()

    rows_before_merge = len(ab)
    merged = ab.merge(countries[["user_id", "country"]], on="user_id",
                      how="left", validate="one_to_one", indicator=True)
    print("Rows before merge:", rows_before_merge)
    print("Rows after merge:", len(merged))
    print("Rows without country match:", int((merged["_merge"] != "both").sum()))
    assert len(merged) == rows_before_merge, "Merge unexpectedly changed row count."
    merged = merged.loc[merged["_merge"] == "both"].drop(columns="_merge").copy()

    print("\nCLEANED SAMPLE")
    print("Rows:", len(merged), "Unique users:", merged["user_id"].nunique())
    print("Group sizes:\n", merged["group"].value_counts().to_string())
    summary = merged.groupby("group", observed=True).agg(
        users=("user_id", "nunique"),
        conversions=("converted", "sum"),
        conversion_rate=("converted", "mean")
    )
    summary["conversion_rate_percent"] = summary["conversion_rate"] * 100
    print("\nDESCRIPTIVE STATISTICS\n", summary.to_string())
    if {"control", "treatment"}.issubset(summary.index):
        print("Treatment − control difference (pp):",
              (summary.loc["treatment", "conversion_rate"] -
               summary.loc["control", "conversion_rate"]) * 100)

    # Test 1: Welch independent-samples t-test
    control = merged.loc[merged["group"] == "control", "converted"].astype(float).to_numpy()
    treatment = merged.loc[merged["group"] == "treatment", "converted"].astype(float).to_numpy()
    t_stat, t_p = stats.ttest_ind(treatment, control, equal_var=False, nan_policy="omit")
    vt, vc = np.var(treatment, ddof=1), np.var(control, ddof=1)
    se = np.sqrt(vt / len(treatment) + vc / len(control))
    df_num = (vt / len(treatment) + vc / len(control)) ** 2
    df_den = (vt / len(treatment)) ** 2 / (len(treatment) - 1) + (vc / len(control)) ** 2 / (len(control) - 1)
    welch_df = df_num / df_den
    diff = np.mean(treatment) - np.mean(control)
    crit = stats.t.ppf(1 - ALPHA / 2, welch_df)
    ci = (diff - crit * se, diff + crit * se)
    print("\nTEST 1 — WELCH T-TEST")
    print(f"Control mean={np.mean(control):.6f}; treatment mean={np.mean(treatment):.6f}")
    print(f"Difference (treatment-control)={diff*100:.4f} percentage points")
    print(f"t={t_stat:.4f}; df={welch_df:.2f}; p={t_p:.6f}")
    print(f"95% CI=[{ci[0]*100:.4f}, {ci[1]*100:.4f}] percentage points")
    print("Decision:", "Reject H0" if t_p < ALPHA else "Fail to reject H0")

    # Test 2: Chi-square test of independence
    table = pd.crosstab(merged["group"], merged["converted"]).reindex(
        index=["control", "treatment"], columns=[0, 1], fill_value=0)
    chi2, chi_p, chi_df, expected = stats.chi2_contingency(table.to_numpy())
    print("\nTEST 2 — CHI-SQUARE TEST")
    print("Contingency table (rows=group; columns=converted 0, 1):\n", table.to_string())
    print(f"chi-square={chi2:.4f}; df={chi_df}; p={chi_p:.6f}")
    print("Expected frequencies:\n", pd.DataFrame(expected, index=table.index, columns=table.columns).round(3).to_string())
    print("Decision:", "Reject H0" if chi_p < ALPHA else "Fail to reject H0")

    # Test 3A: One-way ANOVA across countries
    country_summary = merged.groupby("country", observed=True).agg(
        users=("user_id", "nunique"), conversions=("converted", "sum"),
        conversion_rate=("converted", "mean"))
    country_summary["conversion_rate_percent"] = country_summary["conversion_rate"] * 100
    country_samples = [g["converted"].astype(float).to_numpy()
                       for _, g in merged.groupby("country", observed=True) if len(g)]
    country_f, country_p = stats.f_oneway(*country_samples)
    country_df = (len(country_samples) - 1, len(merged) - len(country_samples))
    print("\nTEST 3A — COUNTRY ANOVA")
    print(country_summary.to_string())
    print(f"F={country_f:.4f}; df={country_df}; p={country_p:.6f}")
    print("Decision:", "Reject H0" if country_p < ALPHA else "Fail to reject H0")

    # Test 3B: One-way ANOVA across country × strategy combinations.
    merged["country_strategy"] = merged["group"].astype(str).str.title() + " – " + merged["country"].astype(str)
    combo_summary = merged.groupby("country_strategy", observed=True).agg(
        users=("user_id", "nunique"), conversions=("converted", "sum"),
        conversion_rate=("converted", "mean"))
    combo_summary["conversion_rate_percent"] = combo_summary["conversion_rate"] * 100
    combo_samples = [g["converted"].astype(float).to_numpy()
                     for _, g in merged.groupby("country_strategy", observed=True) if len(g)]
    combo_f, combo_p = stats.f_oneway(*combo_samples)
    combo_df = (len(combo_samples) - 1, len(merged) - len(combo_samples))
    print("\nTEST 3B — COUNTRY–STRATEGY ANOVA")
    print(combo_summary.to_string())
    print(f"F={combo_f:.4f}; df={combo_df}; p={combo_p:.6f}")
    print("Decision:", "Reject H0" if combo_p < ALPHA else "Fail to reject H0")
    print("\nNote: One-way ANOVA across country–strategy combinations is not a formal interaction test.")

    results = pd.DataFrame([
        ["Welch t-test", f"t={t_stat:.4f}", f"{welch_df:.2f}", t_p, "Reject H0" if t_p < ALPHA else "Fail to reject H0"],
        ["Chi-square", f"chi2={chi2:.4f}", str(chi_df), chi_p, "Reject H0" if chi_p < ALPHA else "Fail to reject H0"],
        ["Country ANOVA", f"F={country_f:.4f}", str(country_df), country_p, "Reject H0" if country_p < ALPHA else "Fail to reject H0"],
        ["Country–strategy ANOVA", f"F={combo_f:.4f}", str(combo_df), combo_p, "Reject H0" if combo_p < ALPHA else "Fail to reject H0"],
    ], columns=["Test", "Statistic", "Degrees of freedom", "P-value", "Decision"])
    print("\nRESULTS SUMMARY\n", results.to_string(index=False, formatters={"P-value": lambda x: f"{x:.6f}"}))
    merged.to_csv("ab_test_cleaned_merged.csv", index=False)
    results.to_csv("ab_test_results_summary.csv", index=False)
    print("\nSaved ab_test_cleaned_merged.csv and ab_test_results_summary.csv")


if __name__ == "__main__":
    main()
