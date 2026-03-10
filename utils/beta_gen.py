import re
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd


def _find_col_exact_or_fuzzy(df_columns: List[str], name: str) -> Optional[str]:
    """Exact -> case-insensitive -> suffix *_name. Return None if ambiguous/not found."""
    if name in df_columns:
        return name
    low = {c.lower(): c for c in df_columns}
    if name.lower() in low:
        return low[name.lower()]
    hits = [
        c for c in df_columns
        if c.lower() == name.lower() or c.lower().endswith("_" + name.lower())
    ]
    return hits[0] if len(hits) == 1 else None


def _select_x_cols_by_regex(
    cols: List[str],
    pattern: str = r"^[A-Za-z]{2}\d{3,4}$",
    *,
    sort_by_csv_order: bool = True,
) -> List[str]:
    rx = re.compile(pattern)
    picked = [c for c in cols if rx.match(str(c).strip())]
    if not picked:
        raise ValueError(
            f"No X columns matched regex pattern={pattern}. "
            f"Expected like AA001, SS010, GF1234."
        )
    return picked if sort_by_csv_order else sorted(picked)


def _build_quadratic_scheffe_features(
    X: np.ndarray,
    X_cols: List[str],
) -> Tuple[np.ndarray, List[str]]:
    """
    Build Phi = [x_i] + [x_i*x_j for i<j]
    Return (Phi, feat_names) where feat_names aligns with Phi columns.
    """
    N, D = X.shape
    pairs = [(i, j) for i in range(D) for j in range(i + 1, D)]

    Phi_lin = X  # (N,D)
    Phi_int = np.empty((N, len(pairs)), dtype=float)
    for k, (i, j) in enumerate(pairs):
        Phi_int[:, k] = X[:, i] * X[:, j]

    Phi = np.hstack([Phi_lin, Phi_int])  # (N,P)

    feat_names = (
        [f"x[{c}]" for c in X_cols]
        + [f"x[{X_cols[i]}]*x[{X_cols[j]}]" for i, j in pairs]
    )
    return Phi, feat_names


def compute_V_task_correlation(
    BETAS_PATH: str,
    DATA_PATH: str,
    *,
    # X selection
    x_regex: str = r"^[A-Za-z]{2}\d{3,4}$",
    # tasks: if None -> infer from betas columns
    tasks: Optional[List[str]] = None,
    # intercept row detection
    intercept_feature_regex: str = r"intercept",  # case-insensitive contains
    # behavior
    save_v_csv: bool = True,
    v_csv_path: str = "task_covariance_V.csv",
    encoding: str = "utf-8-sig",
    verbose: bool = True,
) -> np.ndarray:
    """
    Compute V_hat = (1/N) * E^T E where E = Y - (Phi@B + B0)

    Inputs:
      - BETAS_PATH: betas_all.csv produced by your fitting function
      - DATA_PATH: original fitting data CSV containing X cols + task cols

    Returns:
      - V_hat as numpy array (M,M), and prints it.
    """

    # 1) Load
    betas = pd.read_csv(BETAS_PATH)
    df = pd.read_csv(DATA_PATH)

    if "feature" not in betas.columns:
        raise ValueError("betas_all.csv must contain a 'feature' column.")

    # 2) Resolve X columns by regex in data
    cols_data = list(df.columns)
    X_cols = _select_x_cols_by_regex(cols_data, pattern=x_regex, sort_by_csv_order=True)
    D = len(X_cols)
    if verbose:
        print(f"[INFO] X columns ({D}): {X_cols}")

    # 3) Decide tasks
    # If tasks not provided, infer from betas columns by excluding meta columns.
    meta_cols = {"feature", "type", "nonzero_row_l2", "active"}
    if tasks is None:
        inferred = [c for c in betas.columns if c not in meta_cols]
        if not inferred:
            raise ValueError(
                "Cannot infer tasks from betas_all.csv columns. "
                "Please pass tasks=['SPGR','TE',...]."
            )
        tasks = inferred

    # 4) Resolve task columns in data and in betas (fuzzy)
    task_cols_data = []
    for t in tasks:
        c = _find_col_exact_or_fuzzy(cols_data, t)
        if c is None:
            raise ValueError(f"Cannot find task '{t}' in data CSV.")
        task_cols_data.append(c)

    task_cols_betas = []
    cols_betas = list(betas.columns)
    for t in tasks:
        c = _find_col_exact_or_fuzzy(cols_betas, t)
        if c is None:
            raise ValueError(f"Cannot find task '{t}' in betas_all.csv.")
        task_cols_betas.append(c)

    if verbose:
        print(f"[INFO] Tasks(data):  {task_cols_data}")
        print(f"[INFO] Tasks(betas): {task_cols_betas}")

    # 5) Filter valid rows (non-null X & Y)
    use_cols = X_cols + task_cols_data
    work = df[use_cols].copy()
    for c in use_cols:
        work[c] = pd.to_numeric(work[c], errors="coerce")
    work = work.dropna(subset=use_cols).reset_index(drop=True)

    N = len(work)
    if N == 0:
        raise ValueError("No valid rows after dropna. Check your CSV content.")
    if verbose:
        print(f"[INFO] Rows used for V (N): {N}")

    X = work[X_cols].to_numpy(dtype=float)
    Y = work[task_cols_data].to_numpy(dtype=float)  # (N,M)
    M = Y.shape[1]

    # 6) Build Phi + feature names (must match how betas were produced)
    Phi, feat_names = _build_quadratic_scheffe_features(X, X_cols)
    P = Phi.shape[1]
    if verbose:
        print(f"[INFO] Phi shape: {Phi.shape} (P={P})")

    # 7) Read coefficients and intercept from betas_all.csv
    # Intercept row: feature contains "intercept" (case-insensitive) by default
    is_intercept = betas["feature"].astype(str).str.contains(
        intercept_feature_regex, case=False, na=False
    )
    intercept_row = betas.loc[is_intercept]
    features_rows = betas.loc[~is_intercept]

    if not intercept_row.empty:
        B0 = intercept_row[task_cols_betas].to_numpy(dtype=float)[0]  # (M,)
        if verbose:
            print(f"[INFO] Found Intercept: {B0}")
    else:
        B0 = np.zeros(M, dtype=float)
        if verbose:
            print("[WARN] No Intercept found! Predictions will be centered at 0.")

    # Feature coefficients map (P,M) aligning with feat_names
    bmap = features_rows.set_index("feature")[task_cols_betas]
    B = bmap.reindex(feat_names).to_numpy(dtype=float)  # (P,M)

    if np.isnan(B).any():
        missing = pd.isna(bmap.reindex(feat_names)).any(axis=1)
        missing_names = list(np.array(feat_names)[missing.values])
        raise ValueError(f"betas_all missing some features, e.g. {missing_names[:10]}")

    # 8) Predict, residual, V
    # Yhat = Phi @ B + B0
    Yhat = (Phi @ B) + B0[None, :]  # (N,M)
    E = Y - Yhat                    # (N,M)
    V_hat = (E.T @ E) / N           # (M,M)

    # 9) Print
    if verbose:
        print("\n[FORMULA]")
        print("Yhat = (Phi @ B) + B0")
        print("E = Y - Yhat")
        print("V_hat = (1/N) * (E.T @ E)")
        print(f"Shapes: Phi {Phi.shape}, B {B.shape}, B0 {B0.shape}, Y {Y.shape}, E {E.shape}")

        V_df = pd.DataFrame(V_hat, index=tasks, columns=tasks)
        print("\n[RESULT] V_hat (Raw Scale Residual Covariance)")
        print(V_df)

    # 10) Save (optional)
    if save_v_csv:
        V_df = pd.DataFrame(V_hat, index=tasks, columns=tasks)
        V_df.to_csv(v_csv_path, encoding=encoding)
        if verbose:
            print(f"\n[DONE] Wrote: {v_csv_path}")

    return V_hat

def _infer_tasks_from_betas_columns(betas: pd.DataFrame) -> List[str]:
    """Infer task columns from betas_all.csv by excluding metadata columns."""
    meta = {"feature", "type", "nonzero_row_l2", "active"}
    tasks = [c for c in betas.columns if c not in meta]
    if not tasks:
        raise ValueError("Cannot infer tasks from betas columns. Please pass tasks explicitly.")
    return tasks


def _safe_factor_psd(A: np.ndarray) -> np.ndarray:
    """
    Return a matrix L such that L @ L.T ~= A, even if A is only PSD (not strictly PD).
    Prefer Cholesky; fallback to eigen-decomposition with clipping.
    """
    A = np.asarray(A, dtype=float)
    try:
        return np.linalg.cholesky(A)
    except np.linalg.LinAlgError:
        w, Q = np.linalg.eigh(A)
        w = np.clip(w, 0.0, None)
        return Q @ np.diag(np.sqrt(w))


def _parse_feature_count_from_betas(betas: pd.DataFrame) -> int:
    return len(betas)


def _optional_check_feature_names_match_regex(
    betas: pd.DataFrame,
    x_regex: str = r"^[A-Za-z]{2}\d{3,4}$",
) -> None:
    """
    Optional sanity check: try to see if betas 'feature' names contain ingredient tokens
    matching the regex (e.g., x[AA001], x[AA001]*x[SS010]).
    This is NOT required for sampling; it's only a guardrail.
    """
    if "feature" not in betas.columns:
        return

    rx = re.compile(x_regex)
    feats = betas["feature"].astype(str).tolist()

    # Extract tokens inside x[...]
    tokens = []
    for f in feats:
        tokens += re.findall(r"x\[(.*?)\]", f)

    # If there are tokens but none match regex, warn.
    if tokens and not any(rx.match(t) for t in tokens):
        print(
            f"[WARN] Feature tokens found, but none matched x_regex={x_regex}. "
            "This may be OK if your naming scheme differs."
        )


def beta_generate(
    sample_n: int,
    beta_hat: str,
    task_correlation: np.ndarray,
    feature_correlation: Optional[np.ndarray] = None,
    *,
    # I/O
    out_dir: str = "./data",
    out_prefix: str = "beta",
    encoding: str = "utf-8-sig",
    seed: int = 42,
    # sampling options
    sample_intercept_correction: bool = True,
    intercept_type_keyword: str = "intercept",  # match in 'type' (case-insensitive)
    # tasks
    tasks: Optional[List[str]] = None,  # if None, infer from betas columns
    # active recompute
    thresh_active: float = 1e-10,
    # optional checks (incl. your regex requirement)
    x_regex: str = r"^[A-Za-z]{2}\d{3,4}$",
    check_feature_names: bool = False,
    verbose: bool = True,
) -> List[str]:
    """
    Sample B ~ MN(Bhat, U, V) where:
      - Bhat is read from beta_hat CSV (P x M)
      - V = task_correlation (M x M)
      - U = feature_correlation (P x P), default I

    Returns:
      List of written CSV paths.
    """

    if sample_n <= 0:
        raise ValueError("sample_n must be >= 1")

    # 1) Load betas_all.csv
    betas = pd.read_csv(beta_hat)
    if "feature" not in betas.columns or "type" not in betas.columns:
        raise ValueError("betas_all.csv must contain 'feature' and 'type' columns.")

    if tasks is None:
        tasks = _infer_tasks_from_betas_columns(betas)

    # check task columns exist
    for t in tasks:
        if t not in betas.columns:
            raise ValueError(f"betas_all.csv missing task column: {t}")

    P = _parse_feature_count_from_betas(betas)
    M = len(tasks)

    if verbose:
        print(f"[INFO] betas rows (P) = {P}")
        print(f"[INFO] tasks (M) = {M}: {tasks}")
        print("[INFO] type counts:\n", betas["type"].value_counts())

    if check_feature_names:
        _optional_check_feature_names_match_regex(betas, x_regex=x_regex)

    # 2) Build sample mask (which rows are sampled)
    type_lower = betas["type"].astype(str).str.lower()
    sample_mask = np.ones(P, dtype=bool)

    if not sample_intercept_correction:
        sample_mask &= ~type_lower.str.contains(intercept_type_keyword.lower())

    n_rows_to_sample = int(sample_mask.sum())
    if n_rows_to_sample == 0:
        raise ValueError("No rows selected to sample. Check sample_intercept_correction / intercept keyword.")

    if verbose:
        print(f"[INFO] rows to sample = {n_rows_to_sample}")

    # 3) Validate and factor correlations
    V = np.asarray(task_correlation, dtype=float)
    if V.shape != (M, M):
        raise ValueError(f"task_correlation shape must be ({M},{M}), got {V.shape}")

    # Feature correlation U
    if feature_correlation is None:
        U = np.eye(n_rows_to_sample, dtype=float)
        if verbose:
            print(f"[INFO] feature_correlation not provided -> default U = I_{n_rows_to_sample}")
    else:
        U = np.asarray(feature_correlation, dtype=float)
        if U.shape != (n_rows_to_sample, n_rows_to_sample):
            raise ValueError(
                f"feature_correlation must be shape ({n_rows_to_sample},{n_rows_to_sample}), got {U.shape}. "
                "Note: its dimension follows 'rows actually sampled' (after masking intercept if excluded)."
            )

    # Factor matrices such that:
    #   If Z ~ N(0, I) (n_rows_to_sample x M),
    #   then Noise = LU @ Z @ LV.T  has Cov(rows)=U, Cov(cols)=V
    LU = _safe_factor_psd(U)  # (n_rows_to_sample, n_rows_to_sample)
    LV = _safe_factor_psd(V)  # (M, M)

    # 4) Extract Bhat (P x M)
    Bhat = betas[tasks].to_numpy(dtype=float)  # (P,M)

    # 5) Sample and write CSVs
    rng = np.random.default_rng(seed)
    out_paths: List[str] = []
    out_dir_path = Path(out_dir)
    out_dir_path.mkdir(parents=True, exist_ok=True)

    # Indices of rows that are sampled
    idx = np.where(sample_mask)[0]

    for k in range(1, sample_n + 1):
        B = Bhat.copy()

        # Z ~ N(0, I)
        Z = rng.standard_normal((n_rows_to_sample, M))

        # correlated noise: (n_rows_to_sample x M)
        noise = (LU @ Z) @ LV.T

        # apply only to sampled rows
        B[idx, :] = Bhat[idx, :] + noise

        out = betas.copy()
        out.loc[:, tasks] = B

        # recompute row norms across tasks (for all rows)
        out["nonzero_row_l2"] = np.linalg.norm(B, axis=1)
        out["active"] = out["nonzero_row_l2"] > thresh_active

        out_path = out_dir_path / f"{out_prefix}{k}.csv"
        out.to_csv(out_path, index=False, encoding=encoding)
        out_paths.append(str(out_path))

        if verbose:
            print(f"[DONE] wrote {out_path}")

    return out_paths

if __name__ == "__main__":
    v = compute_V_task_correlation(
        BETAS_PATH="./data/betas_all.csv",
        DATA_PATH="./data/20260203_data_47d_spgr_te.csv",
    )
    print("\n[V as numpy array]")
    print(v)

    beta_generate(sample_n=5, beta_hat="./data/betas_all.csv", task_correlation=v)
