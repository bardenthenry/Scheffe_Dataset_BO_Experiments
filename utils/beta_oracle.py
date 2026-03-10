import numpy as np
import pandas as pd
import torch
import re
from typing import Dict, List, Optional, Union


def _infer_target_cols_from_beta_csv(df: pd.DataFrame) -> List[str]:
    """
    target cols are all columns except metadata columns.
    Works for 2+ tasks (SPGR/TE/TS/...)
    """
    meta = {"feature", "type", "nonzero_row_l2", "active"}
    targets = [c for c in df.columns if c not in meta]
    if not targets:
        raise ValueError("Cannot infer TARGET_COLS from beta CSV columns.")
    return targets


def _infer_feature_cols_from_beta_features(
    df: pd.DataFrame,
    x_regex: str = r"^[A-Za-z]{2}\d{3,4}$",
) -> List[str]:
    """
    Infer FEATURE_COLS by parsing linear features like x[AA001]
    and filtering tokens by x_regex, then sort them by (prefix, numeric).
    """
    if "feature" not in df.columns:
        raise ValueError("beta CSV must contain a 'feature' column.")

    rx = re.compile(x_regex)

    # only linear rows are the cleanest source of feature list
    lin = df[df["type"].astype(str).str.lower().eq("linear")].copy()
    if lin.empty:
        raise ValueError("No linear rows found in beta CSV; cannot infer FEATURE_COLS.")

    mats = lin["feature"].astype(str).str.extract(r"x\[(.+?)\]")[0].dropna().tolist()
    mats = [m.strip() for m in mats]
    mats = [m for m in mats if rx.match(m)]

    if not mats:
        raise ValueError(
            f"Parsed linear materials, but none matched x_regex={x_regex}. "
            "Check your feature naming scheme."
        )

    mats = sorted(set(mats), key=lambda s: (s[:2].upper(), int(re.findall(r"\d+", s)[0])))
    return mats


def load_beta_oracle(
    beta_csv_path: str,
    *,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float64,
    # inference rules
    x_regex: str = r"^[A-Za-z]{2}\d{3,4}$",
    # behavior
    keep_only_active: bool = True,
) -> Dict[str, torch.Tensor]:
    """
    Read beta*.csv and build a fast oracle.

    Infer:
      - feature_cols from linear feature names x[...], filtered by x_regex
      - target_cols from csv columns (excluding metadata columns)

    Returns a dict with tensors on the given device/dtype:
      - intercept: (m,)
      - beta_lin: (m, q)
      - pairs: (K, 2)
      - beta_inter: (m, K)
      - feature_cols: List[str]
      - target_cols: List[str]
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    df = pd.read_csv(beta_csv_path)

    # optionally keep only active coefficients
    if keep_only_active and "active" in df.columns:
        df = df[df["active"].astype(bool)].copy()

    # infer columns
    target_cols = _infer_target_cols_from_beta_csv(df)
    feature_cols = _infer_feature_cols_from_beta_features(df, x_regex=x_regex)

    m = len(target_cols)
    q = len(feature_cols)

    # ---- intercept ----
    inter = df[df["type"].astype(str).str.lower().eq("intercept_correction")]
    if inter.empty:
        raise ValueError("No row with type == 'intercept_correction' found in beta CSV.")

    intercept = torch.tensor(
        inter.iloc[0][target_cols].values.astype(float),
        device=device,
        dtype=dtype,
    )  # (m,)

    # map material -> index
    mat_to_idx = {mat: i for i, mat in enumerate(feature_cols)}

    # ---- linear terms ----
    lin_df = df[df["type"].astype(str).str.lower().eq("linear")].copy()
    lin_df["mat"] = lin_df["feature"].astype(str).str.extract(r"x\[(.+?)\]")[0]
    lin_df["idx"] = lin_df["mat"].map(mat_to_idx)

    if lin_df["idx"].isna().any():
        missing = lin_df[lin_df["idx"].isna()]["mat"].dropna().unique().tolist()
        raise ValueError(f"Linear beta has unknown materials: {missing}")

    beta_lin = torch.zeros((m, q), device=device, dtype=dtype)
    for _, row in lin_df.iterrows():
        j = int(row["idx"])
        beta_lin[:, j] = torch.tensor(
            row[target_cols].values.astype(float),
            device=device,
            dtype=dtype,
        )

    # ---- interaction terms ----
    int_df = df[df["type"].astype(str).str.lower().eq("interaction")].copy()
    if not int_df.empty:
        mats = int_df["feature"].astype(str).str.extract(r"x\[(.+?)\]\*x\[(.+?)\]")
        int_df["mat_i"] = mats[0]
        int_df["mat_j"] = mats[1]
        int_df["i"] = int_df["mat_i"].map(mat_to_idx)
        int_df["j"] = int_df["mat_j"].map(mat_to_idx)

        if int_df[["i", "j"]].isna().any().any():
            missing_i = int_df[int_df["i"].isna()]["mat_i"].dropna().unique().tolist()
            missing_j = int_df[int_df["j"].isna()]["mat_j"].dropna().unique().tolist()
            raise ValueError(f"Interaction beta has unknown materials: {missing_i + missing_j}")

        pairs = torch.tensor(
            int_df[["i", "j"]].values.astype(int),
            device=device,
            dtype=torch.long,
        )  # (K,2)

        beta_inter = torch.tensor(
            int_df[target_cols].values.astype(float),
            device=device,
            dtype=dtype,
        ).T  # (m,K)
    else:
        # allow "no interaction" case
        pairs = torch.empty((0, 2), device=device, dtype=torch.long)
        beta_inter = torch.empty((m, 0), device=device, dtype=dtype)

    return {
        "intercept": intercept,      # (m,)
        "beta_lin": beta_lin,        # (m,q)
        "pairs": pairs,              # (K,2)
        "beta_inter": beta_inter,    # (m,K)
        "feature_cols": feature_cols,
        "target_cols": target_cols,
        "device": device,
        "dtype": dtype,
    }


def noise_variance(
    data: str,
    target_cols: List[str],
    *,
    ddof: int = 0,
    verbose: bool = True,
) -> np.ndarray:
    """
    Estimate per-task noise variance sigma^2 from data CSV.

    Current implementation:
      sigma^2_k = Var(Y_k) computed from the column itself after dropna.

    Returns:
      np.ndarray shape (m,) in the same order as target_cols
    """
    df = pd.read_csv(data)
    cols = list(df.columns)

    def find_col_exact_or_fuzzy(df_columns, name):
        if name in df_columns:
            return name
        low = {c.lower(): c for c in df_columns}
        if name.lower() in low:
            return low[name.lower()]
        hits = [c for c in df_columns if c.lower() == name.lower() or c.lower().endswith("_" + name.lower())]
        return hits[0] if len(hits) == 1 else None

    resolved = []
    for t in target_cols:
        c = find_col_exact_or_fuzzy(cols, t)
        if c is None:
            raise ValueError(f"Cannot find target col '{t}' in data CSV.")
        resolved.append(c)

    work = df[resolved].copy()
    for c in resolved:
        work[c] = pd.to_numeric(work[c], errors="coerce")
    work = work.dropna(subset=resolved).reset_index(drop=True)

    if len(work) == 0:
        raise ValueError("No valid rows after dropna for target cols.")

    # variance per column
    var = work.var(axis=0, ddof=ddof).to_numpy(dtype=float)  # (m,)
    if verbose:
        std = np.sqrt(var)
        print("[INFO] noise_variance sigma^2 (per task):")
        for name, v, s in zip(target_cols, var, std):
            print(f"  - {name}: var={v:.6g}, std={s:.6g}")
    return var


def oracle_eval(
    X_tensor: torch.Tensor,
    oracle: Dict[str, torch.Tensor],
    *,
    noisy: bool = False,
    noise_variance: Optional[Union[float, np.ndarray, torch.Tensor, List[float]]] = None,
) -> torch.Tensor:
    """
    X_tensor: (n, q) in fraction scale (0~1), sum(x)=1
    return:   (n, m) aligned with oracle["target_cols"]

    If noisy=True:
      add epsilon ~ N(0, sigma^2)
      - noise_variance must be provided
      - can be:
          * scalar (same sigma^2 for all tasks)
          * array-like length m (one sigma^2 per task)
    """
    # device = torch.get_default_device()
    # dtype = torch.get_default_dtype()
    # print(device)

    intercept = oracle["intercept"]      # (m,)
    beta_lin = oracle["beta_lin"]        # (m,q)
    pairs = oracle["pairs"]              # (K,2)
    beta_inter = oracle["beta_inter"]    # (m,K)

    # linear term: (n,q) @ (q,m) -> (n,m)
    lin_term = X_tensor @ beta_lin.T

    # interaction term
    if pairs.numel() == 0:
        inter_term = torch.zeros((X_tensor.shape[0], intercept.shape[0]),
                                 device=X_tensor.device, dtype=X_tensor.dtype)
    else:
        cross = X_tensor[:, pairs[:, 0]] * X_tensor[:, pairs[:, 1]]  # (n,K)
        inter_term = cross @ beta_inter.T  # (n,m)

    Y = intercept.unsqueeze(0) + lin_term + inter_term  # (n,m)

    if not noisy:
        return Y

    # ---- noisy path ----
    if noise_variance is None:
        raise ValueError("When noisy=True, noise_variance must be provided.")

    m = Y.shape[1]
    n = Y.shape[0]
    dev = Y.device
    dt = Y.dtype

    # Convert noise_variance to tensor of shape (m,)
    if isinstance(noise_variance, (float, int)):
        var_t = torch.full((m,), float(noise_variance), device=dev, dtype=dt)
    elif isinstance(noise_variance, torch.Tensor):
        var_t = noise_variance.to(device=dev, dtype=dt).flatten()
    else:
        var_t = torch.tensor(np.array(noise_variance, dtype=float), device=dev, dtype=dt).flatten()

    if var_t.numel() == 1:
        var_t = var_t.repeat(m)

    if var_t.numel() != m:
        raise ValueError(
            f"noise_variance must be scalar or length m={m}. Got length {var_t.numel()}."
        )

    if torch.any(var_t < 0):
        raise ValueError("noise_variance contains negative values; sigma^2 must be >= 0.")

    sigma = torch.sqrt(var_t)  # (m,)
    eps = torch.randn((n, m), device=dev, dtype=dt) * sigma.unsqueeze(0)  # (n,m)

    return Y + eps

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float64

    oracle = load_beta_oracle("./data/beta1.csv", device=device, dtype=dtype)

    # X_frac (n,q)
    q = len(oracle["feature_cols"])
    X_frac = torch.distributions.Dirichlet(torch.ones(q, device=device, dtype=dtype)).sample((100,))

    # noise-free eval
    Y = oracle_eval(X_frac, oracle)

    # 算每個 task 的 sigma^2
    var = noise_variance(
        data="./data/20260203_data_47d_spgr_te.csv",
        target_cols=oracle["target_cols"],
    )

    # noisy eval
    Y_noisy = oracle_eval(X_frac, oracle, noisy=True, noise_variance=var)