import time
from typing import Optional, Tuple
import os
import numpy as np
from sklearn.preprocessing import StandardScaler


def compute_mape(y, y_hat, eps: float = 1e-12) -> float:
    y = np.asarray(y).reshape(-1)
    y_hat = np.asarray(y_hat).reshape(-1)
    denom = np.maximum(np.abs(y), eps)
    return float(np.mean(np.abs(y - y_hat) / denom))


def compute_rmse(y, y_hat) -> float:
    y = np.asarray(y).reshape(-1)
    y_hat = np.asarray(y_hat).reshape(-1)
    return float(np.sqrt(np.mean((y - y_hat) ** 2)))


def laplacian_kernel_cyclic(T: int, tau: int) -> np.ndarray:
    ell = np.zeros(T, dtype=float)
    ell[0] = 2.0 * tau
    for k in range(1, tau + 1):
        ell[k] = -1.0
        ell[-k] = -1.0
    return ell


def apply_circulant_laplacian_time(X: np.ndarray, tau: int) -> np.ndarray:
    out = (2.0 * tau) * X
    for k in range(1, tau + 1):
        out = out - np.roll(X, shift=k, axis=1) - np.roll(X, shift=-k, axis=1)
    return out


def complex_soft_threshold(Z: np.ndarray, thresh: float, eps: float = 1e-12) -> np.ndarray:
    mag = np.abs(Z)
    scale = np.maximum(1.0 - thresh / (mag + eps), 0.0)
    return Z * scale


def prox_circulant_tensor_nuclear_norm(V: np.ndarray, thresh: float) -> np.ndarray:
    V_hat = np.fft.fft2(V, axes=(0, 1), norm="ortho")
    V_hat_shrink = complex_soft_threshold(V_hat, thresh)
    X = np.fft.ifft2(V_hat_shrink, axes=(0, 1), norm="ortho").real
    return X


def pearson_corr_ignore_nan(a: np.ndarray, b: np.ndarray, min_common: int = 5) -> float:
    mask = (~np.isnan(a)) & (~np.isnan(b))
    if np.sum(mask) < min_common:
        return 0.0

    aa = a[mask].astype(float)
    bb = b[mask].astype(float)

    aa = aa - aa.mean()
    bb = bb - bb.mean()

    sa = np.sqrt(np.mean(aa**2))
    sb = np.sqrt(np.mean(bb**2))
    if sa < 1e-12 or sb < 1e-12:
        return 0.0

    rho = float(np.mean(aa * bb) / (sa * sb))
    if np.isnan(rho) or np.isinf(rho):
        return 0.0
    return float(np.clip(rho, -1.0, 1.0))


def build_signed_laplacian(Y_obs: np.ndarray, min_common: int = 5) -> np.ndarray:
    N, _ = Y_obs.shape
    S = np.zeros((N, N), dtype=float)
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            S[i, j] = pearson_corr_ignore_nan(Y_obs[i], Y_obs[j], min_common=min_common)

    D = np.diag(np.sum(np.abs(S), axis=1))
    Ls = D - S
    Ls = 0.5 * (Ls + Ls.T)
    return Ls


def spectral_norm_symmetric(A: np.ndarray) -> float:
    vals = np.linalg.eigvalsh(A)
    return float(np.max(np.abs(vals)))


def solve_x_subproblem_pgm(
    X_init: np.ndarray,
    Z: np.ndarray,
    Wdual: np.ndarray,
    Ls: np.ndarray,
    tau_t: int,
    gamma_t: float,
    gamma_s: float,
    lam: float,
    inner_max_iter: int = 50,
    inner_tol: float = 1e-6,
) -> np.ndarray:
    X = X_init.copy()

    T = X.shape[1]
    ell_t = laplacian_kernel_cyclic(T, tau_t)

    eig_Lt = np.fft.fft(ell_t)
    abs_eig2 = (eig_Lt.conj() * eig_Lt).real
    LtLt_norm = float(np.max(abs_eig2))

    Ls_norm = spectral_norm_symmetric(Ls) if gamma_s > 0 else 0.0
    L_lip = gamma_t * LtLt_norm + 2.0 * gamma_s * Ls_norm + lam
    alpha = 1.0 / (L_lip + 1e-12)

    V = Z - Wdual / lam

    for _ in range(inner_max_iter):
        X_old = X

        X_hat = np.fft.fft(X, axis=1)
        if X.ndim == 2:
            Xt2 = np.fft.ifft(X_hat * abs_eig2[None, :], axis=1).real
        else:
            Xt2 = np.fft.ifft(X_hat * abs_eig2[None, :, None], axis=1).real
        grad_t = gamma_t * Xt2

        if gamma_s > 0:
            if X.ndim == 2:
                grad_s = 2.0 * gamma_s * (Ls @ X)
            else:
                grad_s = 2.0 * gamma_s * np.tensordot(Ls, X, axes=(1, 0))
        else:
            grad_s = 0.0

        grad_p = lam * (X - V)
        grad = grad_t + grad_s + grad_p

        X = prox_circulant_tensor_nuclear_norm(X - alpha * grad, thresh=alpha)

        denom = np.linalg.norm(X_old.ravel()) + 1e-12
        rel = np.linalg.norm((X - X_old).ravel()) / denom
        if rel < inner_tol:
            break

    return X

def admm_impute(
    Y: np.ndarray,
    tau_t: int = 2,
    gamma_t: float = 1.0,
    gamma_s: float = 1.0,
    lam: float = 10.0,
    eta: float = 1000.0,
    admm_max_iter: int = 200,
    inner_max_iter: int = 50,
    tol_primal: float = 1e-5,
    tol_dual: float = 1e-5,
    min_common_corr: int = 5,
    verbose: int = 1,
) -> np.ndarray:
    Y = np.array(Y, dtype=float)

    if Y.ndim == 2:
        Y2 = Y
        N, _ = Y2.shape
    elif Y.ndim == 3:
        N, T, L = Y.shape
        Y2 = Y.reshape(N, T * L)
    else:
        raise ValueError("Y must be 2D (N,T) or 3D (N,T,L)")

    if gamma_s > 0:
        Ls = build_signed_laplacian(Y2, min_common=min_common_corr)
    else:
        Ls = np.zeros((N, N), dtype=float)

    Y_filled = Y.copy()
    if Y.ndim == 2:
        for i in range(N):
            row = Y_filled[i]
            mm = np.nanmean(row)
            if np.isnan(mm):
                mm = 0.0
            row[np.isnan(row)] = mm
            Y_filled[i] = row
    else:
        for i in range(N):
            row = Y_filled[i].reshape(-1)
            mm = np.nanmean(row)
            if np.isnan(mm):
                mm = 0.0
            Y_filled[i][np.isnan(Y_filled[i])] = mm

    X = Y_filled.copy()
    Z = Y_filled.copy()
    Wdual = np.zeros_like(Y_filled)
    Y_obs = Y.copy()

    if verbose:
        print("ADMM start:")
        print(f"  shape: {Y.shape}, tau_t={tau_t}, gamma_t={gamma_t}, gamma_s={gamma_s}, lam={lam}, eta={eta}")
        if gamma_s > 0:
            print(f"  Ls spectral norm ~ {spectral_norm_symmetric(Ls):.6f}")

    for k in range(admm_max_iter):
        Z_prev = Z.copy()

        X = solve_x_subproblem_pgm(
            X_init=X,
            Z=Z,
            Wdual=Wdual,
            Ls=Ls,
            tau_t=tau_t,
            gamma_t=gamma_t,
            gamma_s=gamma_s,
            lam=lam,
            inner_max_iter=inner_max_iter,
            inner_tol=1e-6,
        )

        temp = lam * X + Wdual

        obs = ~np.isnan(Y_obs)
        Z = temp / lam
        Z[obs] = (temp[obs] + eta * Y_obs[obs]) / (lam + eta)

        Wdual = Wdual + lam * (X - Z)

        r = np.linalg.norm((X - Z).ravel())
        s = lam * np.linalg.norm((Z - Z_prev).ravel())
        Xn = np.linalg.norm(X.ravel()) + 1e-12
        Wn = np.linalg.norm(Wdual.ravel()) + 1e-12
        r_rel = r / Xn
        s_rel = s / Wn

        if verbose and ((k + 1) % max(1, verbose) == 0):
            print(f"Iter {k + 1:4d} | primal r={r_rel:.3e} | dual s={s_rel:.3e}")

        if r_rel < tol_primal and s_rel < tol_dual:
            if verbose:
                print(f"Converged at iter {k + 1}.")
            break

    return X


def main() -> None:
    data = np.load("./data/data.npy")
    mask_path = "./data/mask_contiguous_missing.npy"
    masks = np.load(mask_path)

    scenario = "point_missing" if "point" in mask_path else "contiguous_missing"

    print("data.shape:", data.shape)
    T, N = data.shape

    scaler = StandardScaler()
    data_norm = scaler.fit_transform(data)

    data_trans = data_norm.T

    data_masked = data_norm.copy()
    data_masked[masks == 1] = np.nan
    Y = data_masked.T

    Y_true = data_trans

    lam_factor = 0.0006313387185634842
    lam = lam_factor * N * T

    gamma_ratio_t = 0.46799479581839276
    gamma_t = gamma_ratio_t * lam

    gamma_s_ratio = 0.0073985368195960185
    gamma_s = gamma_s_ratio * lam

    tau_t = 2

    eta_factor = 102.1810726021865
    eta = eta_factor * lam

    admm_max_iter = 24
    inner_max_iter = 26

    print(f"Params: lam={lam:.3f}, gamma_t={gamma_t:.3f}, gamma_s={gamma_s:.3f}, eta={eta:.3f}, tau_t={tau_t}")

    start = time.time()
    X_hat = admm_impute(
        Y=Y,
        tau_t=tau_t,
        gamma_t=gamma_t,
        gamma_s=gamma_s,
        lam=lam,
        eta=eta,
        admm_max_iter=admm_max_iter,
        inner_max_iter=inner_max_iter,
        tol_primal=1e-5,
        tol_dual=1e-5,
        min_common_corr=5,
        verbose=1,
    )
    end = time.time()

    X_hat_denorm = scaler.inverse_transform(X_hat.T)

    save_dir = "./result"
    os.makedirs(save_dir, exist_ok=True)

    filename = (
        f"Result_{scenario}"
        f"_tau{tau_t}"
        f"_lam{lam:.0f}"
        f"_gt{gamma_t:.0f}"
        f"_gs{gamma_s:.0f}.npy"
    )

    save_path = os.path.join(save_dir, filename)

    np.save(save_path, X_hat_denorm)

    print(f"Saved: {save_path}")
    print(f"Running time: {end - start:.4f} seconds.")

    y_true_missing = data[masks == 1]
    y_hat_missing = X_hat_denorm[masks == 1]

    print(f"MAPE (missing): {compute_mape(y_true_missing, y_hat_missing):.6f}")
    print(f"RMSE (missing): {compute_rmse(y_true_missing, y_hat_missing):.6f}")


if __name__ == "__main__":
    main()