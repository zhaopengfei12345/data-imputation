# 需要额外安装：
# pip install optuna tqdm

import time
import numpy as np
import optuna
from tqdm.auto import tqdm
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

    eig_Lt = np.fft.fft(ell_t, norm="ortho")
    abs_eig2 = (eig_Lt.conj() * eig_Lt).real
    LtLt_norm = float(np.max(abs_eig2))

    Ls_norm = spectral_norm_symmetric(Ls) if gamma_s > 0 else 0.0
    L_lip = gamma_t * LtLt_norm + 2.0 * gamma_s * Ls_norm + lam
    alpha = 1.0 / (L_lip + 1e-12)

    V = Z - Wdual / lam

    for _ in range(inner_max_iter):
        X_old = X

        X_hat = np.fft.fft(X, axis=1, norm="ortho")
        if X.ndim == 2:
            Xt2 = np.fft.ifft(X_hat * abs_eig2[None, :], axis=1, norm="ortho").real
        else:
            Xt2 = np.fft.ifft(X_hat * abs_eig2[None, :, None], axis=1, norm="ortho").real
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


def _fill_nan_with_row_mean(Y: np.ndarray) -> np.ndarray:
    Y_filled = Y.copy()
    if Y.ndim == 2:
        N, _ = Y_filled.shape
        for i in range(N):
            row = Y_filled[i]
            mm = np.nanmean(row)
            if np.isnan(mm):
                mm = 0.0
            row[np.isnan(row)] = mm
            Y_filled[i] = row
    else:
        N = Y_filled.shape[0]
        for i in range(N):
            row_flat = Y_filled[i].reshape(-1)
            mm = np.nanmean(row_flat)
            if np.isnan(mm):
                mm = 0.0
            Y_filled[i][np.isnan(Y_filled[i])] = mm
    return Y_filled


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

    Y_filled = _fill_nan_with_row_mean(Y)

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
    data_path = "./data/data.npy"
    mask_path = "./data/mask_point_missing.npy"   # 你也可以换成 contiguous 的 mask

    n_trials = 500          # 想跑更多就改大
    seed = 42

    # 1) 读取数据（只读一次，避免每个 trial 重复 IO）
    data = np.load(data_path)          # shape: (T, N)
    masks = np.load(mask_path)         # shape: (T, N)；缺失位置=1
    T, N = data.shape

    # 2) 标准化并打 mask（与原 main 保持一致）
    scaler = StandardScaler()
    data_norm = scaler.fit_transform(data)

    data_masked = data_norm.copy()
    data_masked[masks == 1] = np.nan
    Y = data_masked.T  # (N, T)

    # 真实缺失值（用于算 MAPE，原尺度）
    y_true_missing = data[masks == 1]

    # 3) Optuna 设置（减少日志，只保留进度条）
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    sampler = optuna.samplers.TPESampler(seed=seed)
    pruner = optuna.pruners.NopPruner()  # 不剪枝，确保每次完整运行

    study = optuna.create_study(direction="minimize", sampler=sampler, pruner=pruner)

    # 4) 进度条
    pbar = tqdm(total=n_trials, desc="Optuna Trials", dynamic_ncols=True)

    def _cb(study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        # 更新进度
        pbar.update(1)

        # 如果已经有 best_value，就显示
        if study.best_trial is not None and study.best_value is not None:
            best_mape = study.best_value
            pbar.set_postfix({
                "best_mape": f"{best_mape:.6f}"
            })

    # 5) 目标函数：最小化缺失点 MAPE
    def objective(trial: optuna.trial.Trial) -> float:
        # —— 搜索空间（围绕你原来的超参做合理扩展）——
        lam_factor = trial.suggest_float("lam_factor", 1e-5, 5e-3, log=True)

        gamma_ratio_t = trial.suggest_float("gamma_ratio_t", 1e-2, 5.0, log=True)
        gamma_s_ratio = trial.suggest_float("gamma_s_ratio", 1e-4, 0.5, log=True)

        eta_factor = trial.suggest_float("eta_factor", 1.0, 5e3, log=True)

        tau_t = trial.suggest_int("tau_t", 1, 6)

        admm_max_iter = trial.suggest_int("admm_max_iter", 10, 80)
        inner_max_iter = trial.suggest_int("inner_max_iter", 10, 80)

        min_common_corr = trial.suggest_int("min_common_corr", 3, 20)

        # —— 按你原代码的参数关系计算 —— #
        lam = lam_factor * N * T
        gamma_t = gamma_ratio_t * lam
        gamma_s = gamma_s_ratio * lam
        eta = eta_factor * lam

        try:
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
                min_common_corr=min_common_corr,
                verbose=0,  # 不打印 ADMM 迭代信息，避免刷屏（进度条更干净）
            )

            # 反标准化回原尺度：(N,T) -> (T,N)
            X_hat_denorm = scaler.inverse_transform(X_hat.T)
            y_hat_missing = X_hat_denorm[masks == 1]

            mape = compute_mape(y_true_missing, y_hat_missing)
            if not np.isfinite(mape):
                return float("inf")
            return float(mape)

        except Exception:
            # 单个 trial 崩了就给一个极大值，继续下一个
            return float("inf")

    # 6) 开跑（不会因为 trial 报错停止）
    study.optimize(
        objective,
        n_trials=n_trials,
        callbacks=[_cb],
        catch=(Exception,),  # 双保险：即使异常没被 objective 吞掉也继续
    )

    pbar.close()

    # 7) 最终只打印最优参数（按你的要求）
    print(study.best_params)


if __name__ == "__main__":
    main()