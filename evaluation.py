import os
import numpy as np


def mae(y_true, y_pred, mask):
    idx = mask == 1
    return np.mean(np.abs(y_true[idx] - y_pred[idx]))


def rmse(y_true, y_pred, mask):
    idx = mask == 1
    return np.sqrt(np.mean((y_true[idx] - y_pred[idx]) ** 2))


def mape(y_true, y_pred, mask, eps=1e-12):
    idx = mask == 1
    denom = np.maximum(np.abs(y_true[idx]), eps)
    return np.mean(np.abs(y_true[idx] - y_pred[idx]) / denom)


def main():

    result_file = "result/Result_point_missing_tau2_lam7_gt3_gs0.npy"

    if not os.path.exists(result_file):
        raise FileNotFoundError(f"Result file not found: {result_file}")

    data = np.load("./data/data.npy")
    mask = np.load("./data/mask_point_missing.npy")
    pred = np.load(result_file)

    print("Data shape :", data.shape)
    print("Mask shape :", mask.shape)
    print("Pred shape :", pred.shape)

    if data.shape != pred.shape:
        raise ValueError("Shape mismatch between data and prediction!")

    T, N = data.shape

    print("\n========== Overall Missing Evaluation ==========")

    overall_mae = mae(data, pred, mask)
    overall_rmse = rmse(data, pred, mask)
    overall_mape = mape(data, pred, mask)

    print(f"MAE  : {overall_mae:.6f}")
    print(f"RMSE : {overall_rmse:.6f}")
    print(f"MAPE : {overall_mape * 100:.3f}%")

    print("\n========== Per Variable Evaluation ==========")

    for i in range(N):
        var_mae = mae(data[:, i], pred[:, i], mask[:, i])
        var_rmse = rmse(data[:, i], pred[:, i], mask[:, i])
        var_mape = mape(data[:, i], pred[:, i], mask[:, i])

        print(
            f"Variable {i:02d} | "
            f"MAE: {var_mae:.6f}  "
            f"RMSE: {var_rmse:.6f}  "
            f"MAPE: {var_mape * 100:.3f}%"
        )


if __name__ == "__main__":
    main()