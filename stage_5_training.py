# noqa: D100
"""Stage 5 — Training profit-aware, calibration, regime models, meta-filter."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import TimeSeriesSplit

from utils.experiment import write_json
from utils.logging_config import stage_logger
from utils.paths import ensure_dir
from utils.regime import (
    REGIME_HIGH_VOL,
    REGIME_RANGE,
    REGIME_TREND_DOWN,
    REGIME_TREND_UP,
    detect_regime,
)

TARGET = "target"
REGIMES = [REGIME_TREND_UP, REGIME_TREND_DOWN, REGIME_RANGE, REGIME_HIGH_VOL]
PROTECTED_FEATURES = {"daily_range_pos", "rsi_zscore", "bar_momentum", "hurst_proxy"}


class TemperatureScaling:
    """Temperature scaling per-class berbasis logits aproksimasi dari probabilitas."""

    def __init__(self, temperatures: np.ndarray):
        self.temperatures = np.asarray(temperatures, dtype=float)

    @staticmethod
    def _probs_to_logits(probs: np.ndarray) -> np.ndarray:
        p = np.clip(probs, 1e-7, 1 - 1e-7)
        logits = np.log(p / (1.0 - p))
        return logits

    @staticmethod
    def _softmax(z: np.ndarray) -> np.ndarray:
        z = z - np.max(z, axis=1, keepdims=True)
        ez = np.exp(z)
        return ez / (np.sum(ez, axis=1, keepdims=True) + 1e-12)

    def transform(self, probs: np.ndarray) -> np.ndarray:
        logits = self._probs_to_logits(probs)
        t = np.clip(self.temperatures.reshape(1, -1), 0.05, 10.0)
        return self._softmax(logits / t)


class CalibratedModelWrapper:
    """Wrapper model dengan interface predict_proba tetap."""

    def __init__(self, base_model: Any, mode: str, calibrator: Any = None, temp_scaler: TemperatureScaling | None = None):
        self.base_model = base_model
        self.mode = mode
        self.calibrator = calibrator
        self.temp_scaler = temp_scaler

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        probs = self.base_model.predict_proba(X)
        if self.mode == "isotonic" and self.calibrator is not None:
            return self.calibrator.predict_proba(X)
        if self.mode == "temperature" and self.temp_scaler is not None:
            return self.temp_scaler.transform(probs)
        return probs


def _create_xgb_classifier(**params: Any) -> xgb.XGBClassifier:
    """Buat XGBClassifier; buang parameter tidak dikenal jika versi XGB berbeda."""
    p = dict(params)
    while p:
        try:
            return xgb.XGBClassifier(**p)
        except TypeError as exc:
            msg = str(exc)
            unknown = None
            if "unexpected keyword argument" in msg and "'" in msg:
                unknown = msg.split("'")[1]
            if unknown and unknown in p:
                p.pop(unknown)
                continue
            raise
    raise ValueError("Tidak ada parameter valid untuk XGBClassifier")


def _fit_xgb_cv_early_stop(
    clf: xgb.XGBClassifier,
    X_tr: pd.DataFrame,
    yt: np.ndarray,
    X_va: pd.DataFrame,
    yv: np.ndarray,
    sample_weight_tr: np.ndarray,
    es_rounds: int,
) -> None:
    fit_kw: Dict[str, Any] = {
        "X": X_tr,
        "y": yt,
        "sample_weight": sample_weight_tr,
        "eval_set": [(X_va, yv)],
        "verbose": False,
    }
    try:
        clf.fit(**fit_kw, early_stopping_rounds=es_rounds)
        return
    except TypeError:
        pass
    try:
        from xgboost.callback import EarlyStopping

        clf.fit(**fit_kw, callbacks=[EarlyStopping(rounds=es_rounds)])
    except (ImportError, TypeError, ValueError):
        clf.fit(**fit_kw)


def _tail(df: pd.DataFrame, frac: float, max_rows: int) -> pd.DataFrame:
    if frac < 1.0:
        n = max(int(len(df) * frac), 5000)
        df = df.iloc[-n:].copy()
    if len(df) > max_rows:
        df = df.iloc[-max_rows:].copy()
    return df.reset_index(drop=True)


def _feature_columns(df: pd.DataFrame) -> List[str]:
    exclude = {
        "time",
        TARGET,
        "target_up_binary",
        "target_down_binary",
        "forward_log_return",
        "tp_sl_outcome",
        "mfe_long",
        "mae_long",
        "mfe_short",
        "mae_short",
        "keep_for_training",
        "gap_before",
        "spread_spike",
        "session",
        "vol_regime",
        "flat_return_threshold",
        "horizon_bars",
        "label_atr_window",
        "label_sl_atr_multiplier",
        "label_tp_rr",
        "regime",
    }
    price_cols = {"open", "high", "low", "close", "spread"}
    feats: List[str] = []
    for c in df.columns:
        if c in exclude or c in price_cols:
            continue
        s = df[c]
        if pd.api.types.is_datetime64_any_dtype(s) or s.dtype == object:
            continue
        if pd.api.types.is_bool_dtype(s) or pd.api.types.is_numeric_dtype(s):
            feats.append(c)
    return feats


def _map_folds(
    fold_path: Path,
    df_ref: pd.DataFrame,
    df_opt: pd.DataFrame,
) -> Optional[List[Tuple[np.ndarray, np.ndarray]]]:
    if not fold_path.is_file():
        return None
    rows = json.loads(fold_path.read_text(encoding="utf-8"))
    off = len(df_ref) - len(df_opt)
    out: List[Tuple[np.ndarray, np.ndarray]] = []
    for fold in rows:
        tr = np.asarray(fold["train_indices"], dtype=int)
        te = np.asarray(fold["test_indices"], dtype=int)
        tr = tr[tr >= off] - off
        te = te[te >= off] - off
        if len(tr) < 400 or len(te) < 40:
            continue
        out.append((tr, te))
    return out or None


def _row_sample_weights(y: np.ndarray, mode: str, *, minority_boost: float = 1.0) -> np.ndarray:
    y = y.astype(int)
    n = len(y)
    if mode == "none" or n == 0:
        return np.ones(n, dtype=np.float32)
    cnt = np.bincount(y, minlength=3).astype(np.float64)
    cnt = np.maximum(cnt, 1.0)
    inv = (n / (3.0 * cnt))[y]
    if mode == "inverse_sqrt":
        inv = np.sqrt(inv)
    elif mode != "inverse":
        raise ValueError(f"stage_5.class_balance tidak dikenal: {mode}")
    if minority_boost > 1.0:
        inv = np.where(y != 0, inv * float(minority_boost), inv)
    inv = inv / (inv.mean() + 1e-12)
    return inv.astype(np.float32)


def _simulate_trade_metrics(
    probs: np.ndarray,
    y_true: np.ndarray,
    y_outcome: Optional[np.ndarray],
    *,
    conf_up: float,
    conf_down: float,
    abstain_tau: float,
    rr: float = 2.0,
) -> Dict[str, float]:
    p0, p1, p2 = probs[:, 0], probs[:, 1], probs[:, 2]
    pred = np.zeros(len(probs), dtype=np.int8)
    long_m = (p1 >= conf_up) & (p0 < abstain_tau) & (p1 >= p2)
    short_m = (p2 >= conf_down) & (p0 < abstain_tau) & (p2 > p1)
    pred[long_m] = 1
    pred[short_m] = 2
    dir_m = pred != 0
    n = int(dir_m.sum())
    if n == 0:
        return {
            "n_trades": 0,
            "winrate": 0.0,
            "average_R": 0.0,
            "expectancy": -1.0,
            "max_drawdown_R": 0.0,
            "sharpe_R": -1.0,
            "precision_dir": 0.0,
        }

    if y_outcome is not None:
        out = y_outcome[dir_m]
        rets = np.where(out == 1, rr, np.where(out == -1, -1.0, 0.0)).astype(float)
    else:
        yt = y_true[dir_m]
        yp = pred[dir_m]
        hit = ((yt == 1) & (yp == 1)) | ((yt == 2) & (yp == 2))
        rets = np.where(hit, rr, -1.0).astype(float)

    cum = np.cumsum(rets)
    peak = np.maximum.accumulate(cum)
    dd = peak - cum
    max_dd = float(dd.max()) if len(dd) else 0.0
    avg_r = float(np.mean(rets))
    std_r = float(np.std(rets) + 1e-9)
    sharpe = float(avg_r / std_r * np.sqrt(max(len(rets), 1)))
    winrate = float((rets > 0).mean())
    prec = float((((y_true[dir_m] == 1) & (pred[dir_m] == 1)) | ((y_true[dir_m] == 2) & (pred[dir_m] == 2))).mean())
    return {
        "n_trades": float(n),
        "winrate": winrate,
        "average_R": avg_r,
        "expectancy": avg_r,
        "max_drawdown_R": max_dd,
        "sharpe_R": sharpe,
        "precision_dir": prec,
    }


def _simulate_pred_labels(
    probs: np.ndarray,
    *,
    conf_up: float,
    conf_down: float,
    abstain_tau: float,
) -> np.ndarray:
    p0, p1, p2 = probs[:, 0], probs[:, 1], probs[:, 2]
    pred = np.zeros(len(probs), dtype=np.int8)
    long_m = (p1 >= conf_up) & (p0 < abstain_tau) & (p1 >= p2)
    short_m = (p2 >= conf_down) & (p0 < abstain_tau) & (p2 > p1)
    pred[long_m] = 1
    pred[short_m] = 2
    return pred


def _expected_calibration_error(y_true: np.ndarray, probs: np.ndarray, n_bins: int = 10) -> float:
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    acc = (pred == y_true).astype(float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        m = (conf >= bins[i]) & (conf < bins[i + 1])
        if not np.any(m):
            continue
        ece += float(np.abs(acc[m].mean() - conf[m].mean()) * m.mean())
    return float(ece)


def _maximum_calibration_error(y_true: np.ndarray, probs: np.ndarray, n_bins: int = 10) -> float:
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    acc = (pred == y_true).astype(float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    mce = 0.0
    for i in range(n_bins):
        m = (conf >= bins[i]) & (conf < bins[i + 1])
        if not np.any(m):
            continue
        mce = max(mce, float(np.abs(acc[m].mean() - conf[m].mean())))
    return float(mce)


def _brier_per_class(y_true: np.ndarray, probs: np.ndarray, n_classes: int = 3) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for k in range(n_classes):
        yk = (y_true == k).astype(float)
        out[f"class_{k}"] = float(np.mean((probs[:, k] - yk) ** 2))
    return out


def _fit_temperature_scaling(probs_val: np.ndarray, y_val: np.ndarray) -> TemperatureScaling:
    # Coordinate search sederhana per class untuk minimasi NLL.
    t = np.ones(probs_val.shape[1], dtype=float)

    def nll(temp: np.ndarray) -> float:
        ts = TemperatureScaling(temp)
        p = np.clip(ts.transform(probs_val), 1e-9, 1 - 1e-9)
        return float(-np.mean(np.log(p[np.arange(len(y_val)), y_val])))

    best = nll(t)
    grid = np.linspace(0.5, 3.0, 26)
    for _ in range(3):
        improved = False
        for i in range(len(t)):
            local_best_t = t[i]
            local_best = best
            for g in grid:
                cand = t.copy()
                cand[i] = float(g)
                s = nll(cand)
                if s < local_best:
                    local_best = s
                    local_best_t = float(g)
            if local_best < best:
                t[i] = local_best_t
                best = local_best
                improved = True
        if not improved:
            break
    return TemperatureScaling(t)


def _save_reliability_diagram(
    y_true: np.ndarray,
    probs_dict: Dict[str, np.ndarray],
    out_path: Path,
) -> Dict[str, Dict[str, float]]:
    # Fokus directional confidence (UP+DOWN) agar relevan trading.
    y_bin = (y_true != 0).astype(int)
    plt.figure(figsize=(6, 5))
    plt.plot([0, 1], [0, 1], "k--", linewidth=1)
    metrics: Dict[str, Dict[str, float]] = {}
    for name, probs in probs_dict.items():
        p = 1.0 - probs[:, 0]
        frac, mean = calibration_curve(y_bin, p, n_bins=10, strategy="uniform")
        plt.plot(mean, frac, marker="o", label=name)
        metrics[name] = {
            "ece": _expected_calibration_error(y_true, probs),
            "mce": _maximum_calibration_error(y_true, probs),
            "brier": _brier_per_class(y_true, probs),
        }
    plt.title("Reliability Diagram (Directional)")
    plt.xlabel("Mean Predicted Probability")
    plt.ylabel("Observed Frequency")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=140)
    plt.close()
    return metrics


def _xgb_params_from_bundle(bundle_path: Path, s5: Dict[str, Any]) -> Dict[str, Any]:
    """Ambil hyperparameter terbaik dari bundle model utama sebagai starting point."""
    params: Dict[str, Any] = {
        "n_estimators": int(s5.get("final_n_estimators", 400)),
        "max_depth": 4,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 10,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "random_state": 42,
        "eval_metric": "auc",
        "objective": "binary:logistic",
        "tree_method": "hist",
    }
    if bundle_path.is_file():
        try:
            bundle = joblib.load(bundle_path)
            bp = bundle.get("optuna", {}).get("xgb", {}).get("best_params", {})
            if isinstance(bp, dict):
                for k in ("max_depth", "learning_rate", "subsample", "colsample_bytree", "min_child_weight", "gamma", "reg_alpha", "reg_lambda", "n_estimators"):
                    if k in params and bp[k] is not None:
                        params[k] = bp[k]
        except Exception:
            pass
    return params


def train_binary_models(
    df_train: pd.DataFrame,
    features: List[str],
    cfg: Dict[str, Any],
    run_dir: Path,
    log,
) -> Dict[str, Any]:
    """Train dua binary XGBoost: UP-vs-rest dan DOWN-vs-rest."""
    s5 = cfg.get("stage_5", {})
    bundle_path = run_dir / "stage_5" / "xgb_model.joblib"
    xgb_params = _xgb_params_from_bundle(bundle_path, s5)

    leak_cols = {"target", "target_up_binary", "target_down_binary"}
    features = [f for f in features if f not in leak_cols]
    X = df_train[features].astype(np.float32).values
    results: Dict[str, Any] = {}

    for side, col in [("up", "target_up_binary"), ("down", "target_down_binary")]:
        if col not in df_train.columns:
            log.warning("Binary training: kolom %s tidak ada, skip.", col)
            continue

        y = df_train[col].to_numpy(dtype=np.int8)
        n_pos = int(y.sum())
        n_neg = int((y == 0).sum())
        if n_pos < 50:
            log.warning("Binary %s: terlalu sedikit positive samples (%d), skip.", side, n_pos)
            continue

        scale_pw = max(1.0, n_neg / max(n_pos, 1))
        params = {**xgb_params, "scale_pos_weight": scale_pw}

        split = int(len(X) * 0.80)
        split = min(max(split, 500), len(X) - max(120, int(len(X) * 0.08)))
        X_tr, X_ho = X[:split], X[split:]
        y_tr, y_ho = y[:split], y[split:]

        model = _create_xgb_classifier(**params)
        model.fit(
            X_tr,
            y_tr,
            eval_set=[(X_ho, y_ho)],
            verbose=False,
        )

        prob_ho = model.predict_proba(X_ho)[:, 1]
        best_thresh = 0.50
        best_f1 = 0.0
        for thr in np.arange(0.35, 0.80, 0.02):
            pred_thr = (prob_ho >= thr).astype(int)
            if pred_thr.sum() < 5:
                continue
            f1 = float(f1_score(y_ho, pred_thr, zero_division=0))
            if f1 > best_f1:
                best_f1 = f1
                best_thresh = float(thr)

        pred_ho = (prob_ho >= best_thresh).astype(int)
        prec = float(precision_score(y_ho, pred_ho, zero_division=0))
        rec = float(recall_score(y_ho, pred_ho, zero_division=0))
        auc = float(roc_auc_score(y_ho, prob_ho)) if len(np.unique(y_ho)) > 1 else 0.0
        signal_rate = float(pred_ho.mean())

        log.info(
            "Binary %s | threshold=%.3f | precision=%.4f | recall=%.4f | AUC=%.4f | signal_rate=%.3f",
            side,
            best_thresh,
            prec,
            rec,
            auc,
            signal_rate,
        )

        out_path = run_dir / "stage_5" / f"xgb_binary_{side}.joblib"
        joblib.dump(
            {
                "model": model,
                "features": features,
                "threshold": best_thresh,
                "side": side,
                "params": params,
                "holdout_metrics": {
                    "precision": prec,
                    "recall": rec,
                    "auc_roc": auc,
                    "f1": best_f1,
                    "signal_rate": signal_rate,
                    "n_positive_holdout": int(y_ho.sum()),
                    "n_total_holdout": int(len(y_ho)),
                },
            },
            out_path,
        )
        results[side] = {
            "threshold": best_thresh,
            "precision": prec,
            "recall": rec,
            "auc_roc": auc,
            "f1": best_f1,
            "signal_rate": signal_rate,
            "n_positive_holdout": int(y_ho.sum()),
            "n_total_holdout": int(len(y_ho)),
        }
        log.info("Binary %s model disimpan ke %s", side, out_path)

    return results


def _fit_calibrator_prefit(clf: xgb.XGBClassifier, X_cal: pd.DataFrame, y_cal: np.ndarray):
    import sklearn
    from packaging import version

    sk_ver = version.parse(sklearn.__version__)
    if sk_ver >= version.parse("1.4.0"):
        try:
            from sklearn.frozen import FrozenEstimator

            cal = CalibratedClassifierCV(FrozenEstimator(clf), method="isotonic")
        except ImportError:
            cal = CalibratedClassifierCV(estimator=clf, method="isotonic", cv=3)
    else:
        try:
            cal = CalibratedClassifierCV(estimator=clf, method="isotonic", cv="prefit")
        except (TypeError, ValueError):
            cal = CalibratedClassifierCV(base_estimator=clf, method="isotonic", cv="prefit")
    cal.fit(X_cal, y_cal)
    return cal


def run_stage_5(
    df_in: pd.DataFrame,
    cfg: Dict[str, Any],
    *,
    run_dir: Path,
    task_flags: Optional[Dict[str, bool]] = None,
) -> Dict[str, Any]:
    flags = task_flags or {}
    out = ensure_dir(run_dir / "stage_5")
    log = stage_logger("stage_5_training", out)
    s5 = cfg["stage_5"]

    if str(s5.get("model_type", "xgboost")).lower() != "xgboost":
        raise ValueError("Pipeline ini hanya mendukung stage_5.model_type: xgboost")

    df = df_in.copy()
    if "keep_for_training" in df.columns:
        df = df[df["keep_for_training"] == 1].reset_index(drop=True)
    df["regime"] = detect_regime(df, cfg.get("stage_5", {}).get("regime_detection", {}))
    df_ref = df.copy()
    df_opt = _tail(df_ref, float(s5["optuna_data_frac"]), int(s5["optuna_max_rows"]))

    feats_raw = _feature_columns(df_opt)
    X_all = df_opt[feats_raw].astype(np.float32)
    y = df_opt[TARGET].astype(int).to_numpy()
    y_outcome = df_opt["tp_sl_outcome"].to_numpy(dtype=np.int8) if "tp_sl_outcome" in df_opt.columns else None
    if not np.isin(y, [0, 1, 2]).all():
        bad = int((~np.isin(y, [0, 1, 2])).sum())
        raise ValueError(f"Stage5 mengharapkan target 0=FLAT, 1=UP, 2=DOWN; baris lain: {bad}")

    fold_path = run_dir / "stage_4" / "stage_4_fold_indices.json"
    cv = _map_folds(fold_path, df_ref, df_opt)
    cv_source = "purged_stage4" if cv else "TimeSeriesSplit"
    if cv is None:
        cv = list(TimeSeriesSplit(n_splits=int(s5["n_cv_splits"])).split(X_all))

    cb_mode = str(s5.get("class_balance", "inverse")).strip().lower()
    class_share = np.bincount(y, minlength=3).astype(float) / max(len(y), 1)
    flat_share = float(class_share[0]) if len(class_share) else 0.0
    minority_boost = 1.0
    if bool(s5.get("dynamic_minority_boost_enabled", True)) and flat_share > 0.50:
        minority_boost = max(1.0, float(s5.get("dynamic_minority_boost_factor", 1.25)))
    rr_for_obj = float(cfg.get("stage_2", {}).get("risk_labeling", {}).get("tp_rr", 2.0))

    sim_conf_up = float(s5.get("optuna_sim_conf_up", 0.52))
    sim_conf_down = float(s5.get("optuna_sim_conf_down", 0.52))
    sim_abstain_tau = float(s5.get("optuna_sim_abstain_tau", 0.42))
    min_trades_frac = float(s5.get("optuna_min_trades_frac", 0.05))
    w_exp = float(s5.get("optuna_weight_expectancy", 0.35))
    w_sharpe = float(s5.get("optuna_weight_sharpe", 0.20))
    w_prec = float(s5.get("optuna_weight_prec_dir", 0.25))
    w_f1 = float(s5.get("optuna_weight_f1_dir", 0.20))

    def objective(trial: optuna.Trial) -> float:
        params = {
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.1),
            "subsample": trial.suggest_float("subsample", 0.6, 0.9),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 0.9),
            "min_child_weight": trial.suggest_int("min_child_weight", 3, 24),
            "gamma": trial.suggest_float("gamma", 0.0, 4.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 2.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-2, 8.0, log=True),
            "n_estimators": 500,
        }
        fold_scores: List[float] = []
        for tr, te in cv:
            tr = np.asarray(tr, dtype=int)
            te = np.asarray(te, dtype=int)
            yt, yv = y[tr], y[te]
            if len(np.unique(yt)) < 2 or len(np.unique(yv)) < 2:
                continue
            sw_tr = _row_sample_weights(yt, cb_mode, minority_boost=minority_boost)
            clf = _create_xgb_classifier(
                objective="multi:softprob",
                num_class=3,
                random_state=42,
                tree_method="hist",
                eval_metric="mlogloss",
                **params,
            )
            _fit_xgb_cv_early_stop(
                clf, X_all.iloc[tr], yt, X_all.iloc[te], yv, sw_tr, int(s5["early_stopping_rounds"])
            )
            probs = clf.predict_proba(X_all.iloc[te])
            pred = np.argmax(probs, axis=1).astype(int)
            f1_dir = float(f1_score(yv, pred, average="macro", labels=[1, 2], zero_division=0))
            dir_mask = pred != 0
            prec_dir = float(np.mean(yv[dir_mask] == pred[dir_mask])) if int(dir_mask.sum()) > 0 else 0.0
            if flags.get("skip_task1", False):
                fold_scores.append(f1_dir)
                continue
            m = _simulate_trade_metrics(
                probs,
                y_true=yv,
                y_outcome=(y_outcome[te] if y_outcome is not None else None),
                conf_up=sim_conf_up,
                conf_down=sim_conf_down,
                abstain_tau=sim_abstain_tau,
                rr=rr_for_obj,
            )
            min_trades_expected = max(10, int(len(te) * min_trades_frac))
            if int(m["n_trades"]) < min_trades_expected:
                fold_scores.append(-1.0)
                continue
            score = (
                w_exp * m["expectancy"]
                + w_sharpe * m["sharpe_R"]
                + w_prec * prec_dir
                + w_f1 * f1_dir
            )
            fold_scores.append(float(score))
        if not fold_scores:
            return -3.0
        return float(np.mean(fold_scores))

    log.info("Stage5 | CV=%s | folds=%d | rows_opt=%d", cv_source, len(cv), len(df_opt))
    study = optuna.create_study(direction="maximize", study_name="xgb_stage5")
    study.optimize(objective, n_trials=int(s5["n_trials"]), show_progress_bar=False)
    results = {"best_value": float(study.best_value), "best_params": study.best_params}

    hold_frac = float(s5.get("final_holdout_frac", 0.15))
    n_rows = len(X_all)
    split = int(n_rows * (1.0 - hold_frac))
    split = min(max(split, 500), n_rows - max(120, int(n_rows * 0.08)))
    X_tr0, X_ho0 = X_all.iloc[:split], X_all.iloc[split:]
    y_tr, y_ho = y[:split], y[split:]
    sw_tr = _row_sample_weights(y_tr, cb_mode, minority_boost=minority_boost)

    bp = results["best_params"]
    final_xgb_raw = _create_xgb_classifier(
        objective="multi:softprob",
        num_class=3,
        random_state=42,
        n_estimators=int(s5.get("final_n_estimators", 1200)),
        tree_method="hist",
        eval_metric="mlogloss",
        **{k: v for k, v in bp.items() if k != "n_estimators"},
    )
    _fit_xgb_cv_early_stop(
        final_xgb_raw, X_tr0, y_tr, X_ho0, y_ho, sw_tr, int(s5.get("early_stopping_rounds", 60))
    )

    # Task 5: permutation importance + cleanup feature registry
    active_feats = list(feats_raw)
    if not flags.get("skip_task5", False):
        pi = permutation_importance(
            final_xgb_raw,
            X_ho0,
            y_ho,
            n_repeats=5,
            random_state=42,
            scoring="balanced_accuracy",
        )
        imp_df = pd.DataFrame(
            {"feature": feats_raw, "importance_mean": pi.importances_mean, "importance_std": pi.importances_std}
        ).sort_values("importance_mean", ascending=False)
        cutoff_q = float(np.nanquantile(imp_df["importance_mean"].to_numpy(), 0.05))
        keep_mask = (imp_df["importance_mean"] > 0.0) & (imp_df["importance_mean"] >= cutoff_q)
        active_feats = imp_df.loc[keep_mask, "feature"].tolist() or feats_raw
        active_set = set(active_feats)
        active_set.update([f for f in PROTECTED_FEATURES if f in feats_raw])
        active_feats = [f for f in feats_raw if f in active_set]
        write_json(
            out / "feature_registry.json",
            {
                "active_features": active_feats,
                "total_features": len(feats_raw),
                "active_count": len(active_feats),
                "importance_threshold_q05": float(cutoff_q),
                "protected_features": sorted([f for f in PROTECTED_FEATURES if f in feats_raw]),
                "importance": imp_df.to_dict(orient="records"),
            },
        )
    else:
        write_json(
            out / "feature_registry.json",
            {"active_features": active_feats, "total_features": len(feats_raw), "active_count": len(active_feats)},
        )

    X_tr = X_all.iloc[:split][active_feats].astype(np.float32)
    X_ho = X_all.iloc[split:][active_feats].astype(np.float32)
    final_xgb = _create_xgb_classifier(
        objective="multi:softprob",
        num_class=3,
        random_state=42,
        n_estimators=int(s5.get("final_n_estimators", 1200)),
        tree_method="hist",
        eval_metric="mlogloss",
        **{k: v for k, v in bp.items() if k != "n_estimators"},
    )
    _fit_xgb_cv_early_stop(final_xgb, X_tr, y_tr, X_ho, y_ho, sw_tr, int(s5.get("early_stopping_rounds", 60)))

    # FIX 1: hybrid calibration (none / isotonic / temperature scaling)
    model_for_infer: Any = final_xgb
    calibration_meta: Dict[str, Any] = {"enabled": False}
    probs_before = final_xgb.predict_proba(X_ho)
    probs_after = probs_before
    calibration_scores: Dict[str, Any] = {}
    if not (flags.get("skip_fix1", False) or flags.get("skip_task3", False)) and len(cv) > 0:
        _, last_te = cv[-1]
        cal_idx = np.asarray(last_te, dtype=int)
        cal_idx = cal_idx[cal_idx < split]
        if len(cal_idx) >= 100:
            X_cal = X_all.iloc[cal_idx][active_feats].astype(np.float32)
            y_cal = y[cal_idx]
            probs_none = probs_before
            # isotonic fitted on validation fold
            iso = _fit_calibrator_prefit(final_xgb, X_cal, y_cal)
            probs_iso = iso.predict_proba(X_ho)
            # temperature scaling fitted on validation fold
            probs_val = final_xgb.predict_proba(X_cal)
            ts = _fit_temperature_scaling(probs_val, y_cal)
            probs_temp = ts.transform(probs_before)

            probs_dict = {
                "none": probs_none,
                "isotonic": probs_iso,
                "temperature": probs_temp,
            }
            calibration_scores = _save_reliability_diagram(y_ho, probs_dict, out / "calibration_report.png")
            best_name = min(calibration_scores.keys(), key=lambda k: calibration_scores[k]["ece"])
            if best_name == "isotonic":
                model_for_infer = CalibratedModelWrapper(final_xgb, mode="isotonic", calibrator=iso)
                probs_after = probs_iso
            elif best_name == "temperature":
                model_for_infer = CalibratedModelWrapper(final_xgb, mode="temperature", temp_scaler=ts)
                probs_after = probs_temp
            else:
                model_for_infer = CalibratedModelWrapper(final_xgb, mode="none")
                probs_after = probs_none
            calibration_meta = {
                "enabled": True,
                "selected": best_name,
                "scores": calibration_scores,
                "temperature": ts.temperatures.tolist(),
            }
            write_json(out / "calibration_report.json", calibration_meta)

    pred_ho = np.argmax(probs_after, axis=1).astype(int)
    hold_macro_f1 = float(f1_score(y_ho, pred_ho, average="macro", labels=[0, 1, 2], zero_division=0))
    hold_bal = float(balanced_accuracy_score(y_ho, pred_ho))

    # Task 1: threshold optimization
    threshold_cfg: Dict[str, Any] = {
        "best": {"conf_up": 0.55, "conf_down": 0.55, "abstain_tau": 0.45},
        "metrics": {},
    }
    if not flags.get("skip_task1", False):
        best_key = (-999.0, -999.0)
        best_item: Dict[str, Any] = {}
        min_signal_rate_target = float(s5.get("min_signal_rate_target", 0.25))
        for cu in np.arange(0.45, 0.81, 0.05):
            for cd in np.arange(0.45, 0.81, 0.05):
                for tau in np.arange(0.25, 0.61, 0.05):
                    sim_pred = _simulate_pred_labels(
                        probs_after,
                        conf_up=float(cu),
                        conf_down=float(cd),
                        abstain_tau=float(tau),
                    )
                    m = _simulate_trade_metrics(
                        probs_after,
                        y_true=y_ho,
                        y_outcome=(y_outcome[split:] if y_outcome is not None else None),
                        conf_up=float(cu),
                        conf_down=float(cd),
                        abstain_tau=float(tau),
                        rr=float(cfg.get("stage_2", {}).get("risk_labeling", {}).get("tp_rr", 2.0)),
                    )
                    dir_pred_mask = sim_pred != 0
                    prec_directional = (
                        float(np.mean(y_ho[dir_pred_mask] == sim_pred[dir_pred_mask]))
                        if int(dir_pred_mask.sum()) > 5
                        else 0.0
                    )
                    combined_score = 0.40 * m["expectancy"] + 0.40 * m["sharpe_R"] + 0.20 * prec_directional
                    key = (combined_score, m["expectancy"], m["sharpe_R"])
                    signal_rate = float(m["n_trades"] / max(len(y_ho), 1))
                    if key > best_key and m["n_trades"] >= 20 and signal_rate >= min_signal_rate_target:
                        best_key = key
                        best_item = {
                            "conf_up": float(cu),
                            "conf_down": float(cd),
                            "abstain_tau": float(tau),
                            "prec_directional": prec_directional,
                            "combined_score": combined_score,
                            **m,
                        }
        if best_item:
            threshold_cfg["best"] = {
                "conf_up": best_item["conf_up"],
                "conf_down": best_item["conf_down"],
                "abstain_tau": best_item["abstain_tau"],
            }
            threshold_cfg["metrics"] = {
                k: v
                for k, v in best_item.items()
                if k not in {"conf_up", "conf_down", "abstain_tau"}
            }
            threshold_cfg["metrics"]["signal_rate"] = float(best_item["n_trades"] / max(len(y_ho), 1))
            # Guard: jika signal rate tetap terlalu rendah, longgarkan threshold bertahap.
            relax = float(s5.get("threshold_relax_step", 0.03))
            while float(threshold_cfg["metrics"].get("signal_rate", 0.0)) < min_signal_rate_target:
                cu = max(0.35, float(threshold_cfg["best"]["conf_up"]) - relax)
                cd = max(0.35, float(threshold_cfg["best"]["conf_down"]) - relax)
                tau = max(0.20, float(threshold_cfg["best"]["abstain_tau"]) - relax)
                m_relax = _simulate_trade_metrics(
                    probs_after,
                    y_true=y_ho,
                    y_outcome=(y_outcome[split:] if y_outcome is not None else None),
                    conf_up=cu,
                    conf_down=cd,
                    abstain_tau=tau,
                    rr=float(cfg.get("stage_2", {}).get("risk_labeling", {}).get("tp_rr", 2.0)),
                )
                sig_relax = float(m_relax["n_trades"] / max(len(y_ho), 1))
                threshold_cfg["best"] = {"conf_up": cu, "conf_down": cd, "abstain_tau": tau}
                threshold_cfg["metrics"] = {
                    "n_trades": float(m_relax["n_trades"]),
                    "winrate": float(m_relax["winrate"]),
                    "average_R": float(m_relax["average_R"]),
                    "expectancy": float(m_relax["expectancy"]),
                    "max_drawdown_R": float(m_relax["max_drawdown_R"]),
                    "sharpe_R": float(m_relax["sharpe_R"]),
                    "precision_dir": float(m_relax["precision_dir"]),
                    "signal_rate": sig_relax,
                }
                if sig_relax >= min_signal_rate_target or (cu <= 0.35 and cd <= 0.35 and tau <= 0.20):
                    break
    write_json(out / "threshold_config.json", threshold_cfg)

    # FIX 2: meta label execution filter (XGBoost binary + constrained threshold)
    meta_payload: Dict[str, Any] = {"enabled": False}
    if not (flags.get("skip_fix2", False) or flags.get("skip_task2", False)):
        probs_all = model_for_infer.predict_proba(X_all[active_feats].astype(np.float32))
        meta_base = pd.DataFrame(
            {
                "p_flat": probs_all[:, 0],
                "p_up": probs_all[:, 1],
                "p_down": probs_all[:, 2],
                "spread": df_opt["spread"].to_numpy(dtype=float),
                "atr_zscore": df_opt.get("atr_zscore", pd.Series(np.zeros(len(df_opt)))).to_numpy(dtype=float),
                "session_london": df_opt.get("session_london", pd.Series(np.zeros(len(df_opt)))).to_numpy(dtype=float),
                "session_ny": df_opt.get("session_ny", pd.Series(np.zeros(len(df_opt)))).to_numpy(dtype=float),
                "london_open_proxy": df_opt.get("london_open_proxy", pd.Series(np.zeros(len(df_opt)))).to_numpy(
                    dtype=float
                ),
            }
        ).fillna(0.0)
        y_meta = (
            (df_opt["tp_sl_outcome"].to_numpy(dtype=np.int8) == 1).astype(int)
            if "tp_sl_outcome" in df_opt.columns
            else ((y != 0).astype(int))
        )
        Xm_tr, Xm_ho = meta_base.iloc[:split], meta_base.iloc[split:]
        ym_tr, ym_ho = y_meta[:split], y_meta[split:]
        pos = max(int(ym_tr.sum()), 1)
        neg = max(int(len(ym_tr) - ym_tr.sum()), 1)
        meta_model = _create_xgb_classifier(
            objective="binary:logistic",
            n_estimators=400,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            tree_method="hist",
            eval_metric="logloss",
            random_state=42,
            scale_pos_weight=float(neg / pos),
        )
        meta_model.fit(Xm_tr, ym_tr)
        p_meta = meta_model.predict_proba(Xm_ho)[:, 1]
        # baseline directional precision per side (tanpa meta filter)
        pred_dir = np.argmax(probs_after, axis=1).astype(int)
        y_dir = y_ho
        baseline_up = float(((y_dir == 1) & (pred_dir == 1)).sum() / max(int((pred_dir == 1).sum()), 1))
        baseline_down = float(((y_dir == 2) & (pred_dir == 2)).sum() / max(int((pred_dir == 2).sum()), 1))
        base_prec = float(ym_ho.mean())
        thresholds = np.arange(0.40, 0.751, 0.01)
        best_t = 0.60
        best_score = -999.0
        best_prec = -1.0
        best_stats: Dict[str, float] = {}
        min_trades = 30
        recall_floor = 0.30
        penalty = 0.25
        for t in thresholds:
            m = p_meta >= t
            n_pass = int(m.sum())
            if n_pass < min_trades:
                continue
            prec = float(ym_ho[m].mean())
            recall = float(n_pass / max(len(ym_ho), 1))
            rec_pen = max(0.0, recall_floor - recall)
            score = float(prec - penalty * rec_pen)
            if score > best_score:
                # directional stats
                ydm = y_dir[m]
                pdm = pred_dir[m]
                up_prec = float(((ydm == 1) & (pdm == 1)).sum() / max(int((pdm == 1).sum()), 1))
                down_prec = float(((ydm == 2) & (pdm == 2)).sum() / max(int((pdm == 2).sum()), 1))
                trade_m = _simulate_trade_metrics(
                    probs_after[m],
                    y_true=ydm,
                    y_outcome=(y_outcome[split:][m] if y_outcome is not None else None),
                    conf_up=0.0,
                    conf_down=0.0,
                    abstain_tau=2.0,
                    rr=float(cfg.get("stage_2", {}).get("risk_labeling", {}).get("tp_rr", 2.0)),
                )
                best_score = score
                best_prec = prec
                best_t = float(t)
                best_stats = {
                    "n_trades": float(n_pass),
                    "precision_UP": up_prec,
                    "precision_DOWN": down_prec,
                    "expectancy": float(trade_m["expectancy"]),
                    "sharpe_R": float(trade_m["sharpe_R"]),
                    "recall_proxy": recall,
                    "score": score,
                }
        threshold_cfg["meta_threshold_optimal"] = float(best_t)
        meta_payload = {
            "enabled": True,
            "threshold": best_t,
            "holdout_precision_profitable": float(best_prec if best_prec >= 0 else 0.0),
            "base_precision_profitable": base_prec,
            "precision_gain": float((best_prec if best_prec >= 0 else 0.0) - base_prec),
            "target_precision_gain_min": 0.05,
            "features": list(meta_base.columns),
            "selection_stats": best_stats,
        }
        # report delta precision directional
        mfin = p_meta >= best_t
        ydm = y_dir[mfin]
        pdm = pred_dir[mfin]
        filtered_up = float(((ydm == 1) & (pdm == 1)).sum() / max(int((pdm == 1).sum()), 1))
        filtered_down = float(((ydm == 2) & (pdm == 2)).sum() / max(int((pdm == 2).sum()), 1))
        meta_report = {
            "baseline_precision_UP": baseline_up,
            "filtered_precision_UP": filtered_up,
            "delta_precision_UP": float(filtered_up - baseline_up),
            "baseline_precision_DOWN": baseline_down,
            "filtered_precision_DOWN": filtered_down,
            "delta_precision_DOWN": float(filtered_down - baseline_down),
            "meta_threshold_used": float(best_t),
            "n_trades_passed": int(mfin.sum()),
            "n_trades_total": int(len(mfin)),
        }
        write_json(out / "meta_model_report.json", meta_report)
        joblib.dump({"model": meta_model, "features": list(meta_base.columns), "threshold": best_t}, out / "meta_model.joblib")
    else:
        write_json(out / "meta_model_skipped.json", {"skipped": True})

    # Task 4: regime-aware models
    regime_payload: Dict[str, Any] = {"enabled": False, "models": {}}
    if not flags.get("skip_task4", False):
        regime_models: Dict[str, Any] = {}
        reg_series = df_opt["regime"].astype(str)
        pred_global = np.argmax(probs_after, axis=1).astype(int)
        bal_global = float(balanced_accuracy_score(y_ho, pred_global))
        regime_eval: Dict[str, Any] = {}
        min_rows = int(cfg.get("stage_5", {}).get("regime_min_rows", 1200))
        for rg in REGIMES:
            idx = np.flatnonzero(reg_series.to_numpy() == rg)
            if len(idx) < min_rows:
                continue
            rg_idx = idx[idx < split]
            rg_ho_idx = idx[idx >= split]
            if len(rg_idx) < 400 or len(rg_ho_idx) < 80:
                continue
            model_rg = _create_xgb_classifier(
                objective="multi:softprob",
                num_class=3,
                random_state=42,
                n_estimators=int(s5.get("final_n_estimators", 1200)),
                tree_method="hist",
                eval_metric="mlogloss",
                **{k: v for k, v in bp.items() if k != "n_estimators"},
            )
            sw_rg = _row_sample_weights(y[rg_idx], cb_mode, minority_boost=minority_boost)
            _fit_xgb_cv_early_stop(
                model_rg,
                X_all.iloc[rg_idx][active_feats].astype(np.float32),
                y[rg_idx],
                X_all.iloc[rg_ho_idx][active_feats].astype(np.float32),
                y[rg_ho_idx],
                sw_rg,
                int(s5.get("early_stopping_rounds", 60)),
            )
            pred_rg = model_rg.predict(X_all.iloc[rg_ho_idx][active_feats].astype(np.float32))
            bal_rg = float(balanced_accuracy_score(y[rg_ho_idx], pred_rg))
            regime_eval[rg] = {"rows_train": int(len(rg_idx)), "rows_holdout": int(len(rg_ho_idx)), "balanced_accuracy": bal_rg}
            if bal_rg >= bal_global:
                regime_models[rg] = {"model": model_rg, "features": active_feats}
        joblib.dump(regime_models, out / "regime_models.joblib")
        regime_payload = {"enabled": True, "global_balanced_accuracy": bal_global, "regime_eval": regime_eval}
    else:
        write_json(out / "regime_models_skipped.json", {"skipped": True})

    # Persist global model bundle (interface predict_proba tetap).
    bundle = {
        "model": model_for_infer,
        "features": active_feats,
        "mode": "directional_3class",
        "thresholds": threshold_cfg,
        "meta": meta_payload,
        "calibration": calibration_meta,
        "regime_enabled": bool(regime_payload.get("enabled", False)),
        "regime_models_path": str((out / "regime_models.joblib").resolve()),
    }
    joblib.dump(bundle, out / "xgb_model.joblib")
    fi = sorted(zip(active_feats, final_xgb.feature_importances_[: len(active_feats)].tolist()), key=lambda x: -x[1])[:50]
    write_json(out / "stage_5_xgb_feature_importance.json", {"top": fi})

    payload = {
        "cv_source": cv_source,
        "features": active_feats,
        "optuna": {"xgb": results},
        "class_balance": cb_mode,
        "class_distribution": {"flat": flat_share, "up": float(class_share[1]), "down": float(class_share[2])},
        "minority_boost": minority_boost,
        "cv_metric": "profit_aware_expectancy_sharpe_plus_f1" if not flags.get("skip_task1", False) else "macro_f1",
        "target_encoding": "0=FLAT, 1=UP, 2=DOWN",
        "holdout_rows": int(len(y_ho)),
        "holdout_macro_f1": hold_macro_f1,
        "holdout_balanced_accuracy": hold_bal,
        "threshold_config": threshold_cfg,
        "meta_model": meta_payload,
        "calibration": calibration_meta,
        "regime_models": regime_payload,
    }
    write_json(out / "stage_5_best_config.json", payload)

    if bool(s5.get("train_binary_models_enabled", True)):
        binary_results = train_binary_models(df, active_feats, cfg, run_dir, log)
        payload["binary_models"] = binary_results
    try:
        x_dummy = X_ho.head(10).copy()
        p = model_for_infer.predict_proba(x_dummy)
        p = np.asarray(p, dtype=float)
        mth = float(threshold_cfg.get("meta_threshold_optimal", meta_payload.get("threshold", 0.6)))
        if "meta_model" in locals() and meta_payload.get("enabled"):
            x_meta_dummy = pd.DataFrame(
                {
                    "p_flat": p[:, 0],
                    "p_up": p[:, 1],
                    "p_down": p[:, 2],
                    "spread": df_opt["spread"].iloc[split : split + len(x_dummy)].to_numpy(dtype=float),
                    "atr_zscore": df_opt.get("atr_zscore", pd.Series(np.zeros(len(df_opt)))).iloc[
                        split : split + len(x_dummy)
                    ].to_numpy(dtype=float),
                    "session_london": df_opt.get("session_london", pd.Series(np.zeros(len(df_opt)))).iloc[
                        split : split + len(x_dummy)
                    ].to_numpy(dtype=float),
                    "session_ny": df_opt.get("session_ny", pd.Series(np.zeros(len(df_opt)))).iloc[
                        split : split + len(x_dummy)
                    ].to_numpy(dtype=float),
                    "london_open_proxy": df_opt.get("london_open_proxy", pd.Series(np.zeros(len(df_opt)))).iloc[
                        split : split + len(x_dummy)
                    ].to_numpy(dtype=float),
                }
            ).fillna(0.0)
            pm = meta_model.predict_proba(x_meta_dummy[list(meta_base.columns)])[:, 1]
            pass_mask = pm >= mth
        else:
            pm = np.full(len(x_dummy), np.nan)
            pass_mask = np.ones(len(x_dummy), dtype=bool)
        integ = {
            "rows": int(len(x_dummy)),
            "shape": list(p.shape),
            "no_nan": bool(np.isfinite(p).all()),
            "sum_to_1": bool(np.allclose(p.sum(axis=1), 1.0, atol=1e-5)),
            "meta_threshold_optimal": mth,
            "meta_pass_count": int(pass_mask.sum()),
        }
        write_json(Path(run_dir).parents[1] / "logs" / "integration_check.json", integ)
    except Exception as exc:
        write_json(Path(run_dir).parents[1] / "logs" / "integration_check.json", {"error": str(exc)})
    log.info(
        "Stage5 selesai | CV=%.4f | holdout macro-F1=%.4f | holdout bal-acc=%.4f",
        results["best_value"],
        hold_macro_f1,
        hold_bal,
    )
    return payload
