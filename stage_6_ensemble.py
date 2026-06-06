# noqa: D100
"""Stage 6 — Probabilitas mentah XGBoost (tanpa kalibrasi / tanpa LightGBM)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score

from utils.experiment import write_json
from utils.logging_config import stage_logger
from utils.paths import ensure_dir


def _tp_sl_holdout_stats(df: pd.DataFrame, pred_val: np.ndarray, split: int) -> Dict[str, Any]:
    if "tp_sl_outcome" not in df.columns:
        return {"available": False}
    y_out = df["tp_sl_outcome"].to_numpy(dtype=np.int8)[split:]
    if len(y_out) == 0:
        return {"available": True, "rows": 0}

    directional_mask = pred_val != 0
    dir_n = int(directional_mask.sum())
    tp_first_rate = float((y_out[directional_mask] == 1).mean()) if dir_n else 0.0
    sl_first_rate = float((y_out[directional_mask] == -1).mean()) if dir_n else 0.0
    unresolved_rate = float((y_out[directional_mask] == 0).mean()) if dir_n else 0.0

    per_side: Dict[str, Any] = {}
    for lbl, name in ((1, "UP"), (2, "DOWN")):
        m = pred_val == lbl
        n = int(m.sum())
        per_side[name] = {
            "n": n,
            "tp_first_rate": float((y_out[m] == 1).mean()) if n else 0.0,
            "sl_first_rate": float((y_out[m] == -1).mean()) if n else 0.0,
            "unresolved_rate": float((y_out[m] == 0).mean()) if n else 0.0,
        }

    return {
        "available": True,
        "rows": int(len(y_out)),
        "directional_rows": dir_n,
        "tp_first_rate_directional": tp_first_rate,
        "sl_first_rate_directional": sl_first_rate,
        "unresolved_rate_directional": unresolved_rate,
        "per_pred_side": per_side,
    }


def xgb_class_probs(df: pd.DataFrame, run_dir: Path) -> np.ndarray:
    """Probabilitas (n,3) kolom [FLAT, UP, DOWN] dari XGBoost."""
    s5dir = run_dir / "stage_5"
    bundle = joblib.load(s5dir / "xgb_model.joblib")
    feats: list = bundle["features"]
    X = df[feats].astype(np.float32)
    P = bundle["model"].predict_proba(X)
    if P.shape[1] != 3:
        raise ValueError(f"xgb_class_probs: proba shape={P.shape}")
    return P.astype(float)


def decide_pred_labels(P: np.ndarray, decision: Dict[str, Any]) -> np.ndarray:
    if decision.get("type") == "directional_thresholds":
        cu = float(decision.get("conf_up", 0.55))
        cd = float(decision.get("conf_down", 0.55))
        tau = float(decision.get("abstain_tau", 0.45))
        p0, p1, p2 = P[:, 0], P[:, 1], P[:, 2]
        pred = np.zeros(len(P), dtype=np.int8)
        long_m = (p1 >= cu) & (p0 < tau) & (p1 >= p2)
        short_m = (p2 >= cd) & (p0 < tau) & (p2 > p1)
        pred[long_m] = 1
        pred[short_m] = 2
        return pred
    if decision.get("type") == "abstain_tau":
        tau = float(decision["tau"])
        p0, p1, p2 = P[:, 0], P[:, 1], P[:, 2]
        flat = p0 >= tau
        pred = np.empty(len(P), dtype=np.int8)
        pred[flat] = 0
        m = ~flat
        pred[m] = np.where(p1[m] >= p2[m], 1, 2)
        return pred
    return np.argmax(P, axis=1).astype(np.int8)


def search_flat_abstain_tau(
    P_val: np.ndarray,
    y_val: np.ndarray,
    *,
    tol: float,
    acc_weight: float,
    bal_weight: float,
    log,
) -> Dict[str, Any]:
    true_flat = float((y_val == 0).mean())
    tols = [tol, tol + 0.1, tol + 0.22]
    best_key: tuple = (-1.0, -1.0, -1.0, 0.0)
    best_meta: Dict[str, Any] = {}

    for t_try in tols:
        for tau in np.arange(0.16, 0.88, 0.012):
            pred = decide_pred_labels(P_val, {"type": "abstain_tau", "tau": float(tau)})
            pf = float((pred == 0).mean())
            if pf < true_flat - t_try or pf > true_flat + t_try:
                continue
            acc = float(accuracy_score(y_val, pred))
            bal = float(balanced_accuracy_score(y_val, pred))
            score = float(acc_weight * acc + bal_weight * bal)
            flat_gap = -abs(pf - true_flat)
            key = (score, bal, flat_gap, tau)
            if key > best_key:
                best_key = key
                best_meta = {
                    "type": "abstain_tau",
                    "tau": float(tau),
                    "val_accuracy": acc,
                    "val_balanced_accuracy": bal,
                    "val_score_mixed": score,
                    "val_pred_flat_rate": pf,
                    "val_true_flat_rate": true_flat,
                    "tol_used": t_try,
                    "score_weights": {"accuracy": acc_weight, "balanced_accuracy": bal_weight},
                }
        if best_meta:
            break

    if not best_meta:
        log.warning("Stage6 | abstain τ: tidak ada kandidat — pakai argmax.")
        return {"type": "argmax"}
    log.info(
        "Stage6 | abstain τ=%.3f | val acc=%.4f | flat%% pred=%.3f true=%.3f",
        best_meta["tau"],
        best_meta["val_accuracy"],
        best_meta["val_pred_flat_rate"],
        true_flat,
    )
    return best_meta


def _temporal_val_split(n: int, val_frac: float, val_min: int) -> int:
    """Indeks awal subset validasi (urut waktu)."""
    if n < val_min * 3:
        return max(n // 2, 1)
    val_rows = max(int(n * val_frac), val_min)
    val_rows = min(val_rows, max(n // 3, val_min))
    return max(n - val_rows, int(n * 0.55))


def load_decision_from_run_dir(run_dir: Path) -> Dict[str, Any]:
    p = run_dir / "stage_6" / "stage_6_metadata.json"
    if not p.is_file():
        return {"type": "argmax"}
    meta = json.loads(p.read_text(encoding="utf-8"))
    return meta.get("decision", {"type": "argmax"})


def binary_model_probs(
    df: pd.DataFrame,
    run_dir: Path,
) -> tuple[np.ndarray | None, np.ndarray | None, float | None, float | None]:
    """Load binary models dan return P_up, P_down arrays. Return None jika tidak ada."""
    path_up = run_dir / "stage_5" / "xgb_binary_up.joblib"
    path_down = run_dir / "stage_5" / "xgb_binary_down.joblib"
    if not path_up.is_file() or not path_down.is_file():
        return None, None, None, None
    bundle_up = joblib.load(path_up)
    bundle_down = joblib.load(path_down)
    feats_up = bundle_up["features"]
    feats_down = bundle_down["features"]
    X_up = df[feats_up].astype(np.float32)
    X_down = df[feats_down].astype(np.float32)
    P_up = bundle_up["model"].predict_proba(X_up)[:, 1]
    P_down = bundle_down["model"].predict_proba(X_down)[:, 1]
    thr_up = float(bundle_up.get("threshold", 0.50))
    thr_down = float(bundle_down.get("threshold", 0.50))
    return P_up, P_down, thr_up, thr_down


def tune_binary_thresholds(
    P_up: np.ndarray,
    P_down: np.ndarray,
    y_true: np.ndarray,
    thr_up0: float,
    thr_down0: float,
    *,
    min_sig: float,
    max_sig: float,
) -> tuple[float, float, Dict[str, Any]]:
    """Cari thr_up/thr_down agar signal rate dalam [min_sig, max_sig] dan accuracy maksimal."""
    best_key: tuple = (-1.0, -1.0, -1.0)
    best_thr = (thr_up0, thr_down0)
    best_meta: Dict[str, Any] = {}
    for thr_up in np.arange(0.35, 0.86, 0.03):
        for thr_down in np.arange(0.35, 0.86, 0.03):
            pred = decide_binary_pred(P_up, P_down, float(thr_up), float(thr_down))
            sig = float((pred != 0).mean())
            if sig < min_sig or sig > max_sig:
                continue
            acc = float(accuracy_score(y_true, pred))
            bal = float(balanced_accuracy_score(y_true, pred))
            sig_pref = -abs(sig - (min_sig + max_sig) / 2.0)
            key = (acc, bal, sig_pref)
            if key > best_key:
                best_key = key
                best_thr = (float(thr_up), float(thr_down))
                best_meta = {
                    "val_accuracy": acc,
                    "val_balanced_accuracy": bal,
                    "val_signal_rate": sig,
                }
    if not best_meta:
        return thr_up0, thr_down0, {}
    return best_thr[0], best_thr[1], best_meta


def search_directional_thresholds(
    P_val: np.ndarray,
    y_val: np.ndarray,
    *,
    min_sig: float,
    max_sig: float,
    acc_w: float,
    bal_w: float,
    base: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """Cari conf_up/conf_down/abstain_tau dengan signal rate dalam [min_sig, max_sig]."""
    base = base or {"conf_up": 0.55, "conf_down": 0.55, "abstain_tau": 0.45}
    best_key: tuple = (-1.0, -1.0, -1.0, -1.0)
    best_decision: Dict[str, Any] = {}

    def _search(cu_grid, cd_grid, tau_grid) -> None:
        nonlocal best_key, best_decision
        for cu in cu_grid:
            for cd in cd_grid:
                for tau in tau_grid:
                    pred_v = decide_pred_labels(
                        P_val,
                        {
                            "type": "directional_thresholds",
                            "conf_up": float(cu),
                            "conf_down": float(cd),
                            "abstain_tau": float(tau),
                        },
                    )
                    sig = float((pred_v != 0).mean())
                    if sig < min_sig or sig > max_sig:
                        continue
                    acc = float(accuracy_score(y_val, pred_v))
                    bal = float(balanced_accuracy_score(y_val, pred_v))
                    score = float(acc_w * acc + bal_w * bal)
                    sig_pref = -abs(sig - (min_sig + max_sig) / 2.0)
                    key = (score, bal, sig_pref, -tau)
                    if key > best_key:
                        best_key = key
                        best_decision = {
                            "type": "directional_thresholds",
                            "conf_up": float(cu),
                            "conf_down": float(cd),
                            "abstain_tau": float(tau),
                            "val_accuracy": acc,
                            "val_balanced_accuracy": bal,
                            "val_score_mixed": score,
                            "val_signal_rate": sig,
                            "target_signal_rate_range": [min_sig, max_sig],
                            "score_weights": {"accuracy": acc_w, "balanced_accuracy": bal_w},
                        }

    bu, bd, bt = base["conf_up"], base["conf_down"], base["abstain_tau"]
    _search(
        np.arange(max(0.35, bu - 0.12), min(0.91, bu + 0.19), 0.03),
        np.arange(max(0.35, bd - 0.12), min(0.91, bd + 0.19), 0.03),
        np.arange(max(0.16, bt - 0.15), min(0.91, bt + 0.16), 0.03),
    )
    if not best_decision:
        _search(
            np.arange(0.45, 0.81, 0.05),
            np.arange(0.45, 0.81, 0.05),
            np.arange(0.20, 0.56, 0.05),
        )
    return best_decision


def decide_binary_pred(
    P_up: np.ndarray,
    P_down: np.ndarray,
    thr_up: float,
    thr_down: float,
) -> np.ndarray:
    """
    Decision logic binary:
    - UP   jika P_up >= thr_up  AND P_down < thr_down
    - DOWN jika P_down >= thr_down AND P_up < thr_up
    - FLAT (0) jika ambiguous atau low confidence
    """
    pred = np.zeros(len(P_up), dtype=np.int8)
    up_mask = (P_up >= thr_up) & (P_down < thr_down)
    down_mask = (P_down >= thr_down) & (P_up < thr_up)
    pred[up_mask] = 1
    pred[down_mask] = 2
    return pred


def _load_stage5_thresholds(run_dir: Path) -> Dict[str, float]:
    p = run_dir / "stage_5" / "threshold_config.json"
    if not p.is_file():
        return {"conf_up": 0.55, "conf_down": 0.55, "abstain_tau": 0.45}
    try:
        obj = json.loads(p.read_text(encoding="utf-8"))
        best = obj.get("best", {}) if isinstance(obj, dict) else {}
        return {
            "conf_up": float(best.get("conf_up", 0.55)),
            "conf_down": float(best.get("conf_down", 0.55)),
            "abstain_tau": float(best.get("abstain_tau", 0.45)),
        }
    except Exception:
        return {"conf_up": 0.55, "conf_down": 0.55, "abstain_tau": 0.45}


def run_stage_6(
    df: pd.DataFrame,
    cfg: Dict[str, Any],
    *,
    run_dir: Path,
) -> Dict[str, Any]:
    out = ensure_dir(run_dir / "stage_6")
    log = stage_logger("stage_6_ensemble", out)
    s6 = cfg.get("stage_6", {})

    P = xgb_class_probs(df, run_dir)
    meta: Dict[str, Any] = {"decision_source": "3class_only"}

    decision: Dict[str, Any] = {"type": "argmax"}
    use_binary = bool(s6.get("use_binary_models", True))
    binary_result = None
    if use_binary:
        try:
            result = binary_model_probs(df, run_dir)
            if result[0] is not None:
                P_up_bin, P_down_bin, thr_up, thr_down = result
                binary_result = {
                    "P_up": P_up_bin,
                    "P_down": P_down_bin,
                    "thr_up": thr_up,
                    "thr_down": thr_down,
                }
                log.info("Stage6 | binary models loaded | thr_up=%.3f thr_down=%.3f", thr_up, thr_down)
        except Exception as e:
            log.warning("Stage6 | gagal load binary models: %s — fallback ke 3-kelas", e)

    min_sig = float(s6.get("min_signal_rate_target", 0.25))
    max_sig = float(s6.get("max_signal_rate_target", 0.50))

    use_binary_pred = False
    pred_binary_final: np.ndarray | None = None

    n_tr = len(df)
    val_frac = float(s6.get("abstain_search_val_frac", 0.15))
    val_min = int(s6.get("abstain_val_min_rows", 120))
    split = _temporal_val_split(n_tr, val_frac, val_min)

    if binary_result is not None:
        P_up_bin = binary_result["P_up"]
        P_down_bin = binary_result["P_down"]
        thr_up_use = binary_result["thr_up"]
        thr_down_use = binary_result["thr_down"]
        if n_tr - split >= val_min:
            tuned_up, tuned_down, tune_meta = tune_binary_thresholds(
                P_up_bin[split:],
                P_down_bin[split:],
                df["target"].to_numpy()[split:].astype(int),
                thr_up_use,
                thr_down_use,
                min_sig=min_sig,
                max_sig=max_sig,
            )
            if tune_meta:
                thr_up_use, thr_down_use = tuned_up, tuned_down
                binary_result["thr_up"] = thr_up_use
                binary_result["thr_down"] = thr_down_use
                log.info(
                    "Stage6 | binary thresholds tuned | thr_up=%.3f thr_down=%.3f | val acc=%.4f sig=%.3f",
                    thr_up_use,
                    thr_down_use,
                    tune_meta["val_accuracy"],
                    tune_meta["val_signal_rate"],
                )
        pred_binary = decide_binary_pred(P_up_bin, P_down_bin, thr_up_use, thr_down_use)
        sig_rate_bin = float((pred_binary != 0).mean())
        log.info("Stage6 | binary pred signal_rate=%.3f", sig_rate_bin)

        if min_sig <= sig_rate_bin <= max_sig:
            pred_binary_final = pred_binary
            use_binary_pred = True
            meta["decision_source"] = "binary_models"
            log.info("Stage6 | PAKAI binary models (signal_rate OK)")
        else:
            tuned = False
            if sig_rate_bin < min_sig:
                for delta in [0.03, 0.06, 0.09, 0.12]:
                    new_thr_up = max(0.35, binary_result["thr_up"] - delta)
                    new_thr_down = max(0.35, binary_result["thr_down"] - delta)
                    pred_try = decide_binary_pred(P_up_bin, P_down_bin, new_thr_up, new_thr_down)
                    sr = float((pred_try != 0).mean())
                    if min_sig <= sr <= max_sig:
                        pred_binary_final = pred_try
                        use_binary_pred = True
                        meta["decision_source"] = f"binary_models_relaxed_delta{delta}"
                        log.info("Stage6 | PAKAI binary relaxed -delta=%.2f (signal_rate=%.3f)", delta, sr)
                        tuned = True
                        break
            elif sig_rate_bin > max_sig:
                for delta in [0.03, 0.06, 0.09, 0.12, 0.15]:
                    new_thr_up = min(0.92, binary_result["thr_up"] + delta)
                    new_thr_down = min(0.92, binary_result["thr_down"] + delta)
                    pred_try = decide_binary_pred(P_up_bin, P_down_bin, new_thr_up, new_thr_down)
                    sr = float((pred_try != 0).mean())
                    if min_sig <= sr <= max_sig:
                        pred_binary_final = pred_try
                        use_binary_pred = True
                        meta["decision_source"] = f"binary_models_tightened_delta{delta}"
                        log.info("Stage6 | PAKAI binary tightened +delta=%.2f (signal_rate=%.3f)", delta, sr)
                        tuned = True
                        break
            if not tuned:
                log.warning(
                    "Stage6 | binary signal_rate=%.3f out of range [%.2f,%.2f] — fallback 3-kelas",
                    sig_rate_bin,
                    min_sig,
                    max_sig,
                )
                meta["decision_source"] = "fallback_3class"
    else:
        meta["decision_source"] = "3class_only"

    if bool(s6.get("use_flat_abstain_search", True)) and n_tr - split >= val_min:
        P_val = P[split:]
        y_val = df["target"].to_numpy()[split:].astype(int)
        tol = float(s6.get("abstain_flat_rate_tolerance", 0.12))
        acc_w = float(s6.get("abstain_score_accuracy_weight", 0.5))
        bal_w = float(s6.get("abstain_score_balanced_accuracy_weight", 0.5))
        wsum = max(acc_w + bal_w, 1e-9)
        acc_w, bal_w = acc_w / wsum, bal_w / wsum
        base = _load_stage5_thresholds(run_dir)
        min_sig = float(s6.get("min_signal_rate_target", 0.25))
        max_sig = float(s6.get("max_signal_rate_target", 0.50))
        relax = float(s6.get("threshold_relax_step", 0.03))
        best_decision = search_directional_thresholds(
            P_val,
            y_val,
            min_sig=min_sig,
            max_sig=max_sig,
            acc_w=acc_w,
            bal_w=bal_w,
            base=base,
        )
        if best_decision:
            decision = best_decision
            meta["decision_source"] = "directional_thresholds"
            log.info(
                "Stage6 | directional thresholds cu=%.3f cd=%.3f tau=%.3f | val acc=%.4f | signal_rate=%.3f",
                decision["conf_up"],
                decision["conf_down"],
                decision["abstain_tau"],
                decision["val_accuracy"],
                decision["val_signal_rate"],
            )
        else:
            # fallback ke search abstain lama, lalu longgarkan jika perlu untuk min signal rate.
            decision = search_flat_abstain_tau(
                P_val,
                y_val,
                tol=tol,
                acc_weight=acc_w,
                bal_weight=bal_w,
                log=log,
            )
            if decision.get("type") == "abstain_tau":
                for _ in range(16):
                    pred_v = decide_pred_labels(P_val, decision)
                    sig = float((pred_v != 0).mean())
                    if min_sig <= sig <= max_sig:
                        break
                    if sig > max_sig:
                        decision["tau"] = min(0.90, float(decision["tau"]) + relax)
                    else:
                        decision["tau"] = max(0.16, float(decision["tau"]) - relax)
                decision["val_signal_rate"] = float((decide_pred_labels(P_val, decision) != 0).mean())
            meta["decision_source"] = "3class_abstain"
    elif bool(s6.get("use_flat_abstain_search", True)):
        log.warning("Stage6 | data terlalu sedikit untuk cari abstain τ — pakai argmax.")

    if use_binary_pred and pred_binary_final is not None:
        pred = pred_binary_final
        decision = {"type": "binary_models", "thr_up": binary_result["thr_up"], "thr_down": binary_result["thr_down"]}
    else:
        pred = decide_pred_labels(P, decision)
    pred_val = pred[split:]
    y_val_full = df["target"].to_numpy()[split:].astype(int)
    conf = P[np.arange(len(P)), pred.astype(int)].astype(float)

    out_df = pd.DataFrame(index=df.index)
    if "time" in df.columns:
        out_df["time"] = df["time"].values
    out_df["pred_label"] = pred
    out_df["confidence"] = conf
    out_df["prob_flat"] = P[:, 0]
    out_df["prob_up"] = P[:, 1]
    out_df["prob_down"] = P[:, 2]
    out_df["actual_label"] = df["target"].to_numpy()
    if "tp_sl_outcome" in df.columns:
        out_df["actual_tp_sl_outcome"] = df["tp_sl_outcome"].to_numpy(dtype=np.int8)

    pq = out / "stage_6_predictions.parquet"
    out_df.to_parquet(pq, index=False)

    sig_rate = float((pred_val != 0).mean()) if len(pred_val) else 0.0
    ho_acc = float(accuracy_score(y_val_full, pred_val)) if len(pred_val) else 0.0
    ho_bal = float(balanced_accuracy_score(y_val_full, pred_val)) if len(pred_val) else 0.0
    tp_sl_stats = _tp_sl_holdout_stats(df, pred_val, split)
    meta_out: Dict[str, Any] = {
        "decision": decision,
        "decision_source": meta.get("decision_source", "3class_only"),
        "model": "xgboost_raw_probs",
        "signal_rate_holdout": sig_rate,
        "signal_rate": sig_rate,
        "holdout_accuracy": ho_acc,
        "holdout_balanced_accuracy": ho_bal,
        "holdout_rows": int(len(pred_val)),
        "tp_sl_holdout_stats": tp_sl_stats,
        "mode": "directional_flat_up_down",
    }
    if binary_result is not None:
        meta_out["binary_prob_up_mean"] = float(np.mean(binary_result["P_up"]))
        meta_out["binary_prob_down_mean"] = float(np.mean(binary_result["P_down"]))
    meta = meta_out
    write_json(out / "stage_6_metadata.json", meta)
    log.info(
        "Stage6 selesai | %s | holdout acc=%.4f bal=%.4f | %s",
        decision.get("type"),
        ho_acc,
        ho_bal,
        pq,
    )
    return meta


