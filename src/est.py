#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import pathlib
from typing import List, Tuple

import numpy as np
import pandas as pd
import joblib

from sklearn.linear_model import LinearRegression, RidgeCV, ElasticNetCV, HuberRegressor
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor


def _logit(y: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    y = np.clip(y, eps, 1 - eps)
    return np.log(y / (1 - y))

def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))

ABSOLUTE_FEATURES = [
    "mean", "median", "max", "percentile_90", "percentile_75",
    "percentile_25", "percentile_10", "min"
]

def extract_statistical_features(df: pd.DataFrame) -> pd.DataFrame:
    client_cols = [c for c in df.columns if c.startswith("Client") and "Accuracy" in c]
    accuracies = df[client_cols].values
    feats = pd.DataFrame({
        'mean': np.mean(accuracies, axis=1),
        'median': np.median(accuracies, axis=1),
        'max': np.max(accuracies, axis=1),
        'percentile_90': np.percentile(accuracies, 90, axis=1),
        'percentile_75': np.percentile(accuracies, 75, axis=1),
        'percentile_25': np.percentile(accuracies, 25, axis=1),
        'percentile_10': np.percentile(accuracies, 10, axis=1),
        'min': np.min(accuracies, axis=1),
    })
    return feats

def add_shape_features(df_abs: pd.DataFrame) -> pd.DataFrame:
    df = df_abs.copy()
    df['std'] = df[['min', 'percentile_10', 'percentile_25', 'median', 'percentile_75', 'percentile_90', 'max']].std(axis=1)
    for c in ['median', 'max', 'percentile_90', 'percentile_75', 'percentile_25', 'percentile_10', 'min']:
        df[f'{c}_dm'] = df[c] - df['mean']
    return df

def select_features(df_in: pd.DataFrame, featureset: str, include_n_clients: bool) -> Tuple[pd.DataFrame, List[str]]:
    df = df_in.copy()
    feat_cols = ABSOLUTE_FEATURES[:]
    if featureset == 'shape':
        df = add_shape_features(df_in)
        feat_cols = ["mean", "std", "max_dm", "percentile_90_dm", "percentile_75_dm",
                     "median_dm", "percentile_25_dm", "percentile_10_dm", "min_dm"]
    elif featureset == 'shape_no_level':
        df = add_shape_features(df_in)
        feat_cols = ["std", "max_dm", "percentile_90_dm", "percentile_75_dm",
                     "median_dm", "percentile_25_dm", "percentile_10_dm", "min_dm"]
    if include_n_clients and "n_clients" in df.columns:
        feat_cols += ["n_clients"]
    return df, feat_cols



def build_model_pipeline(model_type: str, degree: int, scaler: str, random_state: int) -> Pipeline:
    steps = []
    if scaler == 'standard':
        steps.append(('scaler', StandardScaler()))
    mt = model_type.lower()
    if mt == 'linear':
        steps.append(('lr', LinearRegression()))
    elif mt == 'ridge':
        steps.append(('ridge', RidgeCV(alphas=(1e-3, 1e-2, 1e-1, 1, 10, 100))))
    elif mt == 'elasticnet':
        steps.append(('enet', ElasticNetCV(l1_ratio=[.1, .3, .5, .7, .9, 1.0], cv=5, random_state=random_state)))
    elif mt == 'huber':
        steps.append(('huber', HuberRegressor()))
    elif mt == 'polynomial':
        steps = [('poly', PolynomialFeatures(degree=degree, include_bias=False))] + steps + [('lr', LinearRegression())]
    elif mt == 'svr':
        steps.append(('svr', SVR(C=10.0, epsilon=0.02, kernel='rbf')))
    elif mt == 'rf':
        steps.append(('rf', RandomForestRegressor(n_estimators=600, n_jobs=-1, random_state=random_state)))
    elif mt == 'hgb':
        steps.append(('hgb', HistGradientBoostingRegressor(learning_rate=0.05, max_iter=500, random_state=random_state)))
    elif mt == 'tree':
        steps.append(('tree', DecisionTreeRegressor(max_depth=4, random_state=random_state)))
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
    return Pipeline(steps)



def load_all_data(base_path: str, client_counts: List[int], data_list: List[str], approach: str, include_n_clients: bool) -> pd.DataFrame:
    rows = []
    for data in data_list:
        for n in client_counts:
            fname = f"{n}_clients_{data}_{approach}.csv"
            path = os.path.join(base_path, fname)
            if not os.path.isfile(path):
                continue
            df = pd.read_csv(path)
            feats = extract_statistical_features(df)
            if include_n_clients:
                feats['n_clients'] = n
            feats['dataset'] = data
            feats['Global Accuracy'] = df['Global Accuracy'].values
            rows.append(feats)

    return pd.concat(rows, ignore_index=True)

def leave_one_dataset_out_cv(base_path, client_counts, data_list, approach, model_type, degree, random_state,
                             include_n_clients, featureset, scaler, target_mode, logit_target,
                             save_models, output_dir) -> pd.DataFrame:

    data_all = load_all_data(base_path, client_counts, data_list, approach, include_n_clients)
    results = []

    for test_data in sorted(data_all['dataset'].unique()):
        train_df = data_all[data_all['dataset'] != test_data].copy()
        test_df  = data_all[data_all['dataset'] == test_data].copy()

        train_df2, feat_cols = select_features(train_df, featureset, include_n_clients)
        test_df2,  _         = select_features(test_df, featureset, include_n_clients)
        Xtr = train_df2[feat_cols].values
        Xte = test_df2[feat_cols].values

        if target_mode == "delta":
            ytr = train_df['Global Accuracy'].values - train_df['mean'].values
            yte = test_df['Global Accuracy'].values - test_df['mean'].values
            ytr_model = ytr
            inverse = lambda z, df: z + df['mean'].values
            y_true_prob = yte + test_df['mean'].values
        else:
            ytr = train_df['Global Accuracy'].values
            yte = test_df['Global Accuracy'].values
            ytr_model = _logit(ytr) if logit_target else ytr
            inverse = (lambda z, df: _sigmoid(z)) if logit_target else (lambda z, df: z)
            y_true_prob = yte

        pipeline = build_model_pipeline(model_type, degree, scaler, random_state)
        pipeline.fit(Xtr, ytr_model)
        y_pred_model = pipeline.predict(Xte)
        y_pred_prob = inverse(y_pred_model, test_df)

        rmse = float(np.sqrt(mean_squared_error(y_true_prob, y_pred_prob)))
        mae  = float(mean_absolute_error(y_true_prob, y_pred_prob))
        r2   = float(r2_score(y_true_prob, y_pred_prob))
        bias = float(np.mean(y_true_prob - y_pred_prob))
        under_pct = float(np.mean((y_true_prob - y_pred_prob) > 0) * 100.0)

        results.append({
            "left_out_dataset": test_data,
            "rmse": rmse,
            "mae": mae,
            "r2": r2,
            "bias": bias,
            "under_percent": under_pct,
            "model": model_type,
            "approach": approach,
            "target_mode": target_mode,
            "include_n_clients": include_n_clients
        })

        if save_models:
            pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
            filename = f"{model_type}_{featureset}_{scaler}_{target_mode}_{'inclN' if include_n_clients else 'noN'}_leaveout_{test_data}.joblib"
            save_path = os.path.join(output_dir, filename)
            joblib.dump({"pipeline": pipeline, "features": feat_cols}, save_path)

    return pd.DataFrame(results)

def run_all_experiments():
    BASE_PATH = "/history"
    CLIENT_COUNTS = [10,20,30,40,50,60,70,80,90,100]
    DATA_LIST = ["huga", "spambase", "adult", "kdd"]
    MODEL_TYPES = ["ridge", "rf", "hgb" , "linear" , "tree"]
    APPROACHES = ["fedfor", "fedlr"]
    TARGET_MODES = ["delta", "prob"]
    INCLUDE_N_OPTIONS = [True, False]

    FEATURESET = "shape_no_level"
    SCALER = "none"
    DEGREE = 2
    RANDOM_STATE = 42
    SAVE_MODELS = True

    ROOT_DIR = f"/..."
    os.makedirs(ROOT_DIR, exist_ok=True)

    all_df = []

    for model in MODEL_TYPES:
        for approach in APPROACHES:
            for target_mode in TARGET_MODES:
                for include_n in INCLUDE_N_OPTIONS:
                    tag = f"{model}_{approach}_{target_mode}_{'inclN' if include_n else 'noN'}"
                    out_dir = os.path.join(ROOT_DIR, tag)
                    try:
                        df = leave_one_dataset_out_cv(
                            base_path=BASE_PATH,
                            client_counts=CLIENT_COUNTS,
                            data_list=DATA_LIST,
                            approach=approach,
                            model_type=model,
                            degree=DEGREE,
                            random_state=RANDOM_STATE,
                            include_n_clients=include_n,
                            featureset=FEATURESET,
                            scaler=SCALER,
                            target_mode=target_mode,
                            logit_target=False,
                            save_models=SAVE_MODELS,
                            output_dir=out_dir
                        )
                        all_df.append(df)
                    except Exception:
                        pass

    if all_df:
        df_all = pd.concat(all_df, ignore_index=True)
        out_path = os.path.join(ROOT_DIR, "experiments_summary.csv")
        df_all.to_csv(out_path, index=False)

if __name__ == "__main__":
    np.set_printoptions(suppress=True, linewidth=160)
    pd.set_option('display.width', 160)
    run_all_experiments()
