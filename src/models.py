"""Constant, logistic regression, LightGBM and factorization machine models, plus the V1 blend."""
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from .config import (FLAGS, FM_MAX_EPOCHS, FM_PARAMS, FM_THREADS, LGBM_EARLY_STOPPING_ROUNDS, LGBM_MAX_ROUNDS,
                     LGBM_PARAMS, LR_LOG_SCALED, LR_NEVER_SEEN_HOURS, LR_ONE_HOT_NUMERIC, LR_RATES, RETRAIN_HOURS, ROOT,
                     SEED)
from .features import FeatureEncoder, FMEncoder, logit


class ConstantModel:
    """Predicts the training click rate for every impression."""
    name = "constant"

    def fit(self, X, y, **_):
        self.p_ = float(np.mean(y))
        return self

    def predict(self, X):
        return np.full(len(X), self.p_)


class LogisticModel:
    """One-hot categories, deciles and hour of day; log-scaled numerics; L2-regularized logistic regression."""
    name = "logistic_regression"

    def __init__(self, one_hot_columns, C=0.1):
        self.one_hot_columns = one_hot_columns + LR_ONE_HOT_NUMERIC
        self.C = C

    def fit(self, X, y, **_):
        prep = ColumnTransformer([
            ("one_hot", OneHotEncoder(handle_unknown="ignore"), self.one_hot_columns),
            ("log_scaled", make_pipeline(SimpleImputer(strategy="constant", fill_value=LR_NEVER_SEEN_HOURS),
                                         FunctionTransformer(np.log1p), StandardScaler()), LR_LOG_SCALED),
            ("rates", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), LR_RATES),
            ("flags", "passthrough", FLAGS),
        ])
        self.pipeline_ = make_pipeline(prep, LogisticRegression(C=self.C, max_iter=1000, random_state=SEED))
        self.pipeline_.fit(X, y)
        return self

    def predict(self, X):
        return self.pipeline_.predict_proba(X)[:, 1]


class LightGBMModel:
    """Gradient-boosted trees; category codes as native categoricals. Early-stops when given validation data."""
    name = "lightgbm"

    def __init__(self, categorical, num_rounds=None, params=LGBM_PARAMS, max_rounds=LGBM_MAX_ROUNDS):
        self.categorical = categorical
        self.num_rounds = num_rounds
        self.params = params
        self.max_rounds = max_rounds

    def fit(self, X, y, X_val=None, y_val=None):
        train_set = lgb.Dataset(X, y, categorical_feature=self.categorical)
        if X_val is not None:
            val_set = lgb.Dataset(X_val, y_val, reference=train_set)
            self.booster_ = lgb.train(self.params, train_set, num_boost_round=self.max_rounds, valid_sets=[val_set],
                                      callbacks=[lgb.early_stopping(LGBM_EARLY_STOPPING_ROUNDS, verbose=False)])
            self.num_rounds = self.booster_.best_iteration
        else:
            self.booster_ = lgb.train(self.params, train_set, num_boost_round=self.num_rounds)
        return self

    def predict(self, X):
        return self.booster_.predict(X, num_iteration=self.num_rounds)

    def feature_importance(self):
        gain = self.booster_.feature_importance(importance_type="gain")
        return dict(zip(self.booster_.feature_name(), gain / gain.sum()))


class FactorizationMachineModel:
    """
    Factorization machine on its own encoding (FMEncoder), trained and scored by `python -m src.fm` in a separate
    process. Takes raw rows, not the FeatureEncoder matrix. Early-stops on `stop` rows when given (sets num_epochs);
    otherwise trains exactly num_epochs.
    """
    name = "factorization_machine"

    def __init__(self, num_epochs=None, params=FM_PARAMS, max_epochs=FM_MAX_EPOCHS):
        self.num_epochs = num_epochs
        self.params = params
        self.max_epochs = max_epochs

    def fit(self, train, stop=None):
        self.workdir_ = tempfile.TemporaryDirectory(prefix="fm_")
        self.model_path_ = Path(self.workdir_.name) / "model.pt"
        self.encoder_ = FMEncoder().fit(train)
        no_rows = train.iloc[:0]
        arrays = {}
        for part, rows in [("train", train), ("stop", stop if stop is not None else no_rows), ("eval", no_rows)]:
            arrays[f"idx_{part}"], arrays[f"dense_{part}"] = self.encoder_.transform(rows)
        inputs, outputs = Path(self.workdir_.name) / "fit_in.npz", Path(self.workdir_.name) / "fit_out.npz"
        np.savez(inputs, y_train=train["click"].to_numpy(),
                 y_stop=(stop if stop is not None else no_rows)["click"].to_numpy(), **arrays)
        settings = {"seed": SEED, "threads": FM_THREADS, **self.params, "save_path": str(self.model_path_)}
        if stop is None:
            settings.update(fixed_epochs=self.num_epochs, max_epochs=self.num_epochs + 1)
        else:
            settings["max_epochs"] = self.max_epochs
        self._run("fit", inputs, outputs, json.dumps(settings))
        if stop is not None:
            self.num_epochs = float(np.load(outputs)["epochs"])
        return self

    def predict(self, rows):
        idx, dense = self.encoder_.transform(rows)
        inputs, outputs = Path(self.workdir_.name) / "predict_in.npz", Path(self.workdir_.name) / "predict_out.npz"
        np.savez(inputs, idx=idx, dense=dense)
        self._run("predict", self.model_path_, inputs, outputs)
        return np.load(outputs)["p"]

    def export_weights(self, path):
        """Write the learned numbers as plain arrays, for serving without PyTorch (src/serving.py)."""
        self._run("export", self.model_path_, path)

    @staticmethod
    def _run(*args):
        subprocess.run([sys.executable, "-m", "src.fm", *map(str, args)], cwd=ROOT, check=True)


def blend(p_lightgbm, p_fm):
    """V1: average of the two models' log-odds, mapped back to a probability."""
    return 1 / (1 + np.exp(-(logit(p_lightgbm) + logit(p_fm)) / 2))


def retrain_blocks(day, hours=RETRAIN_HOURS):
    """(training cutoff, first hour, end hour) for each block of `day`: the model scoring a block is trained on all rows before its cutoff."""
    for start, end in zip(hours, list(hours[1:]) + [24]):
        yield day + pd.Timedelta(hours=start), start, end


class V1Model:
    """V1 = LightGBM + factorization machine with fixed iterations (from the V1 tuning run), blended 50/50 in log-odds."""

    def __init__(self, lightgbm_rounds, fm_epochs):
        self.lightgbm_rounds = lightgbm_rounds
        self.fm_epochs = fm_epochs

    def fit(self, train):
        self.encoder_ = FeatureEncoder().fit(train)
        self.lightgbm_ = LightGBMModel(self.encoder_.categorical, num_rounds=self.lightgbm_rounds).fit(
            self.encoder_.transform(train), train["click"].to_numpy())
        self.fm_ = FactorizationMachineModel(num_epochs=self.fm_epochs).fit(train)
        return self

    def predict_components(self, rows):
        """(LightGBM probability, factorization machine probability, V1 blend probability)."""
        p_lightgbm = self.lightgbm_.predict(self.encoder_.transform(rows))
        p_fm = self.fm_.predict(rows)
        return p_lightgbm, p_fm, blend(p_lightgbm, p_fm)
