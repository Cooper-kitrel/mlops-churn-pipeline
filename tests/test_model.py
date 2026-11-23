"""Model validation tests: train on a small sample and check behaviour."""
import numpy as np
import pytest
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from src.preprocess import prepare_data
from src.train import build_model
from src.utils import apply_overrides


@pytest.fixture(scope="module")
def trained(raw_df, config):
    """Train the configured model on a 2,000-row sample with a fixed holdout."""
    sample = raw_df.sample(n=2000, random_state=0)
    X, y = prepare_data(sample, config)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=0, stratify=y)
    model = build_model(config).fit(X_train, y_train)
    return model, X_test, y_test


def test_predictions_have_correct_type_and_shape(trained):
    model, X_test, _ = trained
    preds = model.predict(X_test)
    proba = model.predict_proba(X_test)
    assert preds.shape == (len(X_test),)
    assert set(np.unique(preds)) <= {0, 1}
    assert proba.shape == (len(X_test), 2)
    assert np.all((proba >= 0) & (proba <= 1))
    assert np.allclose(proba.sum(axis=1), 1.0)


def test_model_meets_minimum_performance(trained):
    model, X_test, y_test = trained
    auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    assert auc >= 0.75, f"ROC-AUC {auc:.3f} below 0.75 on small-sample holdout"


def test_model_beats_random_baseline(trained):
    model, X_test, y_test = trained
    auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    assert auc > 0.5 + 0.15


@pytest.mark.parametrize("model_type", ["logistic_regression", "random_forest",
                                        "gradient_boosting"])
def test_all_configured_model_types_train(raw_df, config, model_type):
    params = {"logistic_regression": {"max_iter": 500},
              "random_forest": {"n_estimators": 20},
              "gradient_boosting": {"n_estimators": 20}}[model_type]
    cfg = apply_overrides(
        config, {"model.type": model_type, "model.params": params})
    X, y = prepare_data(raw_df.sample(n=500, random_state=1), cfg)
    model = build_model(cfg).fit(X, y)
    assert model.predict(X.head(10)).shape == (10,)


def test_unknown_model_type_raises(config):
    cfg = apply_overrides(config, {"model.type": "xgboost_please"})
    with pytest.raises(ValueError, match="Unknown model type"):
        build_model(cfg)
