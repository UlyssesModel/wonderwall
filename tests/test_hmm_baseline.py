"""Tests for the HMM baseline.

Three layers of test:
  - feature extraction shape
  - Viterbi correctness on a tiny hand-built HMM
  - end-to-end Kirk → HMM regime classification
"""
from __future__ import annotations

import numpy as np
import torch

from eval.hmm_baseline import (
    GaussianHMM,
    classify_stream_with_hmm,
    evaluate_regime_accuracy,
    features_from_kirk_output,
)
from wonderwall.kirk_client import StubKirkClient


def test_features_shape_is_4dim():
    kirk = StubKirkClient(n=32)
    ko = kirk.infer(torch.randn(32, 32))
    feats = features_from_kirk_output(ko)
    assert feats.shape == (4,)
    assert feats.dtype == np.float64


def test_features_finite():
    kirk = StubKirkClient(n=16)
    ko = kirk.infer(torch.randn(16, 16))
    feats = features_from_kirk_output(ko)
    assert np.all(np.isfinite(feats))


def test_default_init_uses_uniform_pi_and_sticky_a():
    hmm = GaussianHMM(n_states=4, n_features=4)
    assert hmm.pi.shape == (4,)
    assert np.allclose(hmm.pi.sum(), 1.0)
    # Sticky transitions: diagonal should dominate
    assert all(hmm.A[i, i] > 0.5 for i in range(4))


def test_viterbi_recovers_obvious_sequence():
    """When the means are well-separated, Viterbi should recover the right state."""
    hmm = GaussianHMM(n_states=2, n_features=1)
    hmm.mu = np.array([[-5.0], [5.0]])
    hmm.sigma = np.array([[1.0], [1.0]])
    hmm.A = np.array([[0.95, 0.05], [0.05, 0.95]])
    hmm.pi = np.array([0.5, 0.5])

    obs = np.array([[-5], [-5], [-5], [5], [5], [5]], dtype=np.float64)
    states = hmm.viterbi(obs)
    assert states.tolist() == [0, 0, 0, 1, 1, 1]


def test_evaluate_regime_accuracy_perfect_match():
    pred = np.array([0, 1, 2, 1, 0])
    gold = np.array([0, 1, 2, 1, 0])
    out = evaluate_regime_accuracy(pred, gold)
    assert out["accuracy"] == 1.0
    assert out["n"] == 5


def test_evaluate_regime_accuracy_per_class_metrics():
    pred = np.array([0, 0, 1, 1])
    gold = np.array([0, 1, 1, 1])
    out = evaluate_regime_accuracy(pred, gold)
    # 3 correct out of 4
    assert out["accuracy"] == 0.75
    # class 0: tp=1, fp=1, fn=0
    assert out["precision_class_0"] == 0.5
    assert out["recall_class_0"] == 1.0
    # class 1: tp=2, fp=0, fn=1
    assert out["precision_class_1"] == 1.0
    assert abs(out["recall_class_1"] - 2 / 3) < 1e-9


def test_save_load_round_trip(tmp_path):
    hmm = GaussianHMM(n_states=3, n_features=2)
    hmm.mu = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float64)
    hmm.sigma = np.array([[0.5, 0.5], [0.5, 0.5], [0.5, 0.5]], dtype=np.float64)
    p = tmp_path / "hmm.json"
    hmm.save(str(p))
    loaded = GaussianHMM.load(str(p))
    assert np.allclose(loaded.mu, hmm.mu)
    assert np.allclose(loaded.sigma, hmm.sigma)


def test_fit_supervised_adjusts_means():
    """After supervised fit, per-state means should match labeled data."""
    hmm = GaussianHMM(n_states=2, n_features=1)
    obs_seqs = [np.array([[-5], [-5], [-5]], dtype=np.float64),
                np.array([[5], [5]], dtype=np.float64)]
    label_seqs = [np.array([0, 0, 0]), np.array([1, 1])]
    hmm.fit_supervised(obs_seqs, label_seqs)
    assert hmm.mu[0, 0] < -3
    assert hmm.mu[1, 0] > 3


def test_classify_stream_runs_end_to_end():
    """Kirk → HMM end-to-end with the stub client."""
    kirk = StubKirkClient(n=16)
    hmm = GaussianHMM(n_states=4, n_features=4)
    tensors = [torch.randn(16, 16) * 0.003 for _ in range(10)]
    states, names = classify_stream_with_hmm(kirk, hmm, tensors)
    assert len(states) == 10
    assert len(names) == 10
    assert all(0 <= s < 4 for s in states)
    assert all(n in hmm.state_names for n in names)


# ---------------------------------------------------------------------------
# Baum-Welch (unsupervised EM)
# ---------------------------------------------------------------------------


def _two_cluster_data(n_per_cluster: int = 50, seed: int = 7) -> np.ndarray:
    """Two well-separated clusters: half at -5, half at +5, single feature."""
    rng = np.random.default_rng(seed)
    a = rng.normal(-5, 0.5, size=(n_per_cluster, 1))
    b = rng.normal(+5, 0.5, size=(n_per_cluster, 1))
    return np.concatenate([a, b], axis=0)


def test_baum_welch_converges_on_two_clusters():
    """EM should recover the two-cluster mean structure."""
    obs = _two_cluster_data(n_per_cluster=50)
    hmm = GaussianHMM(n_states=2, n_features=1)
    # Random-ish init away from the truth
    hmm.mu = np.array([[-2.0], [2.0]])
    hmm.sigma = np.array([[1.0], [1.0]])
    hist = hmm.fit_baum_welch([obs], max_iter=30, tol=1e-4, verbose=False)

    # Log-likelihood is monotonically non-decreasing (modulo numerical noise)
    assert hist[-1] >= hist[0] - 1e-3, f"LL decreased: {hist}"

    # Recovered means are close to ±5 (in some order)
    sorted_means = sorted(hmm.mu[:, 0].tolist())
    assert sorted_means[0] < -3, f"low cluster mean too high: {sorted_means}"
    assert sorted_means[1] > 3, f"high cluster mean too low: {sorted_means}"


def test_baum_welch_log_likelihood_history_returned():
    """Sanity: history is non-empty and consists of finite floats."""
    obs = _two_cluster_data(n_per_cluster=20)
    hmm = GaussianHMM(n_states=2, n_features=1)
    hist = hmm.fit_baum_welch([obs], max_iter=5, verbose=False)
    assert len(hist) >= 1
    assert all(np.isfinite(h) for h in hist)


def test_baum_welch_respects_min_sigma():
    """min_sigma floor prevents variance collapse on tight clusters."""
    obs = np.full((30, 1), 5.0) + np.random.default_rng(0).normal(0, 0.001, (30, 1))
    hmm = GaussianHMM(n_states=2, n_features=1)
    hmm.fit_baum_welch([obs], max_iter=10, min_sigma=0.5, verbose=False)
    assert (hmm.sigma >= 0.5 - 1e-9).all()
