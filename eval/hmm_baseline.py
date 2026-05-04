"""HMM baseline for regime classification — apples-to-apples vs Pipeline C.

Per the Kavara × WMD doc, production today runs:

    raw market data → Ulysses → embedding → Hidden Markov Model → regime label

…and beats other approaches by a large margin on rare regimes (crashes, high
volatility). This module reproduces that pipeline at a level sufficient to
populate the `regime_correct` field on EvalRecords, so the conference deck
can compare:

    Pipeline C (Kirk + adapter + LLM narration)
        ROUGE-L vs teacher: X.XX
        regime accuracy:    Y%

vs

    HMM baseline (Kirk + Gaussian-emission HMM)
        regime accuracy:    Z%

The HMM here is intentionally minimal — Gaussian emissions, hand-tuned
transition matrix (or learned via Baum-Welch on a labeled set if available).
For production-fidelity comparison, swap to whatever HMM Joel has in the
existing pipeline.

Features fed to the HMM (per regime-detection step):
  - entropy (scalar)
  - mean of layer2_marginals real component
  - std of layer2_marginals real component
  - mean of layer2_reconstruction diagonal (proxy for self-energy)

These four features form a 4-dim observation vector. State count defaults
to 4 regimes (calm / trending / volatile / crash) following the WMD doc.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np
import torch

from wonderwall.interfaces import KirkClient, KirkOutput, KirkMode


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------


def features_from_kirk_output(ko: KirkOutput) -> np.ndarray:
    """Extract a 4-dim feature vector from one Kirk output.

    Real component only (production path). Layout:
        [0] entropy
        [1] mean of marginals
        [2] std  of marginals
        [3] mean of layer2_reconstruction diagonal
    """
    marg = ko.layer2_marginals.real if torch.is_complex(ko.layer2_marginals) else ko.layer2_marginals
    recon = ko.layer2_reconstruction.real if torch.is_complex(ko.layer2_reconstruction) else ko.layer2_reconstruction
    diag = torch.diagonal(recon)
    return np.array(
        [
            float(ko.entropy.item()),
            float(marg.mean().item()),
            float(marg.std().item()),
            float(diag.mean().item()),
        ],
        dtype=np.float64,
    )


# ---------------------------------------------------------------------------
# Gaussian HMM — minimal implementation, no sklearn / hmmlearn dependency
# ---------------------------------------------------------------------------


@dataclass
class GaussianHMM:
    """Tiny Gaussian-emission HMM.

    Implementation: forward-backward + Viterbi. Uses log-space arithmetic to
    avoid underflow on long sequences. No EM / Baum-Welch — parameters are
    set explicitly (or via :meth:`fit_supervised` if regime labels are
    available).
    """

    n_states: int
    n_features: int

    # Initial probabilities: shape (n_states,)
    pi: np.ndarray = field(default_factory=lambda: np.array([]))
    # Transition matrix: shape (n_states, n_states)
    A: np.ndarray = field(default_factory=lambda: np.array([]))
    # Emission means and covariances (assumed diagonal)
    mu: np.ndarray = field(default_factory=lambda: np.array([]))
    sigma: np.ndarray = field(default_factory=lambda: np.array([]))

    state_names: list[str] = field(
        default_factory=lambda: ["calm", "trending", "volatile", "crash"]
    )

    def __post_init__(self):
        if self.pi.size == 0:
            self.pi = np.full(self.n_states, 1.0 / self.n_states)
        if self.A.size == 0:
            # Sticky transition: 0.85 self, 0.15/(K-1) elsewhere
            stay = 0.85
            switch = (1.0 - stay) / (self.n_states - 1)
            self.A = np.full((self.n_states, self.n_states), switch)
            np.fill_diagonal(self.A, stay)
        if self.mu.size == 0:
            self.mu = np.zeros((self.n_states, self.n_features))
        if self.sigma.size == 0:
            self.sigma = np.ones((self.n_states, self.n_features))

    # -- emissions -----------------------------------------------------------

    def _log_emission(self, obs: np.ndarray) -> np.ndarray:
        """Log emission probability of observation `obs` under each state.

        Diagonal Gaussian: log P(obs | state) = sum_d log N(obs[d] | mu, sigma).
        Returns shape (n_states,).
        """
        var = self.sigma ** 2  # (n_states, n_features)
        diff = obs[None, :] - self.mu  # (n_states, n_features)
        log_norm = -0.5 * (np.log(2 * np.pi * var) + diff ** 2 / var)
        return log_norm.sum(axis=1)

    # -- inference -----------------------------------------------------------

    def viterbi(self, observations: np.ndarray) -> np.ndarray:
        """Most likely state sequence for `observations` of shape (T, n_features)."""
        T = observations.shape[0]
        log_pi = np.log(self.pi + 1e-300)
        log_A = np.log(self.A + 1e-300)

        delta = np.full((T, self.n_states), -np.inf)
        psi = np.zeros((T, self.n_states), dtype=np.int32)

        delta[0] = log_pi + self._log_emission(observations[0])
        for t in range(1, T):
            log_em = self._log_emission(observations[t])
            for j in range(self.n_states):
                trans = delta[t - 1] + log_A[:, j]
                psi[t, j] = int(np.argmax(trans))
                delta[t, j] = trans[psi[t, j]] + log_em[j]

        states = np.zeros(T, dtype=np.int32)
        states[-1] = int(np.argmax(delta[-1]))
        for t in range(T - 2, -1, -1):
            states[t] = psi[t + 1, states[t + 1]]
        return states

    # -- supervised parameter fitting ---------------------------------------

    def fit_supervised(
        self, observations_per_seq: list[np.ndarray], state_labels_per_seq: list[np.ndarray]
    ) -> None:
        """Fit emissions + transitions from labeled sequences.

        Args:
            observations_per_seq: list of (T_i, n_features) arrays
            state_labels_per_seq: list of (T_i,) integer state arrays
        """
        if len(observations_per_seq) != len(state_labels_per_seq):
            raise ValueError("observations and labels must have the same number of sequences")

        # Emission MLE: per-state mean and variance
        all_obs_by_state = [[] for _ in range(self.n_states)]
        for obs, lab in zip(observations_per_seq, state_labels_per_seq):
            for o, l in zip(obs, lab):
                all_obs_by_state[int(l)].append(o)
        for s in range(self.n_states):
            stacked = np.array(all_obs_by_state[s] or [np.zeros(self.n_features)])
            self.mu[s] = stacked.mean(axis=0)
            self.sigma[s] = stacked.std(axis=0) + 1e-3

        # Transition MLE
        counts = np.zeros((self.n_states, self.n_states))
        for lab in state_labels_per_seq:
            for i in range(len(lab) - 1):
                counts[int(lab[i]), int(lab[i + 1])] += 1
        # Laplace smoothing
        counts += 0.5
        self.A = counts / counts.sum(axis=1, keepdims=True)

        # Initial probabilities
        first_state_counts = np.zeros(self.n_states)
        for lab in state_labels_per_seq:
            first_state_counts[int(lab[0])] += 1
        first_state_counts += 0.5
        self.pi = first_state_counts / first_state_counts.sum()

    # -- unsupervised: Baum-Welch (forward-backward + EM) -------------------

    def _forward_backward_log(
        self, observations: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """Log-space forward-backward.

        Returns:
            log_alpha: (T, K) — log P(o_{1..t}, x_t = k)
            log_beta:  (T, K) — log P(o_{t+1..T} | x_t = k)
            log_lik:   scalar log P(o_{1..T})
        """
        T = observations.shape[0]
        K = self.n_states
        log_pi = np.log(self.pi + 1e-300)
        log_A = np.log(self.A + 1e-300)

        log_em = np.zeros((T, K))
        for t in range(T):
            log_em[t] = self._log_emission(observations[t])

        # Forward
        log_alpha = np.full((T, K), -np.inf)
        log_alpha[0] = log_pi + log_em[0]
        for t in range(1, T):
            for j in range(K):
                # logsumexp over previous-state contributions
                contrib = log_alpha[t - 1] + log_A[:, j]
                m = contrib.max()
                log_alpha[t, j] = m + np.log(np.exp(contrib - m).sum()) + log_em[t, j]

        # Total log-likelihood from final alpha column
        m = log_alpha[-1].max()
        log_lik = m + np.log(np.exp(log_alpha[-1] - m).sum())

        # Backward
        log_beta = np.full((T, K), -np.inf)
        log_beta[-1] = 0.0
        for t in range(T - 2, -1, -1):
            for i in range(K):
                contrib = log_A[i, :] + log_em[t + 1] + log_beta[t + 1]
                m = contrib.max()
                log_beta[t, i] = m + np.log(np.exp(contrib - m).sum())

        return log_alpha, log_beta, log_lik

    def _e_step(
        self, observations: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """E-step: posterior gamma and pairwise xi from one sequence.

        gamma: (T, K)        posterior over states at each step
        xi:    (T-1, K, K)   posterior over consecutive state pairs
        log_lik: scalar
        """
        T = observations.shape[0]
        K = self.n_states
        log_alpha, log_beta, log_lik = self._forward_backward_log(observations)

        # gamma_t(k) = P(x_t = k | obs)
        log_gamma = log_alpha + log_beta - log_lik
        gamma = np.exp(log_gamma)
        gamma /= gamma.sum(axis=1, keepdims=True) + 1e-300

        # xi_t(i,j) = P(x_t=i, x_{t+1}=j | obs)
        xi = np.zeros((T - 1, K, K))
        log_A = np.log(self.A + 1e-300)
        for t in range(T - 1):
            log_em_next = self._log_emission(observations[t + 1])
            log_xi_t = (
                log_alpha[t][:, None]
                + log_A
                + log_em_next[None, :]
                + log_beta[t + 1][None, :]
                - log_lik
            )
            m = log_xi_t.max()
            xi_t = np.exp(log_xi_t - m)
            xi_t /= xi_t.sum() + 1e-300
            xi[t] = xi_t

        return gamma, xi, log_lik

    def fit_baum_welch(
        self,
        observations_per_seq: list[np.ndarray],
        max_iter: int = 50,
        tol: float = 1e-3,
        min_sigma: float = 1e-3,
        verbose: bool = False,
    ) -> list[float]:
        """Unsupervised parameter estimation via Baum-Welch (EM).

        Args:
            observations_per_seq: list of (T_i, n_features) sequences
            max_iter: hard cap on EM iterations
            tol: stop when total log-likelihood improves by less than this
            min_sigma: variance floor — prevents Σ → 0 collapse on small clusters
            verbose: print log-likelihood per iteration

        Returns: list of total log-likelihoods, one per iteration.

        NOTE: Baum-Welch finds a local optimum. For better behavior:
          - Initialize via fit_supervised on a small labeled subset, then EM
          - Run multiple random restarts and pick the best LL
          - Use min_sigma to avoid degenerate clusters
        """
        K = self.n_states
        D = self.n_features
        history: list[float] = []

        for it in range(max_iter):
            # E-step
            total_ll = 0.0
            gammas: list[np.ndarray] = []
            xis: list[np.ndarray] = []
            for obs in observations_per_seq:
                g, xi, ll = self._e_step(obs)
                gammas.append(g)
                xis.append(xi)
                total_ll += ll
            history.append(float(total_ll))
            if verbose:
                print(f"[bw] iter={it} log_lik={total_ll:.4f}")

            # Convergence check
            if len(history) >= 2 and abs(history[-1] - history[-2]) < tol:
                if verbose:
                    print(f"[bw] converged at iter={it}")
                break

            # M-step: update pi, A, mu, sigma
            # pi: average gamma_0 across sequences
            pi_sum = np.zeros(K)
            for g in gammas:
                pi_sum += g[0]
            self.pi = pi_sum / max(1, len(gammas))
            self.pi = self.pi / (self.pi.sum() + 1e-300)

            # A: aggregate xi
            A_num = np.zeros((K, K))
            A_den = np.zeros(K)
            for xi, g in zip(xis, gammas):
                A_num += xi.sum(axis=0)
                A_den += g[:-1].sum(axis=0)
            A_num += 1e-3  # smoothing
            A_den = A_den + K * 1e-3
            self.A = A_num / A_den[:, None]

            # mu, sigma: weighted by gamma
            mu_new = np.zeros((K, D))
            mu_den = np.zeros(K)
            for g, obs in zip(gammas, observations_per_seq):
                mu_new += g.T @ obs
                mu_den += g.sum(axis=0)
            self.mu = mu_new / (mu_den[:, None] + 1e-300)

            sig_new = np.zeros((K, D))
            for g, obs in zip(gammas, observations_per_seq):
                diff = obs[:, None, :] - self.mu[None, :, :]   # (T, K, D)
                sig_new += (g[:, :, None] * diff ** 2).sum(axis=0)
            self.sigma = np.sqrt(sig_new / (mu_den[:, None] + 1e-300))
            self.sigma = np.maximum(self.sigma, min_sigma)

        return history

    # -- serialization ------------------------------------------------------

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump(
                {
                    "n_states": self.n_states,
                    "n_features": self.n_features,
                    "pi": self.pi.tolist(),
                    "A": self.A.tolist(),
                    "mu": self.mu.tolist(),
                    "sigma": self.sigma.tolist(),
                    "state_names": self.state_names,
                },
                f,
                indent=2,
            )

    @classmethod
    def load(cls, path: str) -> "GaussianHMM":
        with open(path) as f:
            data = json.load(f)
        hmm = cls(
            n_states=data["n_states"],
            n_features=data["n_features"],
            pi=np.array(data["pi"]),
            A=np.array(data["A"]),
            mu=np.array(data["mu"]),
            sigma=np.array(data["sigma"]),
        )
        hmm.state_names = data.get("state_names", hmm.state_names)
        return hmm


# ---------------------------------------------------------------------------
# Convenience: full Kirk → HMM regime classification
# ---------------------------------------------------------------------------


def classify_stream_with_hmm(
    kirk: KirkClient,
    hmm: GaussianHMM,
    tensors: Sequence[torch.Tensor],
    mode: KirkMode = KirkMode.ACTIVE_INFERENCE,
) -> tuple[np.ndarray, list[str]]:
    """Run a stream of input tensors through Kirk + HMM, return regime labels.

    Returns (state_indices, state_names_per_step).
    """
    kos = kirk.infer_stream(tensors, mode=mode)
    feats = np.stack([features_from_kirk_output(ko) for ko in kos])
    states = hmm.viterbi(feats)
    return states, [hmm.state_names[s] for s in states]


def evaluate_regime_accuracy(
    pred_states: np.ndarray, gold_states: np.ndarray
) -> dict[str, float]:
    """Per-state precision/recall + overall accuracy for a labeled sequence."""
    if len(pred_states) != len(gold_states):
        raise ValueError("pred and gold must have the same length")
    correct = (pred_states == gold_states).sum()
    accuracy = float(correct / len(pred_states))

    out = {"accuracy": accuracy, "n": len(pred_states)}
    classes = np.unique(np.concatenate([pred_states, gold_states]))
    for c in classes:
        tp = int(((pred_states == c) & (gold_states == c)).sum())
        fp = int(((pred_states == c) & (gold_states != c)).sum())
        fn = int(((pred_states != c) & (gold_states == c)).sum())
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        out[f"precision_class_{c}"] = precision
        out[f"recall_class_{c}"] = recall
    return out
