"""Phase 4: Moirai Zero-Shot Adapter (`Salesforce/moirai-1.1-R-small`)."""

from __future__ import annotations

import argparse
import time
import typing
from typing import Dict, List, Tuple

import jaxtyping
import numpy as np
import torch


class _PyTreeShim:
    """Subscriptable shim so `uni2ts` imports cleanly on PyTorch-only hosts without `jax`."""

    def __class_getitem__(cls, item: typing.Any) -> typing.Any:
        return typing.Any


if not hasattr(jaxtyping, "_tsdomain_pytree_shimmed"):
    try:
        _ = jaxtyping.PyTree
    except Exception:
        jaxtyping.PyTree = _PyTreeShim  # type: ignore[attr-defined]
    jaxtyping._tsdomain_pytree_shimmed = True  # type: ignore[attr-defined]

from uni2ts.model.moirai import MoiraiForecast, MoiraiModule  # noqa: E402

from tsdomain.models.base import TSFMBaseAdapter  # noqa: E402
from tsdomain.utils import (  # noqa: E402
    QUANTILE_COLS,
    QUANTILE_LEVELS,
    detect_device,
    set_global_seed,
)


class MoiraiAdapter(TSFMBaseAdapter):
    """Adapter for Salesforce Moirai (`Salesforce/moirai-1.1-R-small`) via `uni2ts`."""

    def __init__(
        self,
        checkpoint_id: str = "Salesforce/moirai-1.1-R-small",
        device: str = "auto",
        context_length: int = 512,
        patch_size: int = 32,
        num_samples: int = 20,
        batch_size: int = 32,
        seed: int = 20260410,
    ) -> None:
        self.model_id = "Moirai_small"
        self.checkpoint_id = checkpoint_id
        self.param_count = 13_827_528
        self.device = detect_device(device)
        self.context_length = context_length
        self.patch_size = patch_size
        self.num_samples = num_samples
        self.batch_size = batch_size
        self.seed = seed
        self.module: MoiraiModule | None = None

    def load_model(self) -> float:
        t0 = time.perf_counter()
        self.module = MoiraiModule.from_pretrained(self.checkpoint_id)
        self.module.to(self.device)
        self.module.eval()
        self.param_count = int(sum(p.numel() for p in self.module.parameters()))
        return float(time.perf_counter() - t0)

    def predict_batch(
        self,
        contexts: List[np.ndarray],
        horizon: int,
        season_length: int = 1,
    ) -> List[Tuple[np.ndarray, Dict[str, np.ndarray]]]:
        if self.module is None:
            self.load_model()
        assert self.module is not None

        set_global_seed(self.seed)
        forecaster = MoiraiForecast(
            module=self.module,
            prediction_length=int(horizon),
            context_length=int(self.context_length),
            patch_size=int(self.patch_size),
            num_samples=int(self.num_samples),
            target_dim=1,
            feat_dynamic_real_dim=0,
            past_feat_dynamic_real_dim=0,
        ).to(self.device)
        forecaster.eval()

        plen = int(forecaster.past_length)
        results: List[Tuple[np.ndarray, Dict[str, np.ndarray]]] = []

        for start in range(0, len(contexts), self.batch_size):
            batch_ctx = contexts[start : start + self.batch_size]
            bsz = len(batch_ctx)

            past_target = torch.zeros(
                (bsz, plen, 1), dtype=torch.float32, device=self.device
            )
            past_observed = torch.zeros(
                (bsz, plen, 1), dtype=torch.bool, device=self.device
            )
            past_is_pad = torch.ones((bsz, plen), dtype=torch.bool, device=self.device)

            for i, c in enumerate(batch_ctx):
                arr = np.asarray(c, dtype=np.float32)
                arr = arr[np.isfinite(arr)]
                if len(arr) == 0:
                    arr = np.zeros(1, dtype=np.float32)
                if len(arr) > plen:
                    arr = arr[-plen:]
                n_obs = len(arr)
                past_target[i, plen - n_obs :, 0] = torch.from_numpy(arr).to(self.device)
                past_observed[i, plen - n_obs :, 0] = True
                past_is_pad[i, plen - n_obs :] = False

            with torch.inference_mode():
                samples = forecaster(
                    past_target=past_target,
                    past_observed_target=past_observed,
                    past_is_pad=past_is_pad,
                )  # (bsz, num_samples, horizon) or (bsz, num_samples, horizon, 1)

            samples_np = samples.detach().cpu().numpy()
            if samples_np.ndim == 4:
                samples_np = samples_np[..., 0]

            for i in range(bsz):
                s_i = np.clip(samples_np[i], -1e4, 1e4)  # guard extreme heavy-tail Student-t draws
                mean_pred = np.median(s_i, axis=0).astype(float)
                q_dict = {
                    col: np.quantile(s_i, q, axis=0).astype(float)
                    for q, col in zip(QUANTILE_LEVELS, QUANTILE_COLS)
                }
                results.append((mean_pred, q_dict))

        return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Moirai zero-shot inference on a config.")
    parser.add_argument("--config", type=str, default="hospital/M/short")
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    from tsdomain.baselines.run_baselines import load_evaluation_windows_for_config

    adapter = MoiraiAdapter(device=args.device)
    load_sec = adapter.load_model()
    windows = load_evaluation_windows_for_config(args.config)
    fc_df, m_df = adapter.forecast_windows(windows)
    print(
        f"[Moirai_small] {args.config}: load={load_sec:.2f}s, "
        f"MASE={m_df['mase'].mean():.4f}, CRPS={m_df['crps'].mean():.4f}"
    )


if __name__ == "__main__":
    main()
