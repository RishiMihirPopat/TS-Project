"""Phase 4 (and P1): Tiny Time Mixers Zero-Shot Adapter (`ibm-granite/granite-timeseries-ttm-r2`)."""

from __future__ import annotations

import argparse
import time
from typing import Dict, List, Tuple

import numpy as np
import torch
from tsfm_public.models.tinytimemixer import TinyTimeMixerForPrediction

from tsdomain.models.base import TSFMBaseAdapter, gaussian_quantiles_from_context
from tsdomain.utils import detect_device, set_global_seed


class TTMAdapter(TSFMBaseAdapter):
    """Adapter for IBM Granite TinyTimeMixer (`ibm-granite/granite-timeseries-ttm-r2`)."""

    def __init__(
        self,
        checkpoint_id: str = "ibm-granite/granite-timeseries-ttm-r2",
        device: str = "auto",
        context_length: int = 512,
        batch_size: int = 64,
        seed: int = 20260410,
    ) -> None:
        self.model_id = "TTM-R2-Pretrained"
        self.checkpoint_id = checkpoint_id
        self.param_count = 805_280
        self.device = detect_device(device)
        self.context_length = context_length
        self.batch_size = batch_size
        self.seed = seed
        self.model: TinyTimeMixerForPrediction | None = None

    def load_model(self) -> float:
        t0 = time.perf_counter()
        self.model = TinyTimeMixerForPrediction.from_pretrained(self.checkpoint_id)
        self.model.to(self.device)
        self.model.eval()
        self.param_count = int(sum(p.numel() for p in self.model.parameters()))
        return float(time.perf_counter() - t0)

    def _prepare_context_tensor(self, ctx: np.ndarray, req_len: int = 512) -> np.ndarray:
        """Ensure context has length `req_len` by left-padding with edge/periodic continuation."""
        arr = np.asarray(ctx, dtype=np.float32)
        if len(arr) >= req_len:
            return arr[-req_len:]
        if len(arr) == 0:
            return np.zeros(req_len, dtype=np.float32)
        pad_len = req_len - len(arr)
        # Pad on the left by edge value to preserve recent dynamics in the right-aligned patches
        pad = np.full(pad_len, arr[0], dtype=np.float32)
        return np.concatenate([pad, arr], axis=0)

    def predict_batch(
        self,
        contexts: List[np.ndarray],
        horizon: int,
        season_length: int = 1,
    ) -> List[Tuple[np.ndarray, Dict[str, np.ndarray]]]:
        if self.model is None:
            self.load_model()
        assert self.model is not None

        set_global_seed(self.seed)
        req_len = int(getattr(self.model.config, "context_length", 512))
        pred_len = int(getattr(self.model.config, "prediction_length", 96))

        results: List[Tuple[np.ndarray, Dict[str, np.ndarray]]] = []

        for start in range(0, len(contexts), self.batch_size):
            batch_raw = contexts[start : start + self.batch_size]
            batch_arr = np.stack(
                [self._prepare_context_tensor(c, req_len=req_len) for c in batch_raw],
                axis=0,
            )  # (B, req_len)

            preds_collected = []
            rem = int(horizon)
            curr_arr = batch_arr.copy()

            with torch.inference_mode():
                while rem > 0:
                    x_tensor = torch.tensor(
                        curr_arr[:, :, None], dtype=torch.float32, device=self.device
                    )
                    out = self.model(past_values=x_tensor)
                    step_out = (
                        out.prediction_outputs[:, :, 0].detach().cpu().numpy()
                    )  # (B, pred_len)
                    take = min(rem, pred_len)
                    preds_collected.append(step_out[:, :take])
                    rem -= take
                    if rem > 0:
                        curr_arr = np.concatenate(
                            [curr_arr[:, take:], step_out[:, :take]], axis=1
                        )

            full_mean = np.concatenate(preds_collected, axis=1)  # (B, horizon)
            for i, raw_ctx in enumerate(batch_raw):
                mean_i = full_mean[i].astype(float)
                q_dict = gaussian_quantiles_from_context(
                    mean_i, raw_ctx, season_length=season_length
                )
                results.append((mean_i, q_dict))

        return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Run TTM zero-shot inference on a config.")
    parser.add_argument("--config", type=str, default="hospital/M/short")
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    from tsdomain.baselines.run_baselines import load_evaluation_windows_for_config

    adapter = TTMAdapter(device=args.device)
    load_sec = adapter.load_model()
    windows = load_evaluation_windows_for_config(args.config)
    fc_df, m_df = adapter.forecast_windows(windows)
    print(
        f"[TTM_R2] {args.config}: load={load_sec:.2f}s, "
        f"MASE={m_df['mase'].mean():.4f}, CRPS={m_df['crps'].mean():.4f}"
    )


if __name__ == "__main__":
    main()
