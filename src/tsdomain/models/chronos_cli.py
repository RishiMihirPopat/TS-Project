"""Phase 4: Chronos Zero-Shot Adapter (`amazon/chronos-t5-small`)."""

from __future__ import annotations

import argparse
import time
from typing import Dict, List, Tuple

import numpy as np
import torch
from chronos import ChronosPipeline

from tsdomain.models.base import TSFMBaseAdapter
from tsdomain.utils import QUANTILE_COLS, QUANTILE_LEVELS, detect_device, set_global_seed


class ChronosAdapter(TSFMBaseAdapter):
    """Adapter for Amazon Chronos (`amazon/chronos-t5-small`) via `chronos-forecasting`."""

    def __init__(
        self,
        checkpoint_id: str = "amazon/chronos-t5-small",
        device: str = "auto",
        context_length: int = 512,
        num_samples: int = 10,
        batch_size: int = 64,
        seed: int = 20260410,
    ) -> None:
        self.model_id = "Chronos_small"
        self.checkpoint_id = checkpoint_id
        self.param_count = 46_000_000
        self.device = detect_device(device)
        self.context_length = context_length
        self.num_samples = num_samples
        self.batch_size = batch_size
        self.seed = seed
        self.pipeline: ChronosPipeline | None = None

    def load_model(self) -> float:
        t0 = time.perf_counter()
        dtype = torch.float32 if self.device == "cpu" else torch.bfloat16
        self.pipeline = ChronosPipeline.from_pretrained(
            self.checkpoint_id,
            device_map=self.device,
            torch_dtype=dtype,
        )
        # Update exact parameter count from loaded model
        if hasattr(self.pipeline, "model") and hasattr(self.pipeline.model, "model"):
            self.param_count = int(
                sum(p.numel() for p in self.pipeline.model.model.parameters())
            )
        return float(time.perf_counter() - t0)

    def predict_batch(
        self,
        contexts: List[np.ndarray],
        horizon: int,
        season_length: int = 1,
    ) -> List[Tuple[np.ndarray, Dict[str, np.ndarray]]]:
        if self.pipeline is None:
            self.load_model()
        assert self.pipeline is not None

        set_global_seed(self.seed)
        results: List[Tuple[np.ndarray, Dict[str, np.ndarray]]] = []

        for start in range(0, len(contexts), self.batch_size):
            batch_ctx = [
                torch.tensor(np.asarray(c, dtype=np.float32), dtype=torch.float32)
                for c in contexts[start : start + self.batch_size]
            ]
            with torch.inference_mode():
                # ChronosPipeline.predict returns tensor of shape (batch_size, num_samples, horizon)
                samples = self.pipeline.predict(
                    batch_ctx,
                    prediction_length=int(horizon),
                    num_samples=int(self.num_samples),
                    limit_prediction_length=False,
                )
            samples_np = samples.detach().cpu().numpy()
            for i in range(samples_np.shape[0]):
                s_i = samples_np[i]  # (num_samples, horizon)
                mean_pred = np.median(s_i, axis=0)
                q_dict = {
                    col: np.quantile(s_i, q, axis=0)
                    for q, col in zip(QUANTILE_LEVELS, QUANTILE_COLS)
                }
                results.append((mean_pred, q_dict))

        return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Chronos zero-shot inference on a config.")
    parser.add_argument("--config", type=str, default="hospital/M/short")
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    from tsdomain.baselines.run_baselines import load_evaluation_windows_for_config

    adapter = ChronosAdapter(device=args.device)
    load_sec = adapter.load_model()
    windows = load_evaluation_windows_for_config(args.config)
    fc_df, m_df = adapter.forecast_windows(windows)
    print(
        f"[Chronos_small] {args.config}: load={load_sec:.2f}s, "
        f"MASE={m_df['mase'].mean():.4f}, CRPS={m_df['crps'].mean():.4f}"
    )


if __name__ == "__main__":
    main()
