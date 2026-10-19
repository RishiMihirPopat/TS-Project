"""Phase 4: TimesFM Zero-Shot Adapter (`google/timesfm-2.5-200m-pytorch`)."""

from __future__ import annotations

import argparse
import time
from typing import Dict, List, Tuple

import numpy as np
import timesfm

from tsdomain.models.base import TSFMBaseAdapter
from tsdomain.utils import QUANTILE_COLS, detect_device, set_global_seed


class TimesFMAdapter(TSFMBaseAdapter):
    """Adapter for Google TimesFM 200M (`google/timesfm-2.5-200m-pytorch`) via `timesfm`."""

    def __init__(
        self,
        checkpoint_id: str = "google/timesfm-2.5-200m-pytorch",
        device: str = "auto",
        context_length: int = 512,
        batch_size: int = 32,
        seed: int = 20260410,
    ) -> None:
        self.model_id = "TimesFM"
        self.checkpoint_id = checkpoint_id
        self.param_count = 231_289_280
        self.device = detect_device(device)
        self.context_length = context_length
        self.batch_size = batch_size
        self.seed = seed
        self.tfm: timesfm.TimesFM_2p5_200M_torch | None = None
        self._compiled_max_horizon: int = 0

    def load_model(self) -> float:
        t0 = time.perf_counter()
        self.tfm = timesfm.TimesFM_2p5_200M_torch.from_pretrained(
            self.checkpoint_id,
            torch_compile=False,
        )
        self.param_count = int(
            sum(p.numel() for p in self.tfm.model.parameters())
        )
        self._ensure_compiled(max_horizon=128)
        return float(time.perf_counter() - t0)

    def _ensure_compiled(self, max_horizon: int) -> None:
        assert self.tfm is not None
        req_h = max(128, int(max_horizon))
        if self._compiled_max_horizon >= req_h:
            return
        self.tfm.compile(
            timesfm.ForecastConfig(
                max_context=int(self.context_length),
                max_horizon=req_h,
                normalize_inputs=True,
                use_continuous_quantile_head=True,
                force_flip_invariance=True,
                infer_is_positive=False,
                fix_quantile_crossing=True,
                per_core_batch_size=int(self.batch_size),
            )
        )
        self._compiled_max_horizon = req_h

    def predict_batch(
        self,
        contexts: List[np.ndarray],
        horizon: int,
        season_length: int = 1,
    ) -> List[Tuple[np.ndarray, Dict[str, np.ndarray]]]:
        if self.tfm is None:
            self.load_model()
        assert self.tfm is not None
        self._ensure_compiled(max_horizon=int(horizon))

        set_global_seed(self.seed)
        clean_inputs = [
            np.asarray(c, dtype=np.float32)[np.isfinite(c)]
            if np.any(np.isfinite(c))
            else np.zeros(8, dtype=np.float32)
            for c in contexts
        ]

        n_inputs = len(clean_inputs)
        point_fc, quant_fc = self.tfm.forecast(
            horizon=int(horizon),
            inputs=list(clean_inputs),
        )
        # point_fc: (N, horizon), quant_fc: (N, horizon, 10) where index 0 is mean and 1..9 are q0.1..q0.9
        results: List[Tuple[np.ndarray, Dict[str, np.ndarray]]] = []
        for i in range(n_inputs):
            mean_pred = np.asarray(point_fc[i, :horizon], dtype=float)
            q_dict = {
                col: np.asarray(quant_fc[i, :horizon, idx + 1], dtype=float)
                for idx, col in enumerate(QUANTILE_COLS)
            }
            results.append((mean_pred, q_dict))

        return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Run TimesFM zero-shot inference on a config.")
    parser.add_argument("--config", type=str, default="hospital/M/short")
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    from tsdomain.baselines.run_baselines import load_evaluation_windows_for_config

    adapter = TimesFMAdapter(device=args.device)
    load_sec = adapter.load_model()
    windows = load_evaluation_windows_for_config(args.config)
    fc_df, m_df = adapter.forecast_windows(windows)
    print(
        f"[TimesFM] {args.config}: load={load_sec:.2f}s, "
        f"MASE={m_df['mase'].mean():.4f}, CRPS={m_df['crps'].mean():.4f}"
    )


if __name__ == "__main__":
    main()
