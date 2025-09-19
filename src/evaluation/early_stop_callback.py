from typing import Optional

from transformers import TrainerCallback

try:
    import wandb  # type: ignore
except Exception:  # pragma: no cover
    wandb = None


class SimpleEarlyStoppingCallback(TrainerCallback):
    """Early stop without requiring load_best_model_at_end.

    Stops training if the monitored metric does not improve for `patience`
    evaluation events. Uses `greater_is_better` and `min_delta` to decide
    whether a value counts as an improvement.
    """

    def __init__(
        self,
        metric_name: str = "eval_loss",
        greater_is_better: bool = False,
        patience: int = 3,
        min_delta: float = 0.0,
    ) -> None:
        self.metric_name = metric_name
        self.greater_is_better = greater_is_better
        self.patience = patience
        self.min_delta = min_delta

        self.best_value: Optional[float] = None
        self.num_bad_epochs: int = 0

    def _is_improvement(self, current: float) -> bool:
        if self.best_value is None:
            return True
        if self.greater_is_better:
            return current > (self.best_value + self.min_delta)
        else:
            return current < (self.best_value - self.min_delta)

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):  # noqa: N802
        if not metrics or self.metric_name not in metrics:
            return
        val = metrics[self.metric_name]
        if not isinstance(val, (int, float)):
            return

        if self._is_improvement(val):
            self.best_value = float(val)
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1
            if self.num_bad_epochs >= self.patience:
                control.should_training_stop = True
                if wandb is not None and getattr(wandb, "run", None) is not None:
                    wandb.run.summary["early_stop_triggered"] = True
                    wandb.run.summary["early_stop_metric"] = self.metric_name
                    wandb.run.summary["early_stop_at_step"] = state.global_step

