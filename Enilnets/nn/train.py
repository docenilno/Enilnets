"""NeuralNet.TrainBatch/Train/compute_accuracy/compute_precision_recall_f1,
plus the standalone LRScheduler class."""
import math
from typing import Any, Dict, List, Optional, Tuple

from ..core.backend import np
from ..core.utils import iterate_minibatches

def TrainBatch(self: Any, xs: Any, ys: Any, loss_function: Optional[str] = None,
                accumulation_steps: int = 1, **loss_kwargs: Any) -> Tuple[float, Any]:
    """One forward+backward+update step. Returns (loss, predictions).

    accumulation_steps > 1: accumulates gradients over that many calls
    before actually updating weights, for a larger effective batch size
    without more memory. Caller must invoke TrainBatch exactly
    accumulation_steps times per update (Train() does this automatically)."""
    out = self.Forward(xs, training=True)
    if loss_function is None:
        loss_function = "cross_entropy" if self.layers[-1].get("activation") == "softmax" else "mse"
    loss = self.ComputeLoss(out, ys, loss_function, **loss_kwargs)
    self.Backward(ys, loss_function=loss_function, **loss_kwargs)
    if self.grad_clip_norm > 0:
        self.clip_gradients(self.grad_clip_norm)
    if accumulation_steps <= 1:
        self.update()
    else:
        self.accumulate_gradients()
        if self._accum_steps >= accumulation_steps:
            self.apply_accumulated_gradients()
    return loss, out

def compute_accuracy(self: Any, predictions: Any, targets: Any) -> float:
    if predictions.shape[-1] > 1:  # Multi-class
        # axis=-1 (not 1): identical for the common (B, C) case, and correct
        # for (B, S, V) sequence output where axis=1 would argmax over the
        # sequence dimension instead of the classes.
        pred_classes = np.argmax(predictions, axis=-1)
        true_classes = np.argmax(targets, axis=-1)
    else:  # Binary
        pred_classes = (predictions > 0.5).astype(int).flatten()
        true_classes = targets.flatten()
    return float(np.mean(pred_classes == true_classes))

def compute_precision_recall_f1(self: Any, predictions: Any, targets: Any) -> Dict[str, float]:
    """Compute precision, recall, and F1 score for binary classification."""
    if predictions.shape[-1] > 1:
        pred_classes = np.argmax(predictions, axis=1)
        true_classes = np.argmax(targets, axis=1)
    else:
        pred_classes = (predictions > 0.5).astype(int).flatten()
        true_classes = targets.flatten()

    tp = np.sum((pred_classes == 1) & (true_classes == 1))
    fp = np.sum((pred_classes == 1) & (true_classes == 0))
    fn = np.sum((pred_classes == 0) & (true_classes == 1))

    precision = float(tp / (tp + fp + 1e-12))
    recall = float(tp / (tp + fn + 1e-12))
    f1 = float(2 * precision * recall / (precision + recall + 1e-12))
    return {"precision": precision, "recall": recall, "f1": f1}

def Train(self: Any, X_train: Any, Y_train: Optional[Any] = None,
          epochs: int = 10, batch_size: int = 32,
          X_val: Optional[Any] = None, Y_val: Optional[Any] = None,
          loss_function: Optional[str] = None, verbose: bool = True,
          scheduler: Optional["LRScheduler"] = None, early_stopping: Optional[Any] = None,
          accumulation_steps: int = 1, callbacks: Optional[List[Any]] = None,
          **loss_kwargs: Any) -> Dict[str, List[float]]:
    """Train for `epochs`, returning a history dict.

    X_train may instead be a `DataLoader` (with Y_train left None), in which
    case batching, shuffling and any per-batch transform come from it and
    `batch_size` is ignored.

    early_stopping: an EarlyStopping instance monitoring val_loss (when
        X_val/Y_val are given) or train loss otherwise.
    callbacks: duck-typed objects, no base class. `on_epoch_end(epoch, logs,
        model=self)` runs after `history` is updated and before the
        early-stopping check, with logs carrying this epoch's
        loss/accuracy/lr (plus val_loss/val_accuracy if validating);
        `on_train_end(history)` runs once at the end. Missing methods are
        skipped."""
    history = {"loss": [], "val_loss": [], "accuracy": [], "val_accuracy": [], "lr": []}
    # A DataLoader owns its own batching/shuffling/augmentation, so it is
    # used in place of iterate_minibatches rather than alongside it.
    loader = X_train if hasattr(X_train, "_map_style_batches") else None
    if loader is None and Y_train is None:
        raise ValueError(
            "Train() needs Y_train, unless X_train is a DataLoader carrying "
            "its own labels.")
    prev_metric = None
    for epoch in range(epochs):
        if scheduler is not None:
            lr = scheduler.step(epoch, metric=prev_metric)
            self.set_lr(lr)

        epoch_loss = 0.0
        epoch_acc = 0.0
        total_samples = 0
        batches = loader if loader is not None else iterate_minibatches(
            X_train, Y_train, batch_size, shuffle=True)
        for X_batch, Y_batch in batches:
            loss, preds = self.TrainBatch(X_batch, Y_batch, loss_function=loss_function,
                                          accumulation_steps=accumulation_steps, **loss_kwargs)
            batch_size_actual = X_batch.shape[0]
            epoch_loss += loss * batch_size_actual
            epoch_acc += self.compute_accuracy(preds, Y_batch) * batch_size_actual
            total_samples += batch_size_actual
        avg_loss = epoch_loss / total_samples
        avg_acc = epoch_acc / total_samples
        history["loss"].append(avg_loss)
        history["accuracy"].append(avg_acc)
        history["lr"].append(self.learning_rate)
        if X_val is not None and Y_val is not None:
            val_pred = self.Forward(X_val, training=False)
            val_loss = self.ComputeLoss(val_pred, Y_val, loss_function if loss_function is not None else ("cross_entropy" if self.layers[-1].get("activation") == "softmax" else "mse"), **loss_kwargs)
            val_acc = self.compute_accuracy(val_pred, Y_val)
            history["val_loss"].append(val_loss)
            history["val_accuracy"].append(val_acc)
            if verbose:
                print(f"Epoch {epoch+1}/{epochs} - loss: {avg_loss:.4f} - acc: {avg_acc:.4f} - val_loss: {val_loss:.4f} - val_acc: {val_acc:.4f} - lr: {self.learning_rate:.6f}")
            monitored = val_loss
        else:
            if verbose:
                print(f"Epoch {epoch+1}/{epochs} - loss: {avg_loss:.4f} - acc: {avg_acc:.4f} - lr: {self.learning_rate:.6f}")
            monitored = avg_loss
        prev_metric = monitored

        logs = {"loss": avg_loss, "accuracy": avg_acc, "lr": self.learning_rate}
        if X_val is not None and Y_val is not None:
            logs["val_loss"] = val_loss
            logs["val_accuracy"] = val_acc
        for cb in (callbacks or []):
            getattr(cb, "on_epoch_end", lambda *a, **k: None)(epoch, logs, model=self)

        if early_stopping is not None and early_stopping.step(monitored):
            if verbose:
                print(f"Early stopping at epoch {epoch+1} (no improvement for {early_stopping.patience} epochs)")
            break
    for cb in (callbacks or []):
        getattr(cb, "on_train_end", lambda *a, **k: None)(history)
    return history

class LRScheduler:
    """Learning-rate schedules. `step(epoch, metric=None) -> float` is the
    whole contract; Train() calls it once per epoch, with a one-epoch lag so
    the LR used *during* an epoch reflects metrics known *before* it started.
    Every schedule is clamped past its horizon and never returns a negative
    rate. See the README for what each is for.

    mode, with its kwargs (defaults shown):
      "step" drop=0.5, epochs_drop=10 | "exponential" decay=0.95
      "cosine" max_epochs=100 | "warmup_cosine" +warmup_epochs=5
      "polynomial" max_epochs=100, power=1.0, end_lr=0.0
      "cyclic" base_lr, max_lr, step_size=10, policy="triangular", gamma=0.999
      "one_cycle" max_lr, max_epochs=100, pct_start=0.3, div_factor=25,
                  final_div_factor=1e4
      "cosine_warm_restarts" T_0=10, T_mult=1, eta_min=0.0
      "swa" swa_start=0, swa_lr=0.05, anneal_epochs=5
      "lambda" lr_lambda=fn(epoch) -> multiplier (required)
      "sequential" schedulers=[...], milestones=[...] (one fewer milestone)
      "plateau" factor=0.5, patience=10, min_delta=0.0, metric_mode="min";
                needs step(..., metric=). `metric_mode`, not `mode`.
    """
    def __init__(self, initial_lr: float, mode: str = "step", **kwargs: Any) -> None:
        self.initial_lr = initial_lr
        self.mode = mode
        self.kwargs = kwargs
        self._plateau_lr = initial_lr
        self._plateau_best = None
        self._plateau_bad_epochs = 0

    def step(self, epoch: int, metric: Optional[float] = None) -> float:
        """The learning rate for `epoch`, always as a Python float."""
        # float() is not cosmetic: set_lr() stores whatever this returns as
        # the model's learning_rate, so a backend scalar here would drag a
        # 0-d device array through every subsequent weight update and into
        # history["lr"].
        return float(self._lr_for(epoch, metric))

    def _lr_for(self, epoch: int, metric: Optional[float]) -> float:
        if self.mode == "step":
            drop = self.kwargs.get("drop", 0.5)
            epochs_drop = self.kwargs.get("epochs_drop", 10)
            return self.initial_lr * (drop ** (epoch // epochs_drop))
        elif self.mode == "exponential":
            decay = self.kwargs.get("decay", 0.95)
            return self.initial_lr * (decay ** epoch)
        elif self.mode == "cosine":
            max_epochs = self.kwargs.get("max_epochs", 100)
            # Clamp: past max_epochs the cosine argument exceeds pi and the
            # formula would go NEGATIVE (an actively destructive LR, not
            # just a small one) -- hold at the annealed floor instead.
            epoch = min(epoch, max_epochs)
            return self.initial_lr * 0.5 * (1 + math.cos(math.pi * epoch / max_epochs))
        elif self.mode == "warmup_cosine":
            max_epochs = self.kwargs.get("max_epochs", 100)
            warmup_epochs = self.kwargs.get("warmup_epochs", 5)
            if epoch < warmup_epochs:
                return self.initial_lr * (epoch / warmup_epochs)
            else:
                epoch = min(epoch, max_epochs)  # same negative-LR clamp as "cosine"
                return self.initial_lr * 0.5 * (1 + math.cos(math.pi * (epoch - warmup_epochs) / (max_epochs - warmup_epochs)))
        elif self.mode == "plateau":
            if metric is None:
                return self._plateau_lr
            factor = self.kwargs.get("factor", 0.5)
            patience = self.kwargs.get("patience", 10)
            min_delta = self.kwargs.get("min_delta", 0.0)
            metric_mode = self.kwargs.get("metric_mode", "min")

            if self._plateau_best is None:
                is_improvement = True
            elif metric_mode == "min":
                is_improvement = metric < self._plateau_best - min_delta
            else:
                is_improvement = metric > self._plateau_best + min_delta

            if is_improvement:
                self._plateau_best = metric
                self._plateau_bad_epochs = 0
            else:
                self._plateau_bad_epochs += 1
                if self._plateau_bad_epochs >= patience:
                    self._plateau_lr *= factor
                    self._plateau_bad_epochs = 0
            return self._plateau_lr
        elif self.mode == "polynomial":
            max_epochs = self.kwargs.get("max_epochs", 100)
            power = self.kwargs.get("power", 1.0)
            end_lr = self.kwargs.get("end_lr", 0.0)
            # Clamped past max_epochs for the same reason as "cosine": an
            # unclamped (1 - e/E)**p goes negative for odd-ish powers.
            frac = 1.0 - min(epoch, max_epochs) / max_epochs
            return end_lr + (self.initial_lr - end_lr) * (frac ** power)
        elif self.mode == "cyclic":
            # Triangular CLR (Smith 2017): sweep base_lr <-> max_lr every
            # 2 * step_size epochs. "triangular2" halves the amplitude each
            # full cycle; "exp_range" scales it by gamma**epoch.
            base_lr = self.kwargs.get("base_lr", self.initial_lr / 10.0)
            max_lr = self.kwargs.get("max_lr", self.initial_lr)
            step_size = self.kwargs.get("step_size", 10)
            policy = self.kwargs.get("policy", "triangular")
            cycle = int(math.floor(1 + epoch / (2 * step_size)))
            x = abs(epoch / step_size - 2 * cycle + 1)          # 0 at peak, 1 at floor
            amplitude = max_lr - base_lr
            if policy == "triangular2":
                amplitude /= 2.0 ** (cycle - 1)
            elif policy == "exp_range":
                amplitude *= self.kwargs.get("gamma", 0.999) ** epoch
            return base_lr + amplitude * max(0.0, 1.0 - x)
        elif self.mode == "one_cycle":
            # Smith's 1cycle: cosine-anneal UP from initial_lr/div_factor to
            # max_lr over the first pct_start of the run, then all the way
            # DOWN to max_lr/final_div_factor. One cycle, no repeats.
            max_lr = self.kwargs.get("max_lr", self.initial_lr)
            max_epochs = self.kwargs.get("max_epochs", 100)
            pct_start = self.kwargs.get("pct_start", 0.3)
            start_lr = max_lr / self.kwargs.get("div_factor", 25.0)
            final_lr = max_lr / self.kwargs.get("final_div_factor", 1e4)
            warm = max(1.0, pct_start * max_epochs)
            e = min(epoch, max_epochs)
            if e < warm:
                lo, hi, frac = start_lr, max_lr, e / warm
            else:
                lo, hi = max_lr, final_lr
                frac = (e - warm) / max(1.0, max_epochs - warm)
            return lo + (hi - lo) * 0.5 * (1 - math.cos(math.pi * min(frac, 1.0)))
        elif self.mode == "cosine_warm_restarts":
            # SGDR: cosine anneal over T_0 epochs, restart at full LR, and
            # lengthen each successive cycle by T_mult.
            T_0 = self.kwargs.get("T_0", 10)
            T_mult = self.kwargs.get("T_mult", 1)
            eta_min = self.kwargs.get("eta_min", 0.0)
            t, period = epoch, T_0
            while t >= period:
                t -= period
                period = max(1, int(period * T_mult))
            return eta_min + (self.initial_lr - eta_min) * 0.5 * (
                1 + math.cos(math.pi * t / period))
        elif self.mode == "swa":
            # SWA's schedule: run normally until swa_start, linearly anneal
            # down to swa_lr over anneal_epochs, then HOLD it there. The flat
            # high tail is what keeps the model moving around the basin so
            # the averaged snapshots are meaningfully different points.
            swa_start = self.kwargs.get("swa_start", 0)
            swa_lr = self.kwargs.get("swa_lr", 0.05)
            anneal = max(1, self.kwargs.get("anneal_epochs", 5))
            if epoch < swa_start:
                return self.initial_lr
            progress = min(1.0, (epoch - swa_start) / anneal)
            return self.initial_lr + (swa_lr - self.initial_lr) * progress
        elif self.mode == "lambda":
            # Arbitrary user schedule: fn(epoch) -> multiplier on initial_lr.
            fn = self.kwargs.get("lr_lambda")
            if fn is None:
                raise ValueError("mode='lambda' requires lr_lambda=<callable>")
            return self.initial_lr * float(fn(epoch))
        elif self.mode == "sequential":
            # Chain schedulers back to back. `milestones` are the epochs at
            # which each next scheduler takes over; each one is stepped with
            # an epoch counted from ITS OWN start, so a chained cosine
            # anneals over its own span rather than resuming mid-curve.
            schedulers = self.kwargs.get("schedulers")
            milestones = list(self.kwargs.get("milestones", []))
            if not schedulers:
                raise ValueError("mode='sequential' requires schedulers=[...]")
            if len(milestones) != len(schedulers) - 1:
                raise ValueError(
                    f"sequential needs len(milestones) == len(schedulers) - 1; "
                    f"got {len(milestones)} and {len(schedulers)}")
            idx, start = 0, 0
            for i, m in enumerate(milestones):
                if epoch >= m:
                    idx, start = i + 1, m
            return schedulers[idx].step(epoch - start, metric=metric)
        else:
            return self.initial_lr
