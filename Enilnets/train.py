import numpy as np
from .utils import iterate_minibatches

def TrainBatch(self, xs, ys, loss_function=None, accumulation_steps=1, **loss_kwargs):
    """accumulation_steps > 1: accumulates gradients over that many calls
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

def compute_accuracy(self, predictions, targets):
    if predictions.shape[-1] > 1:  # Multi-class
        pred_classes = np.argmax(predictions, axis=1)
        true_classes = np.argmax(targets, axis=1)
    else:  # Binary
        pred_classes = (predictions > 0.5).astype(int).flatten()
        true_classes = targets.flatten()
    return np.mean(pred_classes == true_classes)

def compute_precision_recall_f1(self, predictions, targets):
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

    precision = tp / (tp + fp + 1e-12)
    recall = tp / (tp + fn + 1e-12)
    f1 = 2 * precision * recall / (precision + recall + 1e-12)
    return {"precision": precision, "recall": recall, "f1": f1}

def Train(self, X_train, Y_train, epochs=10, batch_size=32, X_val=None, Y_val=None,
          loss_function=None, verbose=True, scheduler=None, early_stopping=None,
          accumulation_steps=1, callbacks=None, **loss_kwargs):
    """early_stopping: optional Enilnets.EarlyStopping instance monitoring
    val_loss (if X_val/Y_val given) or train loss otherwise; training stops
    as soon as early_stopping.step(...) returns True.

    callbacks: optional list of duck-typed callback objects. No shared base
    class -- implement whichever of these your callback needs:
      on_epoch_end(epoch, logs, model=self) -- called once per epoch, right
        after `history` is updated and before the early-stopping check.
        `logs` is a dict with this epoch's "loss"/"accuracy"/"lr" and, if
        validation data was given, "val_loss"/"val_accuracy".
      on_train_end(history) -- called once after the epoch loop ends
        (whether by exhausting `epochs` or by early stopping).
    Missing methods are simply skipped (no error)."""
    history = {"loss": [], "val_loss": [], "accuracy": [], "val_accuracy": [], "lr": []}
    n_samples = X_train.shape[0]
    prev_metric = None
    for epoch in range(epochs):
        if scheduler is not None:
            lr = scheduler.step(epoch, metric=prev_metric)
            self.set_lr(lr)

        epoch_loss = 0.0
        epoch_acc = 0.0
        total_samples = 0
        for X_batch, Y_batch in iterate_minibatches(X_train, Y_train, batch_size, shuffle=True):
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
    """Learning rate schedulers.

    mode="plateau": real ReduceLROnPlateau. kwargs: factor (default 0.5),
    patience (default 10), min_delta (default 0.0), metric_mode ("min",
    default, or "max" -- NOT `mode`, which is taken by the schedule-name
    param). Call step(epoch, metric=...) every epoch with the just-finished
    epoch's monitored value (Train() does this automatically, with a
    one-epoch lag so the LR used *during* an epoch reflects metrics known
    *before* it started).
    """
    def __init__(self, initial_lr, mode="step", **kwargs):
        self.initial_lr = initial_lr
        self.mode = mode
        self.kwargs = kwargs
        self._plateau_lr = initial_lr
        self._plateau_best = None
        self._plateau_bad_epochs = 0

    def step(self, epoch, metric=None):
        if self.mode == "step":
            drop = self.kwargs.get("drop", 0.5)
            epochs_drop = self.kwargs.get("epochs_drop", 10)
            return self.initial_lr * (drop ** (epoch // epochs_drop))
        elif self.mode == "exponential":
            decay = self.kwargs.get("decay", 0.95)
            return self.initial_lr * (decay ** epoch)
        elif self.mode == "cosine":
            max_epochs = self.kwargs.get("max_epochs", 100)
            return self.initial_lr * 0.5 * (1 + np.cos(np.pi * epoch / max_epochs))
        elif self.mode == "warmup_cosine":
            max_epochs = self.kwargs.get("max_epochs", 100)
            warmup_epochs = self.kwargs.get("warmup_epochs", 5)
            if epoch < warmup_epochs:
                return self.initial_lr * (epoch / warmup_epochs)
            else:
                return self.initial_lr * 0.5 * (1 + np.cos(np.pi * (epoch - warmup_epochs) / (max_epochs - warmup_epochs)))
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
        else:
            return self.initial_lr
