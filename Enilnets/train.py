import numpy as np

def TrainBatch(self, xs, ys, loss_function=None, **loss_kwargs):
    out = self.Forward(xs, training=True)
    if loss_function is None:
        loss_function = "cross_entropy" if self.layers[-1].get("activation") == "softmax" else "mse"
    loss = self.ComputeLoss(out, ys, loss_function, **loss_kwargs)
    self.Backward(ys)
    self.update()
    return loss, out

def compute_accuracy(self, predictions, targets):
    if predictions.shape[-1] > 1:  # Multi-class
        pred_classes = np.argmax(predictions, axis=1)
        true_classes = np.argmax(targets, axis=1)
    else:  # Binary
        pred_classes = (predictions > 0.5).astype(int).flatten()
        true_classes = targets.flatten()
    return np.mean(pred_classes == true_classes)

def Train(self, X_train, Y_train, epochs=10, batch_size=32, X_val=None, Y_val=None, loss_function=None, verbose=True, **loss_kwargs):
    history = {"loss": [], "val_loss": [], "accuracy": [], "val_accuracy": []}
    n_samples = X_train.shape[0]
    for epoch in range(epochs):
        indices = np.random.permutation(n_samples)
        X_shuffled = X_train[indices]
        Y_shuffled = Y_train[indices]
        epoch_loss = 0.0
        epoch_acc = 0.0
        total_samples = 0
        for i in range(0, n_samples, batch_size):
            X_batch = X_shuffled[i:i+batch_size]
            Y_batch = Y_shuffled[i:i+batch_size]
            loss, preds = self.TrainBatch(X_batch, Y_batch, loss_function=loss_function, **loss_kwargs)
            batch_size_actual = X_batch.shape[0]
            epoch_loss += loss * batch_size_actual
            epoch_acc += self.compute_accuracy(preds, Y_batch) * batch_size_actual
            total_samples += batch_size_actual
        avg_loss = epoch_loss / total_samples
        avg_acc = epoch_acc / total_samples
        history["loss"].append(avg_loss)
        history["accuracy"].append(avg_acc)
        if X_val is not None and Y_val is not None:
            val_pred = self.Forward(X_val)
            val_loss = self.ComputeLoss(val_pred, Y_val, loss_function if loss_function is not None else ("cross_entropy" if self.layers[-1].get("activation") == "softmax" else "mse"), **loss_kwargs)
            val_acc = self.compute_accuracy(val_pred, Y_val)
            history["val_loss"].append(val_loss)
            history["val_accuracy"].append(val_acc)
            if verbose:
                print(f"Epoch {epoch+1}/{epochs} - loss: {avg_loss:.4f} - acc: {avg_acc:.4f} - val_loss: {val_loss:.4f} - val_acc: {val_acc:.4f}")
        else:
            if verbose:
                print(f"Epoch {epoch+1}/{epochs} - loss: {avg_loss:.4f} - acc: {avg_acc:.4f}")
    return history
