"""Classification diagnostics used to investigate the VI.4 §15 failure modes.

These metrics describe predictions on a split. They do not reveal which image
detail was lost during resizing/patchification or prove that attention caused
an output; trace those questions through the representation stages themselves.
"""

import torch


class MetricAccumulator:
    """Accumulate classification metrics for the VI.4 companion workflow (§17).

    The confusion matrix follows actual-class rows and predicted-class
    columns. Per-class recall helps reveal uneven errors, as suggested by the
    diagnostic discipline in §15.
    """

    def __init__(self, classes: list[str], device: torch.device | str = "cpu") -> None:
        """Initialize counts for each ground-truth/predicted class pair."""
        self.classes = classes
        self.num_classes = len(classes)
        self.confusion = torch.zeros(
            (self.num_classes, self.num_classes), dtype=torch.int64, device=device
        )
        self.loss_sum = 0.0
        self.samples = 0

    def update(self, logits: torch.Tensor, labels: torch.Tensor, loss_sum: float) -> None:
        """Add a batch of model readouts and targets to the running diagnostics."""
        predictions = logits.argmax(dim=1)
        encoded = labels * self.num_classes + predictions
        self.confusion += torch.bincount(
            encoded, minlength=self.num_classes * self.num_classes
        ).reshape(self.num_classes, self.num_classes)
        self.loss_sum += float(loss_sum)
        self.samples += labels.numel()

    def summary(self) -> dict:
        """Return aggregate and per-class diagnostics for the complete split."""
        if not self.samples:
            raise ValueError("Cannot calculate metrics for an empty dataset")
        confusion = self.confusion.to(torch.float64)
        true_positive = confusion.diag()
        predicted_count = confusion.sum(dim=0)
        actual_count = confusion.sum(dim=1)
        precision = true_positive / predicted_count.clamp_min(1)
        recall = true_positive / actual_count.clamp_min(1)
        f1 = 2 * precision * recall / (precision + recall).clamp_min(1e-12)
        correct = int(true_positive.sum().item())
        per_class = {
            name: {
                "support": int(actual_count[index].item()),
                "correct": int(true_positive[index].item()),
                "precision": float(precision[index].item()),
                "recall": float(recall[index].item()),
                "f1": float(f1[index].item()),
            }
            for index, name in enumerate(self.classes)
        }
        return {
            "loss": self.loss_sum / self.samples,
            "accuracy": correct / self.samples,
            "correct": correct,
            "incorrect": self.samples - correct,
            "macro_precision": float(precision.mean().item()),
            "macro_recall": float(recall.mean().item()),
            "macro_f1": float(f1.mean().item()),
            "per_class": per_class,
            "confusion_matrix": {
                "rows": "actual classes",
                "columns": "predicted classes",
                "class_order": self.classes,
                "values": self.confusion.cpu().tolist(),
            },
            "num_samples": self.samples,
        }
