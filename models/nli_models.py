import pytorch_lightning as pl
import torch
import torch.nn as nn
from transformers import (
    AdamW,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_cosine_schedule_with_warmup,
)

# import pytorch_lightning.metrics.functional as metrics
import torchmetrics.functional as metrics


class NLIClassifier(pl.LightningModule):
    def __init__(self, model, num_steps, num_classes=2):
        super(NLIClassifier, self).__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.encoder = AutoModelForSequenceClassification.from_pretrained(model)
        self.num_steps = num_steps

    def step(self, batch):
        premises, hypotheses, labels = batch
        tokenized = self.tokenizer(
            premises, hypotheses, return_tensors="pt", padding=True
        )["input_ids"]
        encoded = self.encoder(tokenized, labels=labels, return_dict=True,)
        return encoded.logits, encoded.loss

    def forward(self, batch):
        logits, _ = self.step(batch)
        return logits

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=2e-5)
        return [optimizer]

    def training_step(self, batch, batch_idx):
        _, loss = self.step(batch)
        return {"loss": loss}

    def training_epoch_end(self, outputs):
        training_loss = sum([x["loss"] for x in outputs])
        self.log("train_loss", training_loss)
        # return {"train_loss": training_loss, "log": {"train_loss": training_loss}}

    def validation_step(self, batch, batch_idx):
        prem, hyp, labels = batch
        logits, loss = self.step(batch)
        return {"val_loss": loss, "pred": logits.argmax(1), "true": labels}

    def validation_epoch_end(self, outputs):
        val_loss = sum([x["val_loss"] for x in outputs])
        pred = torch.cat([x["pred"] for x in outputs])
        true = torch.cat([x["true"] for x in outputs])
        f_score = metrics.f1(pred, true, num_classes=2)
        accuracy = metrics.accuracy(pred, true)
        out = {"val_loss": val_loss, "val_f_score": f_score, "val_accuracy": accuracy}
        self.log_dict({"val_loss": val_loss, "val_f_score": f_score, "val_accuracy": accuracy})
        return {**out, "log": out}

    def test_step(self, batch, batch_idx):
        prem, hyp, labels = batch
        logits, _ = self.step(batch)
        loss = nn.functional.cross_entropy(logits, labels, reduction="sum")
        return {"test_loss": loss, "pred": logits.argmax(1), "true": labels}

    def test_epoch_end(self, outputs):
        test_loss = sum([x["test_loss"] for x in outputs])
        pred = torch.cat([x["pred"] for x in outputs])
        true = torch.cat([x["true"] for x in outputs])
        f_score = metrics.f1(pred, true, num_classes=2)
        accuracy = metrics.accuracy(pred, true)
        out = {
            "test_loss": test_loss,
            "test_f_score": f_score,
            "test_accuracy": accuracy,
        }
        self.log_dict({"test_loss": val_loss, "test_f_score": f_score, "test_accuracy": accuracy})
        return {**out, "log": out}
