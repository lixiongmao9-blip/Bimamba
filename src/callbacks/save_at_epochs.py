from pathlib import Path
from typing import Iterable, Sequence, Union

from pytorch_lightning import Callback, LightningModule, Trainer


class SaveAtEpochsCallback(Callback):
    """Save a full Lightning checkpoint at the end of selected epochs (after validation).

    Epoch indices follow ``trainer.current_epoch``, matching ``metrics_{epoch}.json`` in this project.
    """

    def __init__(
        self,
        epochs: Sequence[int],
        dirpath: str,
        filename_tpl: str = "epoch_{epoch:03d}.ckpt",
    ) -> None:
        self.epochs = {int(e) for e in epochs}
        self.dirpath = dirpath
        self.filename_tpl = filename_tpl

    def on_validation_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if trainer.sanity_checking:
            return
        if not trainer.is_global_zero:
            return
        if trainer.current_epoch not in self.epochs:
            return

        out = Path(self.dirpath)
        out.mkdir(parents=True, exist_ok=True)
        name = self.filename_tpl.format(epoch=trainer.current_epoch)
        trainer.save_checkpoint(str(out / name))
