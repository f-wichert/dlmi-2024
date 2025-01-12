import lightning as L

from dlmi.train.model import UNet

class UNETModel(pl.LightningModule):

    def __init__(self,):
        super().__init__()
        self.model = UNet()