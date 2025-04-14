# train_collisionnet.py
import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from torchmetrics import Accuracy, Precision, Recall, F1Score

from models.collisionnet import CollisionNet
from data_loader import get_data_loaders_from_tensor

class CollisionNetPL(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = CollisionNet()
        self.loss_fn = nn.BCELoss()
        
        # Initialize metrics
        self.train_acc = Accuracy(task='binary')
        self.val_acc = Accuracy(task='binary')
        self.test_acc = Accuracy(task='binary')
        
        self.val_precision = Precision(task='binary')
        self.val_recall = Recall(task='binary')
        self.val_f1 = F1Score(task='binary')

    def forward(self, xyz, q):
        return self.model(xyz, q)

    def training_step(self, batch, batch_idx):
        xyz = batch['pointcloud']
        q = batch['q']
        labels = batch['collision_flag']
        
        preds = self(xyz, q)
        loss = self.loss_fn(preds.squeeze(), labels)
        
        self.train_acc(preds.squeeze(), labels.int())
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_acc', self.train_acc, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        xyz = batch['pointcloud']
        q = batch['q']
        labels = batch['collision_flag']
        
        preds = self(xyz, q)
        loss = self.loss_fn(preds.squeeze(), labels)
        
        self.val_acc(preds.squeeze(), labels.int())
        self.val_precision(preds.squeeze(), labels.int())
        self.val_recall(preds.squeeze(), labels.int())
        self.val_f1(preds.squeeze(), labels.int())
        
        self.log_dict({
            'val_loss': loss,
            'val_acc': self.val_acc,
            'val_precision': self.val_precision,
            'val_recall': self.val_recall,
            'val_f1': self.val_f1
        }, prog_bar=True)

    def test_step(self, batch, batch_idx):
        xyz = batch['pointcloud']
        q = batch['q']
        labels = batch['collision_flag']
        
        preds = self(xyz, q)
        loss = self.loss_fn(preds.squeeze(), labels)
        
        self.test_acc(preds.squeeze(), labels.int())
        self.log('test_loss', loss)
        self.log('test_acc', self.test_acc)

    def configure_optimizers(self):
        return self.model.configure_optimizers()

def train():
    # Data configuration
    TENSOR_FILE = "./collision_bool/processed.pt"
    BATCH_SIZE = 64
    NUM_WORKERS = 8
    
    # Model configuration
    model = CollisionNetPL()
    
    # Data loaders
    train_loader, val_loader = get_data_loaders_from_tensor(
        TENSOR_FILE,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS
    )
    
    # Training configuration
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath='checkpoints/',
        filename='collisionnet-{epoch:02d}-{val_loss:.2f}',
        save_top_k=3,
        mode='min'
    )
    
    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        patience=10,
        mode='min'
    )
    
    trainer = pl.Trainer(
        max_epochs=60,
        callbacks=[checkpoint_callback, early_stop_callback],
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1,
        log_every_n_steps=10
    )
    
    # Start training
    trainer.fit(model, train_loader, val_loader)

def test():
    # Load best model
    model = CollisionNetPL.load_from_checkpoint(
        checkpoint_path='checkpoints/your-best-checkpoint.ckpt'
    )
    
    # Load test data
    test_loader = get_data_loaders_from_tensor(
        "./collision_bool/processed_test.pt",
        batch_size=64,
        train_ratio=0.01  # No training split, purely test data
    )[1]  # Only get the test loader
    
    # Run testing
    trainer = pl.Trainer(
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1
    )
    
    results = trainer.test(model, dataloaders=test_loader)
    print(f"Test results: {results}")

if __name__ == '__main__':
    train()
    # After training completes, run:
    # test()