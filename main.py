import click
import os
import torch.nn

from trainer import *
from datahandler import *
from model import * 

@click.command()
@click.option("--weights_saving_name", type=str, default="compactCNN_segmentation.pt", help="Name of the model")
@click.option("--save_dir", type=str, default=r"/content/gdrive/My Drive/Colab Notebooks/Saved models/inspection visuelle", help="Directory to save the model")
@click.option("--data_dir", type=str, default=r"/content/gdrive/My Drive/DAGM_all/DAGM_segmentation", help="Directory of the data")
@click.option("--epochs", type=int, default=10, help="Number of epochs")
@click.option("--batch_size", type=int, default=32, help="Batch size")
@click.option("--lr", type=float, default=1e-5, help="Learning rate")
def main(lr: float, batch_size: int, epochs: int, save_dir: str, data_dir: str, weights_saving_name: str):
    
	model = LargeMobileNet()
	print(data_dir)
	dataloaders = segmentation_dataset(data_dir, batch_size)
	trainer = SegmentationTrainer(model, lr, epochs, batch_size, dataloaders)
	trainer.fit(save_dir, weights_saving_name, r"./logs", "log.csv")
	model.predict(".", "1_0002.PNG")
	model.predict(".", "1_0003.PNG")
    
if __name__ == '__main__':
	main()