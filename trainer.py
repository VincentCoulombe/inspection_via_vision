import time, copy
import datetime
import os
from typing import Any, Dict, List
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np

from logger import CsvLogger

class SegmentationTrainer():
    def __init__(self, model: nn.Module, lr: float, epochs: int, batch_size: int, dataloaders: Dict[str, DataLoader]) -> None:
        
        self.model = model
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.epochs = epochs
        self.batch_size = batch_size
        self.dataloaders = dataloaders # {"Train": train_loader, "Val": val_loader}
        self.criterion = torch.nn.MSELoss(reduction="mean")
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)
        
    def _print_memory_usage(self) -> None:
        if self.device == torch.device("cuda:0"):
            memoire_disponible = round(torch.cuda.max_memory_reserved()*1e-9,3)
            memoire_utilisee = round(torch.cuda.max_memory_allocated()*1e-9,3)
            print(f"{memoire_utilisee * 100 / memoire_disponible}% de la mémoire GPU est utilisée.")
        
    def fit(self, metrics: Dict[str, Any], save_dir: str = "", save_name: str = "default_name.pt", log_dir: str = "",
            log_name: str = "log.csv") -> nn.Module:
        # sourcery skip: low-code-quality
        
        start = time.perf_counter()
        best_model_weights = copy.deepcopy(self.model.state_dict())
        best_loss = 0
        metric_names = ["Epoch", "Train_loss", "Val_loss"] + [f"Train_{name}" for name in metrics.keys] +\
                        [f"Val_{name}" for name in metrics.keys]
        if log_dir != "":
            logger = CsvLogger(log_dir, log_name, metric_names)

        for epoch in range(self.epochs):
            
            print(f"\nEpoch {epoch}/{self.epochs - 1}")
            print("-" * 10)
            epoch_metrics = {name: [] for name in metric_names}
            # Each epoch has a training and validation phase
            for phase in ["Train", "Val"]:
                
                # Setuper le model en fonction de la phase
                self.model.train() if phase == "Train" else self.model.eval()

                # Iterate over data.
                for sample in tqdm(iter(self.dataloaders[phase])):
                    inputs = sample["image"].to(self.device)
                    masks = sample["mask"].to(self.device)

                    # reset les gradients
                    self.optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == "Train"):
                        outputs = self.model(inputs)
                        loss = self.criterion(outputs, masks)
                        epoch_metrics[f"{phase}_loss"].append(loss.item())
                        readable_preds = outputs.detach().cpu().numpy().ravel()
                        readable_labels = masks.detach().cpu().numpy().ravel()
                        for metric_name, metric_func in metrics.items():
                            epoch_metrics[f"{phase}_{metric_name}"].append(metric_func(readable_preds, readable_labels))                        
                        # backward + optimize only if in training phase
                        if phase == "Train":
                            loss.backward()
                            self.optimizer.step()

                epoch_metrics_avgs = {np.mean(metric) for metric in epoch_metrics}
                avg_loss = epoch_metrics_avgs[f"{phase}_loss"]
                print(f"{phase} Loss: {avg_loss:.4f}")
                self._print_memory_usage()
                if log_dir != "":
                    logger.log({"Epoch": epoch, **epoch_metrics_avgs})

                # Update les weights du best model si la validation loss est la meilleure depuis le début
                if phase == "Val" and loss < best_loss:
                    best_loss = loss
                    best_model_weights = copy.deepcopy(self.model.state_dict())
                    # Sauvegarder les poids du meilleur modèle
                    if save_dir != "":
                        os.makedirs(save_dir, exist_ok=True)
                        torch.save(best_model_weights, os.path.join(save_dir,save_name))

        duree = datetime.timedelta(seconds=(time.perf_counter() - start))
        print(f"Durée du Training {duree.hour}h {duree.min}m {duree.second}s")
        print(f"Meilleure Loss: {best_loss:.4f}")

        # retourner le model avec les meilleurs weights
        return self.model.load_state_dict(best_model_weights)    
    
