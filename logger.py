import csv
import os
from typing import List


class CsvLogger:
    def __init__(self, log_dir: str, log_name: str, columns_names: List[str]) -> None:
        
        self.log_dir = log_dir
        self.log_name = log_name if log_name.endswith(".csv") else f"{log_name}.csv"
        self.columns_names = columns_names
        os.makedirs(log_dir, exist_ok=True)
        with open(os.path.join(log_dir, log_name), "w", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=self.columns_names)
            writer.writeheader()


    def log(self, row: dict) -> None:
        
        if any(name not in self.columns_names for name in row):
            raise ValueError(f"{row} ne map pas avec les noms des colonnes : {self.columns_names}.")
        
        with open(os.path.join(self.log_dir, self.log_name), "w", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=self.columns_names)
            writer.writerow(row)
