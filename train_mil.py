import argparse
import yaml
import torch
import numpy as np
import pandas as pd
import random
from tqdm import tqdm
from loguru import logger
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

from scripts.models.MIL.abmil2 import ABMIL
from scripts.models.MIL.mil_dataset import MILDataset

# Configure loguru for clean output
logger.remove()
logger.add(lambda msg: print(msg.strip()), format="{time:YYYY-MM-DD HH:mm:ss} {level} {message}", level="INFO")


def set_seed(config: dict):
    seed = config.get("random_seed", 42)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def load_config(config_path):
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="configs/train_mil.yml",
        help="Path to the configuration YAML file.",
    )
    args = parser.parse_args()
    return args


def get_model(config: dict) -> torch.nn.Module:
    if config["model_name"] == "ABMIL":
        model = ABMIL(
            n_classes=config["num_of_classes"],
            in_dim=config["feature_dim"],
            hidden_dim=config["hidden_dim"],
        )
    else:
        raise ValueError(f"Model '{config['model_name']}' not supported.")
    model = model.to(config["device"])
    return model

def get_data_loaders(config: dict) -> tuple[DataLoader, DataLoader]:
    """Create and return train and validation DataLoaders"""
    full_dataset = pd.read_csv(config["metadata_path"])
    train_df, val_df = train_test_split(
        full_dataset,
        test_size=config["validation_size"],
        random_state=config["random_seed"],
        stratify=full_dataset['label'] if 'label' in full_dataset.columns else None
    )
    logger.info(f"Train dataset size: {len(train_df)} bags")
    logger.info(f"Validation dataset size: {len(val_df)} bags")
    
    train_dataset = MILDataset(
        feature_dir=config["feature_dir"],
        metadata_df=train_df,
    )
    val_dataset = MILDataset(
        feature_dir=config["feature_dir"],
        metadata_df=val_df,
    )
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config["num_workers"],
    )
    val_loader = DataLoader(
        dataset = val_dataset,
        batch_size=config["batch_size"],
        shuffle=False, 
        num_workers=config["num_workers"],
    )
    return train_loader, val_loader


def validate(model: torch.nn.Module, data_loader: DataLoader, config: dict):
    model.eval()
    all_predictions = []
    all_labels = []
    with torch.no_grad():
        for patch_features, slide_label in tqdm(data_loader, desc="Validating...", disable=config["disable_progress_bar"]):
            patch_features = patch_features.to(config["device"])
            slide_label = slide_label.to(config["device"])            
            logits = model(patch_features)
            predictions = torch.argmax(logits, dim=1).cpu().numpy()
            all_predictions.append(predictions)
            all_labels.append(slide_label.cpu().numpy())
    
    # Concatenate all predictions and labels before passing to classification_report
    all_predictions_np = np.concatenate(all_predictions, axis=0)
    all_labels_np = np.concatenate(all_labels, axis=0)

    report = classification_report(all_labels_np, all_predictions_np, zero_division=0)
    return report


def train(config: dict):
    set_seed(config)
    model = get_model(config)
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    criterion = torch.nn.CrossEntropyLoss()
    train_dataloader, val_dataloader = get_data_loaders(config)

    history = []
    batch_idx = 1
    for epoch in range(config["num_epochs"]):
        logger.info(f"Starting Epoch {epoch+1}/{config['num_epochs']}")
        for patch_features, slide_label in tqdm(train_dataloader, 
                                                desc = f"Training Epoch {epoch+1}", 
                                                disable=config["disable_progress_bar"]):
            patch_features = patch_features.to(config["device"])
            slide_label = slide_label.to(config["device"])

            logits = model(patch_features)
            loss = criterion(logits, slide_label)
            history.append(loss.item())
            if torch.isnan(patch_features).any() or torch.isinf(patch_features).any():
                logger.error("NaN or Inf found in patch_features!")
                raise ValueError("NaN or Inf found in patch_features!")
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            logger.debug(f"Batch shape: {patch_features.shape}" )


            logger.info(f"Epoch {epoch+1}, Batch {batch_idx}: Loss = {loss.item():.4f}") # Log loss
            
            if batch_idx % config["validation_rate"] == 0:
                val_report = validate(model, val_dataloader, config)
                logger.info(f"Validation report: \n{val_report}")
                model.train()
            
            batch_idx += 1

    logger.info("Training completed")
    train_report = validate(model, train_dataloader, config)
    logger.info(f"Metrics on training set: \n{train_report}")
    plt.plot(history)
    plt.savefig("loss.png")

if __name__ == "__main__":
    args = parse_args()
    config = load_config(args.config)
    train(config)