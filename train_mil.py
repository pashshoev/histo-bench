import comet_ml
from comet_ml import Experiment

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
from sklearn.metrics import classification_report, roc_auc_score
from torch.nn.functional import softmax
import matplotlib.pyplot as plt
import mlflow

from scripts.models.MIL.abmil import ABMIL
from scripts.models.MIL.transmil import TransMIL
from scripts.models.MIL.mil_dataset import MILDataset, mil_collate_fn, filter_data
from scripts.models.MIL.mean_pooling import MeanPooling
from scripts.models.MIL.clam import CLAM_MB
from scripts.models.MIL.wikg import WiKG

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
            dropout=config["dropout"],
        )
    elif config["model_name"] == "TransMIL":
        model = TransMIL(
            n_classes=config["num_of_classes"],
            in_dim=config["feature_dim"],
            hidden_dim=config["hidden_dim"],
            dropout=config["dropout"],
        )
    elif config["model_name"] == "MeanPooling":
        model = MeanPooling(
            n_classes=config["num_of_classes"],
            in_dim=config["feature_dim"],
        )
    elif config["model_name"] == "CLAM_MB":
        model = CLAM_MB(
            n_classes=config["num_of_classes"],
            in_dim=config["feature_dim"],
            hidden_dim=config["hidden_dim"],
        )
    elif config["model_name"] == "WiKG":
        model = WiKG(
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
    from torch.utils.data import WeightedRandomSampler
    
    full_dataset = pd.read_csv(config["metadata_path"])
    full_dataset = filter_data(full_dataset, config["feature_dir"])
    logger.info(f"Full dataset size after filtering: {len(full_dataset)} bags")
    train_df, val_df = train_test_split(
        full_dataset,
        test_size=config["validation_size"],
        random_state=config["random_seed"],
        stratify=full_dataset['label'] if 'label' in full_dataset.columns else None
    )
    
    train_dataset = MILDataset(
        feature_dir=config["feature_dir"],
        metadata_df=train_df,
    )
    val_dataset = MILDataset(
        feature_dir=config["feature_dir"],
        metadata_df=val_df,
    )
    
    # Create weighted sampler for training data if requested
    if config.get("use_weighted_sampler", False):
        # Get class distribution from training data
        train_labels = train_df['label'].values
        class_counts = np.bincount(train_labels)
        
        # Calculate weights: inverse of class frequency
        class_weights = 1.0 / class_counts
        sample_weights = class_weights[train_labels]
        
        # Create weighted sampler
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(train_dataset),
            replacement=True
        )
        
        logger.info(f"Using WeightedRandomSampler with class weights: {class_weights}")
        logger.info(f"Original class distribution: {class_counts}")
        
        train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=config["batch_size"],
            sampler=sampler,  # Use sampler instead of shuffle
            num_workers=config["num_workers"],
            collate_fn=mil_collate_fn,  # Use custom collate function
        )
    else:
        train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=config["batch_size"],
            shuffle=True,
            num_workers=config["num_workers"],
            collate_fn=mil_collate_fn,  # Use custom collate function
        )
    
    val_loader = DataLoader(
        dataset = val_dataset,
        batch_size=config["batch_size"],
        shuffle=False, 
        num_workers=config["num_workers"],
        collate_fn=mil_collate_fn,  # Use custom collate function
    )
    logger.info(f"Train dataset size: {len(train_dataset)} bags")
    logger.info(f"Validation dataset size: {len(val_dataset)} bags")
    return train_loader, val_loader


def plot_history(history: dict):
    plt.figure(figsize=(10, 5))
    plt.plot(history["epoch"], history["losses"]["train"], label="Training", color="blue")
    plt.plot(history["epoch"], history["losses"]["val"], label="Validation", color="red")
    plt.xlabel("Epoch")
    plt.ylabel("Average Loss")
    plt.title("Training and Validation Losses")
    plt.legend()
    plt.savefig("loss.png")
    plt.close()


def flatten_report(report: dict, prefix="r") -> dict:
    flat = {}
    for key, value in report.items():
        if isinstance(value, dict):
            for sub_key, sub_val in value.items():
                flat[f"{prefix}/{key}/{sub_key}"] = float(sub_val)
        else:
            flat[f"{prefix}/{key}"] = float(value)
    return flat


def log_mlflow(config: dict, model: torch.nn.Module, metrics: dict, checkpoint_path: str):
    host = "127.0.0.1"
    port = 5000

    mlflow.set_tracking_uri(uri=f"http://{host}:{port}")
    mlflow.set_experiment(config["experiment_name"])
    mlflow.log_params(config)
    # report_flat = flatten_report(metrics["report"])

    mlflow.log_metrics(metrics)

    mlflow.log_artifact(checkpoint_path, artifact_path="model_state_dict")

    mlflow.pytorch.log_model(
        model,
        name="model",
    )


def validate(model: torch.nn.Module, 
             data_loader: DataLoader, 
             criterion: torch.nn.Module, 
             config: dict):
    model.eval()
    all_predictions = []
    all_labels = []
    all_probs = []
    all_losses = []

    with torch.no_grad():
        for patch_features, slide_labels, attention_masks in tqdm(data_loader, desc="Validating...", disable=config["disable_progress_bar"]):
            patch_features = patch_features.to(config["device"])
            slide_labels = slide_labels.to(config["device"])
            attention_masks = attention_masks.to(config["device"])
            
            logits = model(patch_features)
            loss = criterion(logits, slide_labels)

            probs = softmax(logits, dim=1).cpu().numpy()
            predictions = np.argmax(probs, axis=1)

            all_predictions.append(predictions)
            all_labels.append(slide_labels.cpu().numpy())
            all_probs.append(probs)
            all_losses.append(loss.item())

    average_loss = np.mean(all_losses)
    all_predictions_np = np.concatenate(all_predictions, axis=0)
    all_labels_np = np.concatenate(all_labels, axis=0)
    all_probs_np = np.concatenate(all_probs, axis=0)

    report = classification_report(all_labels_np, all_predictions_np, zero_division=0, output_dict=True)

    # AUC (macro/micro) — requires one-hot encoded labels
    num_classes = all_probs_np.shape[1]
    labels_onehot = np.eye(num_classes)[all_labels_np]

    auc_macro = roc_auc_score(labels_onehot, all_probs_np, average='macro', multi_class='ovr')
    auc_micro = roc_auc_score(labels_onehot, all_probs_np, average='micro', multi_class='ovr')

    result = {
        "report": report,
        "loss": average_loss,
        "auc_macro": auc_macro,
        "auc_micro": auc_micro
    }
    return result

def train(config: dict):
    set_seed(config)
    exp = Experiment(project_name=config["experiment_name"],
                     api_key=config["comet_api_key"],
                     workspace="bakhtierzhon-pashshoev")
    exp.log_parameters(config)

    model = get_model(config)
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    
    # Handle class weights if provided in config (only if not using weighted sampler)
    if "class_weights" in config and not config.get("use_weighted_sampler", False):
        class_weights = torch.tensor(config["class_weights"], dtype=torch.float32).to(config["device"])
        criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
        logger.info(f"Using class weights: {config['class_weights']}")
    else:
        criterion = torch.nn.CrossEntropyLoss()
        if config.get("use_weighted_sampler", False):
            logger.info("Using WeightedRandomSampler - no class weights needed in loss function")
        else:
            logger.info("No class weights specified, using default CrossEntropyLoss")
    
    train_dataloader, val_dataloader = get_data_loaders(config)

    history = {
        "epoch": [],
        "losses":{
            "train": [],
            "val": []
        }
    }

    best_val_loss = float('inf')
    checkpoint_path = "checkpoint.pt"
    for epoch in range(config["num_epochs"]):
        model.train()
        logger.info(f"Starting Epoch {epoch+1}/{config['num_epochs']}")
        epoch_loss = 0
        for patch_features, slide_labels, attention_masks in tqdm(train_dataloader, 
                                                desc = f"Training Epoch {epoch+1}", 
                                                disable=False):
            patch_features = patch_features.to(config["device"])
            slide_labels = slide_labels.to(config["device"])
            attention_masks = attention_masks.to(config["device"])
            
            # Forward pass
            logits = model(patch_features)
            loss = criterion(logits, slide_labels)
            epoch_loss += loss.item()
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        # Logs
        history["epoch"].append(epoch+1)
        avg_train_loss = epoch_loss / len(train_dataloader)
        val_result = validate(model, val_dataloader, criterion, config)
        logger.info(f"Validation report: \n{val_result['report']}")
        logger.info(f"Losses: train = {avg_train_loss:.10f}, val = {val_result['loss']:.4f}")
        
        exp.log_metrics({"train_loss": avg_train_loss, "val_loss": val_result["loss"]}, epoch=epoch)
        exp.log_metrics({"auc_macro": val_result["auc_macro"], "auc_micro": val_result["auc_micro"]}, epoch=epoch)
        exp.log_metrics(val_result["report"], epoch=epoch)
        
        history["losses"]["train"].append(avg_train_loss)
        history["losses"]["val"].append(val_result["loss"])

        if val_result["loss"] < best_val_loss:
            best_val_loss = val_result["loss"]
            torch.save(model.state_dict(), checkpoint_path)

    exp.log_metric("best_val_loss", best_val_loss)
    exp.log_model(
        name="best_model",
        file_or_folder="checkpoint.pt",
        overwrite=True,
        metadata={
            "type": "torch_state_dict",
            "description": "Best model checkpoint based on val loss"
        }
    )    
    exp.end()
    train_result = validate(model, train_dataloader, criterion, config)
    logger.info(f"Metrics on training set: \n{train_result['report']}")
    logger.info("Training finished successfully!")
    plot_history(history)


if __name__ == "__main__":
    # comet_ml.login()
    args = parse_args()
    config = load_config(args.config)

    train(config)