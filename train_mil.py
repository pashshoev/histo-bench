import comet_ml
from comet_ml import Experiment
import os
from dotenv import load_dotenv
import argparse
import torch
import numpy as np
import pandas as pd
import random
from tqdm import tqdm
from loguru import logger
from torch.utils.data import DataLoader
from sklearn.model_selection import GroupShuffleSplit
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

# Load environment variables from .env file
load_dotenv()

# Configure loguru for clean output
logger.remove()
logger.add(lambda msg: print(msg.strip()), format="{time:YYYY-MM-DD HH:mm:ss} {level} {message}", level="INFO")


def set_seed(config: dict):
    seed = config.get("random_seed", 42)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def parse_args():
    parser = argparse.ArgumentParser()
    
    # Required parameters (must be specified)
    parser.add_argument(
        "--metadata_path",
        type=str,
        required=True,
        help="Path to CSV/Excel file with slide_id, label, and case_id columns"
    )
    parser.add_argument(
        "--feature_dir",
        type=str,
        required=True,
        help="Path to directory containing SLIDE_ID.h5 feature files"
    )
    parser.add_argument(
        "--num_of_classes",
        type=int,
        required=True,
        help="Number of output classes"
    )
    parser.add_argument(
        "--feature_dim",
        type=int,
        required=True,
        help="Dimension of the input instance embeddings"
    )
    parser.add_argument(
        "--experiment_name",
        type=str,
        required=True,
        help="Experiment name for logging"
    )
    
    # Core training parameters (with sensible defaults)
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=2,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.0005,
        help="Learning rate for training"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size for DataLoader"
    )
    parser.add_argument(
        "--validation_size",
        type=float,
        default=0.2,
        help="Proportion of the dataset to use for validation"
    )
    
    # Model architecture parameters
    parser.add_argument(
        "--model_name",
        type=str,
        default="MeanPooling",
        help="Model name (ABMIL, TransMIL, MeanPooling, CLAM_MB, WiKG)"
    )
    parser.add_argument(
        "--hidden_dim",
        type=int,
        default=256,
        help="Dimension for the intermediate attention layers"
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.3,
        help="Dropout rate for the final classifier"
    )
    
    # Optimizer and scheduler parameters
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.01,
        help="Weight decay for AdamW optimizer"
    )
    parser.add_argument(
        "--scheduler_T_max",
        type=int,
        default=None,
        help="T_max parameter for CosineAnnealingLR scheduler (defaults to num_epochs)"
    )
    
    # Training configuration
    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="Number of subprocesses for data loading"
    )
    parser.add_argument(
        "--validation_rate",
        type=int,
        default=100,
        help="Perform validation every N training batches"
    )
    parser.add_argument(
        "--use_weighted_sampler",
        action="store_true",
        help="Use WeightedRandomSampler instead of class weights"
    )
    
    # System and reproducibility
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to use (cuda, cpu, mps)"
    )
    parser.add_argument(
        "--random_seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    
    # Logging and monitoring
    parser.add_argument(
        "--disable_progress_bar",
        action="store_true",
        help="Disable progress bars during training/validation"
    )
    
    return parser.parse_args()


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


def setup_optimizer(model: torch.nn.Module, config: dict) -> torch.optim.Optimizer:
    """Initialize and return the AdamW optimizer with configurable weight decay"""
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"]
    )
    logger.info(f"Initialized AdamW optimizer with lr={config['learning_rate']}, weight_decay={config['weight_decay']}")
    return optimizer


def setup_scheduler(optimizer: torch.optim.Optimizer, config: dict) -> torch.optim.lr_scheduler._LRScheduler:
    """Initialize and return the CosineAnnealingLR scheduler"""
    # Set T_max to num_epochs if not specified
    T_max = config.get("scheduler_T_max")
    if T_max is None:
        T_max = config["num_epochs"]
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=T_max
    )
    logger.info(f"Initialized CosineAnnealingLR scheduler with T_max={T_max}")
    return scheduler


def get_data_loaders(config: dict) -> tuple[DataLoader, DataLoader]:
    """Create and return train and validation DataLoaders"""
    from torch.utils.data import WeightedRandomSampler
    
    full_dataset = pd.read_csv(config["metadata_path"])
    full_dataset = filter_data(full_dataset, config["feature_dir"])
    logger.info(f"Full dataset size after filtering: {len(full_dataset)} bags")
    
    # Log patient-wise distribution
    counts = full_dataset.groupby('case_id').size().value_counts().sort_index()
    logger.info(f"Slides per patient distribution:\n{counts}")
    
    # Create patient-wise train/validation split
    gss = GroupShuffleSplit(n_splits=1, test_size=config["validation_size"], random_state=config["random_seed"])
    train_idx, val_idx = next(gss.split(full_dataset, groups=full_dataset['case_id']))
    
    train_df = full_dataset.iloc[train_idx].copy()
    val_df = full_dataset.iloc[val_idx].copy()
    
    # Ensure patient sets are disjoint
    train_patients = set(train_df['case_id'])
    val_patients = set(val_df['case_id'])
    assert train_patients.isdisjoint(val_patients), "Train and validation patient sets must be disjoint"
    
    logger.info(f"Patient-wise splits: train={len(train_patients)} patients, validation={len(val_patients)} patients")
    logger.info(f"Slide-wise splits: train={len(train_df)} slides, validation={len(val_df)} slides")
    
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
    plt.savefig("metrics/loss.png")
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

    # Initialize optimizer and scheduler
    optimizer = setup_optimizer(model, config)
    scheduler = setup_scheduler(optimizer, config)
    
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
    checkpoint_path = "checkpoints/checkpoint.pt"
    
    for epoch in range(config["num_epochs"]):
        model.train()
        current_lr = optimizer.param_groups[0]['lr']
        logger.info(f"Starting Epoch {epoch+1}/{config['num_epochs']} with lr={current_lr:.6f}")
        
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
        
        # Step the scheduler to update learning rate
        scheduler.step()
        
        # Logs
        history["epoch"].append(epoch+1)
        avg_train_loss = epoch_loss / len(train_dataloader)
        val_result = validate(model, val_dataloader, criterion, config)
        logger.info(f"Validation report: \n{val_result['report']}")
        logger.info(f"Losses: train = {avg_train_loss:.10f}, val = {val_result['loss']:.4f}")
        
        # Log metrics including current learning rate
        exp.log_metrics({"train_loss": avg_train_loss, "val_loss": val_result["loss"]}, epoch=epoch)
        exp.log_metrics({"auc_macro": val_result["auc_macro"], "auc_micro": val_result["auc_micro"]}, epoch=epoch)
        exp.log_metrics(val_result["report"], epoch=epoch)
        exp.log_metrics({"learning_rate": current_lr}, epoch=epoch)
        
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
    comet_api_key = os.getenv("COMET_API_KEY")
    
    # Create config dictionary directly from parsed arguments
    config = {
        "num_epochs": args.num_epochs,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "scheduler_T_max": args.scheduler_T_max,
        "hidden_dim": args.hidden_dim,
        "dropout": args.dropout,
        "batch_size": args.batch_size,
        "validation_size": args.validation_size,
        "random_seed": args.random_seed,
        "device": args.device,
        "disable_progress_bar": args.disable_progress_bar,
        "model_name": args.model_name,
        "num_of_classes": args.num_of_classes,
        "feature_dim": args.feature_dim,
        "metadata_path": args.metadata_path,
        "feature_dir": args.feature_dir,
        "num_workers": args.num_workers,
        "validation_rate": args.validation_rate,
        "use_weighted_sampler": args.use_weighted_sampler,
        "experiment_name": args.experiment_name,
        "comet_api_key": comet_api_key,
    }

    train(config)