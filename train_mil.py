import comet_ml
from comet_ml import Experiment
import os
from dotenv import load_dotenv
import argparse
import torch
import numpy as np
import pandas as pd
import random
import copy
from tqdm import tqdm
from loguru import logger
from torch.utils.data import DataLoader
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import classification_report, roc_auc_score
from torch.nn.functional import softmax
import matplotlib.pyplot as plt
import mlflow

from scripts.models.MIL.abmil2 import ABMIL
from scripts.models.MIL.transmil import TransMIL
from scripts.models.MIL.mil_dataset import MILDataset, filter_data
from scripts.models.MIL.mean_pooling import MeanPooling
from scripts.models.MIL.clam import CLAM_MB
from scripts.models.MIL.wikg import WiKG


class EarlyStopping:
    """Early stopping utility to stop training when validation loss stops improving.
    
    Args:
        patience (int): Number of epochs to wait after last improvement before stopping.
        min_delta (float): Minimum change in the monitored quantity to qualify as improvement.
        restore_best_weights (bool): Whether to restore model weights from the epoch with the best value.
        mode (str): One of 'min' or 'max'. In 'min' mode, training will stop when the quantity monitored has stopped decreasing.
    """
    
    def __init__(self, patience=7, min_delta=0, restore_best_weights=True, mode='min'):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.mode = mode
        
        self.best_score = None
        self.counter = 0
        self.best_weights = None
        self.early_stop = False
        
        if mode == 'min':
            self.monitor_op = np.less
            self.min_delta *= -1
        elif mode == 'max':
            self.monitor_op = np.greater
        else:
            raise ValueError(f"Mode {mode} is unknown, use 'min' or 'max'")
    
    def __call__(self, val_score, model):
        """Check if training should be stopped.
        
        Args:
            val_score (float): Current validation score
            model (torch.nn.Module): Model to potentially restore weights for
            
        Returns:
            bool: True if training should be stopped, False otherwise
        """
        if self.best_score is None:
            self.best_score = val_score
            self.save_checkpoint(model)
        elif self.monitor_op(val_score, self.best_score + self.min_delta):
            self.best_score = val_score
            self.counter = 0
            self.save_checkpoint(model)
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                if self.restore_best_weights:
                    model.load_state_dict(self.best_weights)
        
        return self.early_stop
    
    def save_checkpoint(self, model):
        """Save model checkpoint."""
        if self.restore_best_weights:
            # Deep copy to prevent in-place modifications during training affecting saved weights
            self.best_weights = copy.deepcopy(model.state_dict())

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
    
    # Cross-validation parameters
    parser.add_argument(
        "--n_folds",
        type=int,
        default=1,
        help="Number of folds for cross-validation (default: 1)"
    )
    parser.add_argument(
        "--single_fold_test_size",
        type=float,
        default=0.0,
        help="When n_folds==1, fraction of data (by groups) to reserve as test. 0.0 disables test split."
    )
    
    # Early stopping parameters
    parser.add_argument(
        "--early_stopping_patience",
        type=int,
        default=None,
        help="Number of epochs to wait before early stopping (default: None, no early stopping)"
    )
    parser.add_argument(
        "--early_stopping_min_delta",
        type=float,
        default=0.0,
        help="Minimum change in EMA validation AUC to qualify as improvement (default: 0.0)"
    )
    parser.add_argument(
        "--early_stopping_restore_best_weights",
        action="store_true",
        default=True,
        help="Restore model weights from the epoch with the best EMA validation AUC (default: True)"
    )
    parser.add_argument(
        "--ema_alpha",
        type=float,
        default=0.9,
        help="EMA smoothing factor for validation AUC (0-1, higher=more smoothing)"
    )
    
    # Patch sampling parameters
    parser.add_argument(
        "--patch_sampling_ratio",
        type=float,
        default=1.0,
        help="Fraction of patches to randomly sample from each slide (default: 1.0, use all patches)"
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


def get_data_loaders(config: dict, fold: int = 0):
    """Create and return train, validation, and test DataLoaders for a specific fold.

    For cross-validation (n_folds > 1):
      - Hold out one fold as test
      - Split the remaining folds into train/val with GroupShuffleSplit

    For single split (n_folds == 1):
      - Return train/val only; test loader is None
    """
    from torch.utils.data import WeightedRandomSampler
    from sklearn.model_selection import GroupKFold
    
    full_dataset = pd.read_csv(config["metadata_path"])
    full_dataset = filter_data(full_dataset, config["feature_dir"])
    logger.info(f"Full dataset size after filtering: {len(full_dataset)} bags")
    
    # Log patient-wise distribution
    counts = full_dataset.groupby('case_id').size().value_counts().sort_index()
    logger.info(f"Slides per patient distribution:\n{counts}")
    
    if config["n_folds"] > 1:
        # Use GroupKFold for cross-validation (no shuffle in GroupKFold)
        gkf = GroupKFold(n_splits=config["n_folds"]) 
        splits = list(gkf.split(full_dataset, groups=full_dataset['case_id']))
        # Select one fold as test
        remaining_idx, test_idx = splits[fold]
        remaining_df = full_dataset.iloc[remaining_idx].copy()
        # Split remaining into train/val by groups
        from sklearn.model_selection import GroupShuffleSplit
        gss = GroupShuffleSplit(n_splits=1, test_size=config["validation_size"], random_state=config["random_seed"])
        rem_train_local_idx, rem_val_local_idx = next(gss.split(remaining_df, groups=remaining_df['case_id']))
        train_idx = remaining_df.iloc[rem_train_local_idx].index.values
        val_idx = remaining_df.iloc[rem_val_local_idx].index.values
        logger.info(f"Cross-validation fold {fold + 1}/{config['n_folds']} (with test holdout)")
    else:
        # Single-fold mode. Optionally create a held-out test split first, then split remaining into train/val
        from sklearn.model_selection import GroupShuffleSplit
        test_idx = None
        if config.get("single_fold_test_size", 0.0) and config["single_fold_test_size"] > 0.0:
            gss_test = GroupShuffleSplit(n_splits=1, test_size=config["single_fold_test_size"], random_state=config["random_seed"])
            remaining_idx, test_idx = next(gss_test.split(full_dataset, groups=full_dataset['case_id']))
            remaining_df = full_dataset.iloc[remaining_idx].copy()
            logger.info(f"Single-fold: using test_size={config['single_fold_test_size']}")
            # Now split remaining into train/val
            gss_val = GroupShuffleSplit(n_splits=1, test_size=config["validation_size"], random_state=config["random_seed"])
            rem_train_local_idx, rem_val_local_idx = next(gss_val.split(remaining_df, groups=remaining_df['case_id']))
            train_idx = remaining_df.iloc[rem_train_local_idx].index.values
            val_idx = remaining_df.iloc[rem_val_local_idx].index.values
            logger.info("Single train/validation/test split")
        else:
            # Only train/val
            gss = GroupShuffleSplit(n_splits=1, test_size=config["validation_size"], random_state=config["random_seed"])
            train_idx, val_idx = next(gss.split(full_dataset, groups=full_dataset['case_id']))
            logger.info("Single train/validation split (no test)")
    
    train_df = full_dataset.iloc[train_idx].copy()
    val_df = full_dataset.iloc[val_idx].copy()
    test_df = None if test_idx is None else full_dataset.iloc[test_idx].copy()
    
    # Ensure patient sets are disjoint
    train_patients = set(train_df['case_id'])
    val_patients = set(val_df['case_id'])
    assert train_patients.isdisjoint(val_patients), "Train and validation patient sets must be disjoint"
    if test_df is not None:
        test_patients = set(test_df['case_id'])
        assert train_patients.isdisjoint(test_patients), "Train and test patient sets must be disjoint"
        assert val_patients.isdisjoint(test_patients), "Validation and test patient sets must be disjoint"
    
    logger.info(f"Fold {fold + 1}: train={len(train_patients)} patients, validation={len(val_patients)} patients" + (f", test={len(test_patients)} patients" if test_df is not None else ""))
    logger.info(f"Fold {fold + 1}: train={len(train_df)} slides, validation={len(val_df)} slides" + (f", test={len(test_df)} slides" if test_df is not None else ""))
    
    train_dataset = MILDataset(
        feature_dir=config["feature_dir"],
        metadata_df=train_df,
        patch_sampling_ratio=config["patch_sampling_ratio"],
    )
    val_dataset = MILDataset(
        feature_dir=config["feature_dir"],
        metadata_df=val_df,
        patch_sampling_ratio=config["patch_sampling_ratio"],
    )
    test_dataset = None if test_df is None else MILDataset(
        feature_dir=config["feature_dir"],
        metadata_df=test_df,
        patch_sampling_ratio=config["patch_sampling_ratio"],
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
            pin_memory = False,
        )
    else:
        train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=config["batch_size"],
            shuffle=True,
            num_workers=config["num_workers"],
            pin_memory = False,
        )
    
    val_loader = DataLoader(
        dataset = val_dataset,
        batch_size=config["batch_size"],
        shuffle=False, 
        num_workers=config["num_workers"],
        pin_memory = False,
    )
    test_loader = None if test_dataset is None else DataLoader(
        dataset = test_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config["num_workers"],
        pin_memory = False,
    )
    logger.info(f"Train dataset size: {len(train_dataset)} bags")
    logger.info(f"Validation dataset size: {len(val_dataset)} bags")
    if test_dataset is not None:
        logger.info(f"Test dataset size: {len(test_dataset)} bags")
    logger.info(f"Patch sampling ratio: {config['patch_sampling_ratio']:.3f} (using {config['patch_sampling_ratio']*100:.1f}% of patches per slide)")
    return train_loader, val_loader, test_loader


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
        for patch_features, slide_labels in tqdm(data_loader, desc="Validating...", disable=config["disable_progress_bar"]):
            patch_features = patch_features.to(config["device"], non_blocking=True)
            slide_labels = slide_labels.to(config["device"], non_blocking=True)
            
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


def cross_validate(config: dict):
    """Perform cross-validation by training the model on multiple folds"""
    logger.info(f"Starting {config['n_folds']}-fold cross-validation")
    
    fold_results = []
    
    for fold in range(config["n_folds"]):
        logger.info(f"\n{'='*50}")
        logger.info(f"Training Fold {fold + 1}/{config['n_folds']}")
        logger.info(f"{'='*50}")
        
        # Create fold-specific config
        fold_config = config.copy()
        fold_config['current_fold'] = fold
        
        # Train on this fold
        fold_result = train_single_fold(fold_config)
        fold_results.append(fold_result)
        
        logger.info(f"Fold {fold + 1} completed. Best EMA val AUC: {fold_result['best_val_auc_ema']:.6f}")
    
    # Aggregate results across folds
    aggregate_cv_results(fold_results, config)
    
    return fold_results


def aggregate_cv_results(fold_results: list, config: dict):
    """Aggregate and log cross-validation results to Comet ML"""
    logger.info(f"\n{'='*50}")
    logger.info("CROSS-VALIDATION RESULTS SUMMARY")
    logger.info(f"{'='*50}")
    
    # Extract key metrics
    val_ema_aucs = [r['best_val_auc_ema'] for r in fold_results]
    test_aucs_macro = [r['final_test_auc_macro'] for r in fold_results if r['final_test_auc_macro'] is not None]
    test_accuracies = [r['final_test_accuracy'] for r in fold_results if r['final_test_accuracy'] is not None]
    test_weighted_f1_scores = [r['final_test_weighted_f1'] for r in fold_results if r['final_test_weighted_f1'] is not None]
    
    # Calculate statistics
    mean_val_auc_ema = np.mean(val_ema_aucs)
    std_val_auc_ema = np.std(val_ema_aucs)
    mean_test_auc_macro = float(np.mean(test_aucs_macro)) if len(test_aucs_macro) > 0 else float('nan')
    std_test_auc_macro = float(np.std(test_aucs_macro)) if len(test_aucs_macro) > 0 else float('nan')
    mean_test_accuracy = float(np.mean(test_accuracies)) if len(test_accuracies) > 0 else float('nan')
    std_test_accuracy = float(np.std(test_accuracies)) if len(test_accuracies) > 0 else float('nan')
    mean_test_weighted_f1 = float(np.mean(test_weighted_f1_scores)) if len(test_weighted_f1_scores) > 0 else float('nan')
    std_test_weighted_f1 = float(np.std(test_weighted_f1_scores)) if len(test_weighted_f1_scores) > 0 else float('nan')
    
    logger.info(f"Mean EMA Validation AUC: {mean_val_auc_ema:.6f} ± {std_val_auc_ema:.6f}")
    logger.info(f"Mean TEST AUC (Macro): {mean_test_auc_macro:.6f} ± {std_test_auc_macro:.6f}")
    logger.info(f"Mean TEST Accuracy: {mean_test_accuracy:.6f} ± {std_test_accuracy:.6f}")
    logger.info(f"Mean TEST Weighted F1: {mean_test_weighted_f1:.6f} ± {std_test_weighted_f1:.6f}")
    
    # Create summary run in the same experiment
    summary_exp = Experiment(
        project_name=config["experiment_name"],
        api_key=config["comet_api_key"],
        workspace="bakhtierzhon-pashshoev"
    )
    # Log aggregated metrics
    summary_exp.log_metrics({
        "mean_val_auc_ema": mean_val_auc_ema,
        "std_val_auc_ema": std_val_auc_ema,
        "mean_test_auc_macro": mean_test_auc_macro,
        "std_test_auc_macro": std_test_auc_macro,
        "mean_test_accuracy": mean_test_accuracy,
        "std_test_accuracy": std_test_accuracy,
        "mean_test_weighted_f1": mean_test_weighted_f1,
        "std_test_weighted_f1": std_test_weighted_f1,
        "n_folds": config['n_folds']
    })
    
    # Log parameters
    summary_exp.log_parameters(config)
    
    summary_exp.end()
    logger.info("Cross-validation summary logged to Comet ML")


def train_single_fold(config: dict):
    """Train the model on a single fold (renamed from original train function)"""
    set_seed(config)
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("metrics", exist_ok=True)
    
    exp = Experiment(
        project_name=config["experiment_name"],
        api_key=config["comet_api_key"],
        workspace="bakhtierzhon-pashshoev"
    )
    exp.log_parameters(config)
    
    # Log fold information if this is part of cross-validation
    if config.get('current_fold') is not None:
        exp.log_parameter("current_fold", config['current_fold'])
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
    
    train_dataloader, val_dataloader, test_dataloader = get_data_loaders(config, fold=config.get('current_fold', 0))

    history = {
        "epoch": [],
        "losses":{
            "train": [],
            "val": []
        }
    }

    best_val_auc_ema = float('-inf')
    checkpoint_path = "checkpoints/best_model.pt"
    
    # Initialize early stopping
    early_stopping = None
    if config.get("early_stopping_patience") is not None:
        early_stopping = EarlyStopping(
            patience=config["early_stopping_patience"],
            min_delta=config.get("early_stopping_min_delta", 0.0),
            restore_best_weights=config.get("early_stopping_restore_best_weights", True),
            mode='max'
        )
        logger.info(f"Early stopping enabled (monitor=EMA val AUC, mode=max) with patience={config['early_stopping_patience']}, min_delta={config.get('early_stopping_min_delta', 0.0)}, ema_alpha={config.get('ema_alpha', 0.9)}")
    
    val_auc_ema = None
    for epoch in range(config["num_epochs"]):
        model.train()
        current_lr = optimizer.param_groups[0]['lr']
        logger.info(f"Starting Epoch {epoch+1}/{config['num_epochs']} with lr={current_lr:.6f}")
        
        epoch_loss = 0
        for patch_features, slide_labels in tqdm(train_dataloader, 
                                                desc = f"Training Epoch {epoch+1}", 
                                                disable=False):
            patch_features = patch_features.to(config["device"], non_blocking=True)
            slide_labels = slide_labels.to(config["device"], non_blocking=True)
            
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
        # Compute EMA for validation AUC (macro)
        val_auc = float(val_result["auc_macro"])
        if val_auc_ema is None:
            val_auc_ema = val_auc
        else:
            alpha = config.get("ema_alpha", 0.9)
            val_auc_ema = alpha * val_auc_ema + (1 - alpha) * val_auc
        logger.info(f"Validation report: \n{val_result['report']}")
        logger.info(f"Train loss = {avg_train_loss:.10f}, Val loss = {val_result['loss']:.4f}, Val AUC (macro) = {val_auc:.6f}, Val AUC EMA = {val_auc_ema:.6f}")
        
        # Log metrics including current learning rate
        exp.log_metrics({"train_loss": avg_train_loss, "val_loss": val_result["loss"]}, epoch=epoch)
        exp.log_metrics({"val_auc": val_result["auc_macro"], "val_auc_ema": val_auc_ema, "val_auc_micro": val_result["auc_micro"]}, epoch=epoch)
        exp.log_metrics(val_result["report"], epoch=epoch)
        exp.log_metrics({"learning_rate": current_lr}, epoch=epoch)
        
        history["losses"]["train"].append(avg_train_loss)
        history["losses"]["val"].append(val_result["loss"])

        # Checkpoint saving based on EMA val AUC and early stopping on EMA val AUC
        if val_auc_ema > best_val_auc_ema:
            best_val_auc_ema = val_auc_ema
            torch.save(model.state_dict(), checkpoint_path)
            logger.info(f"New best EMA val AUC: {best_val_auc_ema:.6f}")

        if early_stopping is not None:
            if early_stopping(val_auc_ema, model):
                logger.info(f"Early stopping triggered after {early_stopping.counter} epochs without improvement")
                logger.info(f"Best EMA validation AUC: {early_stopping.best_score:.6f}")
                # Ensure the best model checkpoint is saved (weights already restored if configured)
                torch.save(model.state_dict(), checkpoint_path)
                break

    # Ensure final model corresponds to the best EMA val AUC
    try:
        model.load_state_dict(torch.load(checkpoint_path, map_location=config["device"]))
    except Exception:
        pass

    exp.log_metric("best_val_auc_ema", best_val_auc_ema)
    if early_stopping is not None:
        exp.log_metric("epochs_without_improvement", early_stopping.counter)
        exp.log_metric("early_stopping_triggered", early_stopping.early_stop)
        exp.log_metric("early_stopping_best_score", early_stopping.best_score)
    exp.log_model(
        name="best_model",
        file_or_folder=checkpoint_path,
        overwrite=True,
        metadata={
            "type": "torch_state_dict",
            "description": "Best model checkpoint based on val loss"
        }
    )    
    exp.end()
    
    # Evaluate on test set if available
    test_result = None
    if test_dataloader is not None:
        test_result = validate(model, test_dataloader, criterion, config)
        exp.log_metrics({
            "test_loss": test_result["loss"],
            "test_auc_macro": test_result["auc_macro"],
            "test_auc_micro": test_result["auc_micro"],
        })
        exp.log_metrics(test_result["report"])

    logger.info("Training finished successfully!")
    
    # Return results for cross-validation aggregation
    # Use the last validation result from training (batch_size=1, so this is already final)
    return {
        'best_val_auc_ema': best_val_auc_ema,
        'final_val_auc_macro': val_result['auc_macro'],
        'final_val_accuracy': val_result['report']['accuracy'],
        'final_val_weighted_f1': val_result['report']['weighted avg']['f1-score'],
        'final_val_report': val_result['report'],
        'final_test_auc_macro': (test_result['auc_macro'] if test_result is not None else None),
        'final_test_accuracy': (test_result['report']['accuracy'] if test_result is not None else None),
        'final_test_weighted_f1': (test_result['report']['weighted avg']['f1-score'] if test_result is not None else None),
        'final_test_report': (test_result['report'] if test_result is not None else None),
    }


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
        "n_folds": args.n_folds,
        "single_fold_test_size": args.single_fold_test_size,
        "early_stopping_patience": args.early_stopping_patience,
        "early_stopping_min_delta": args.early_stopping_min_delta,
        "early_stopping_restore_best_weights": args.early_stopping_restore_best_weights,
        "patch_sampling_ratio": args.patch_sampling_ratio,
        "ema_alpha": args.ema_alpha,
    }

    if config["n_folds"] > 1:
        cross_validate(config)
    else:
        train_single_fold(config)