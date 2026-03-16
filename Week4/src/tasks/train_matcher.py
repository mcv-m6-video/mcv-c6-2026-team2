import argparse
import os

import torch
import torch.nn as nn
import wandb
from torch.utils.data import DataLoader

from src.utils.reid_training_dataset import (
    RandomIdentitySampler,
    ReIDFolderDataset,
    ReIDSubset,
    build_query_gallery_indices,
    split_indices_by_pid,
)
from src.utils.reid_training_utils import (
    DEFAULT_CONFIG,
    BatchHardTripletLoss,
    ReIDNet,
    build_parser,
    build_transforms,
    evaluate,
    load_config,
    log_dataset_summary,
    set_seed,
    setup_logging,
    train_one_epoch,
)


def main():
    """Runs the full train, validation, and test pipeline."""
    logger = setup_logging()
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", type=str, default=None)
    config_args, remaining_argv = config_parser.parse_known_args()

    config_values = load_config(config_args.config)
    parser_defaults = DEFAULT_CONFIG.copy()
    parser_defaults.update(config_values)

    parser = build_parser(parser_defaults)
    args = parser.parse_args(remaining_argv)
    args.config = config_args.config

    if args.config:
        logger.info("Loaded config file: %s", args.config)
    else:
        logger.info("Running without config file; using defaults and CLI arguments.")
    logger.info("Resolved arguments: %s", vars(args))

    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    logger.info("Seed set to %d", args.seed)
    logger.info("Output directory: %s", args.output_dir)

    wandb.init(
        project="vehicle-reid",
        name=f"{args.backbone}_emb{args.embedding_dim}_lr{args.lr}",
        config=vars(args),
    )
    logger.info("Weights & Biases run initialized.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    train_tfms, test_tfms = build_transforms(args.img_size)
    train_sequences = [seq.strip() for seq in args.train_sequences.split(",") if seq.strip()]
    val_sequences = [seq.strip() for seq in args.val_sequences.split(",") if seq.strip()]
    test_sequences = [seq.strip() for seq in args.test_sequences.split(",") if seq.strip()]
    logger.info(
        "Sequence selection | train=%s | val=%s | test=%s",
        train_sequences,
        val_sequences if val_sequences else "<split from train>",
        test_sequences,
    )

    base_train_set = ReIDFolderDataset(args.data_root, sequences=train_sequences, transform=None)
    log_dataset_summary(logger, "Base train pool", base_train_set)
    val_query_set = None
    val_gallery_set = None

    if val_sequences:
        logger.info("Using dedicated validation sequences.")
        train_set = ReIDSubset(
            dataset=base_train_set,
            indices=list(range(len(base_train_set.samples))),
            transform=train_tfms,
        )

        base_val_set = ReIDFolderDataset(args.data_root, sequences=val_sequences, transform=None)
        log_dataset_summary(logger, "Validation pool", base_val_set)
        val_query_indices, val_gallery_indices = build_query_gallery_indices(base_val_set.samples)
        val_query_set = ReIDSubset(
            dataset=base_val_set,
            indices=val_query_indices,
            transform=test_tfms,
        )
        val_gallery_set = ReIDSubset(
            dataset=base_val_set,
            indices=val_gallery_indices,
            transform=test_tfms,
        )
    else:
        logger.info("Splitting validation identities from training pool with val_ratio=%.3f", args.val_ratio)
        train_indices, val_indices = split_indices_by_pid(
            base_train_set.samples,
            val_ratio=args.val_ratio,
            seed=args.seed,
        )
        logger.info(
            "Identity split complete | train_samples=%d | val_samples=%d",
            len(train_indices),
            len(val_indices),
        )

        train_samples = [base_train_set.samples[idx] for idx in train_indices]
        train_pid2label = {pid: idx for idx, pid in enumerate(sorted({sample.pid for sample in train_samples}))}
        train_set = ReIDSubset(
            dataset=base_train_set,
            indices=train_indices,
            transform=train_tfms,
            pid2label=train_pid2label,
        )

        if val_indices:
            val_query_indices, val_gallery_indices = build_query_gallery_indices(
                [base_train_set.samples[idx] for idx in val_indices]
            )
            val_query_set = ReIDSubset(
                dataset=base_train_set,
                indices=[val_indices[idx] for idx in val_query_indices],
                transform=test_tfms,
            )
            val_gallery_set = ReIDSubset(
                dataset=base_train_set,
                indices=[val_indices[idx] for idx in val_gallery_indices],
                transform=test_tfms,
            )

    base_test_set = ReIDFolderDataset(args.data_root, sequences=test_sequences, transform=None)
    log_dataset_summary(logger, "Test pool", base_test_set)
    test_query_indices, test_gallery_indices = build_query_gallery_indices(base_test_set.samples)
    test_query_set = ReIDSubset(
        dataset=base_test_set,
        indices=test_query_indices,
        transform=test_tfms,
    )
    test_gallery_set = ReIDSubset(
        dataset=base_test_set,
        indices=test_gallery_indices,
        transform=test_tfms,
    )
    log_dataset_summary(logger, "Train subset", train_set)
    if val_query_set is not None and val_gallery_set is not None:
        log_dataset_summary(logger, "Validation query", val_query_set)
        log_dataset_summary(logger, "Validation gallery", val_gallery_set)
    log_dataset_summary(logger, "Test query", test_query_set)
    log_dataset_summary(logger, "Test gallery", test_gallery_set)

    train_sampler = RandomIdentitySampler(
        dataset=train_set,
        batch_size=args.batch_size,
        num_instances=args.num_instances,
    )

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_query_loader = None
    val_gallery_loader = None
    if val_query_set is not None and val_gallery_set is not None:
        val_query_loader = DataLoader(
            val_query_set,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
        )
        val_gallery_loader = DataLoader(
            val_gallery_set,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
        )
    test_query_loader = DataLoader(
        test_query_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    test_gallery_loader = DataLoader(
        test_gallery_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    logger.info(
        "Data loaders ready | train_batches=%d | val_query_batches=%s | val_gallery_batches=%s | test_query_batches=%d | test_gallery_batches=%d",
        len(train_loader),
        len(val_query_loader) if val_query_loader is not None else "n/a",
        len(val_gallery_loader) if val_gallery_loader is not None else "n/a",
        len(test_query_loader),
        len(test_gallery_loader),
    )

    num_classes = len(train_set.pids)
    logger.info("Training classes: %d", num_classes)

    model = ReIDNet(
        num_classes=num_classes,
        backbone_name=args.backbone,
        embedding_dim=args.embedding_dim,
        pretrained=True,
    ).to(device)
    logger.info(
        "Model initialized | backbone=%s | embedding_dim=%d",
        args.backbone,
        args.embedding_dim,
    )

    ce_loss_fn = nn.CrossEntropyLoss()
    tri_loss_fn = BatchHardTripletLoss(margin=args.margin)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    logger.info(
        "Optimizer and scheduler ready | lr=%.6f | weight_decay=%.6f | margin=%.3f",
        args.lr,
        args.weight_decay,
        args.margin,
    )

    best_rank1 = -1.0
    best_path = os.path.join(args.output_dir, "best_model.pth")

    for epoch in range(1, args.epochs + 1):
        logger.info("Starting epoch %03d/%03d", epoch, args.epochs)
        train_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            ce_loss_fn=ce_loss_fn,
            tri_loss_fn=tri_loss_fn,
            device=device,
            ce_weight=args.ce_weight,
            tri_weight=args.tri_weight,
            logger=logger,
            epoch=epoch,
            total_epochs=args.epochs,
            log_interval=args.log_interval,
        )

        scheduler.step()
        logger.info(
            "Finished epoch %03d/%03d | avg_loss=%.4f | avg_ce=%.4f | avg_tri=%.4f | lr=%.6f",
            epoch,
            args.epochs,
            train_metrics["loss"],
            train_metrics["ce_loss"],
            train_metrics["tri_loss"],
            scheduler.get_last_lr()[0],
        )

        eval_metrics = None
        if val_query_loader is not None and val_gallery_loader is not None:
            logger.info("Running validation for epoch %03d", epoch)
            eval_metrics = evaluate(model, val_query_loader, val_gallery_loader, device)

        if eval_metrics is not None:
            logger.info(
                "Validation results | epoch=%03d | mAP=%.4f | R1=%.4f | R5=%.4f | R10=%.4f",
                epoch,
                eval_metrics["mAP"],
                eval_metrics["rank1"],
                eval_metrics["rank5"],
                eval_metrics["rank10"],
            )

        wandb_payload = {
            "epoch": epoch,
            "train/loss": train_metrics["loss"],
            "train/ce_loss": train_metrics["ce_loss"],
            "train/tri_loss": train_metrics["tri_loss"],
            "lr": scheduler.get_last_lr()[0],
        }
        if eval_metrics is not None:
            wandb_payload.update(
                {
                    "val/mAP": eval_metrics["mAP"],
                    "val/rank1": eval_metrics["rank1"],
                    "val/rank5": eval_metrics["rank5"],
                    "val/rank10": eval_metrics["rank10"],
                }
            )
        wandb.log(wandb_payload)

        ckpt = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "args": vars(args),
            "eval_metrics": eval_metrics,
            "train_pid2label": train_set.pid2label,
        }

        torch.save(ckpt, os.path.join(args.output_dir, "last_model.pth"))

        current_rank1 = eval_metrics["rank1"] if eval_metrics is not None else -train_metrics["loss"]
        if current_rank1 > best_rank1:
            best_rank1 = current_rank1
            torch.save(ckpt, best_path)
            logger.info("Saved new best checkpoint to %s", best_path)

    logger.info("Loading best checkpoint from %s", best_path)
    best_ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(best_ckpt["model_state_dict"])
    logger.info("Running final test evaluation on sequences: %s", test_sequences)
    test_metrics = evaluate(model, test_query_loader, test_gallery_loader, device)

    logger.info("Training finished.")
    if val_query_loader is not None and val_gallery_loader is not None:
        logger.info("Best validation Rank-1: %.4f", best_rank1)
    logger.info(
        "Test results | sequences=%s | mAP=%.4f | R1=%.4f | R5=%.4f | R10=%.4f",
        ",".join(test_sequences),
        test_metrics["mAP"],
        test_metrics["rank1"],
        test_metrics["rank5"],
        test_metrics["rank10"],
    )

    wandb.log(
        {
            "test/mAP": test_metrics["mAP"],
            "test/rank1": test_metrics["rank1"],
            "test/rank5": test_metrics["rank5"],
            "test/rank10": test_metrics["rank10"],
        }
    )

    wandb.finish()


if __name__ == "__main__":
    main()
