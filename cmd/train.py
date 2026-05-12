#!/usr/bin/env python
"""
Training command for machine learning models.

Usage:
    python main.py train --data custom_data.csv --epochs 20
    python -m cmd.train --data custom_data.csv --epochs 20
"""

import argparse
import logging
from ultralytics import YOLO
from config import Config

logger = logging.getLogger(__name__)


def parse_args(argv, config: Config):
    """Parse command line arguments with config defaults"""
    parser = argparse.ArgumentParser(
        description="Train a machine learning model"
    )

    parser.add_argument(
        "--data", "-d",
        type=str,
        default=config.Train.train_data_path,
        help=f"Path to training data (default: {config.Train.train_data_path})"
    )

    parser.add_argument(
        "--output", "-o",
        type=str,
        default="models/best.pt",
        help="Path to save the trained model"
    )

    parser.add_argument(
        "--epochs", "-e",
        type=int,
        default=50,
        help="Number of training epochs (default 50)"
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=config.Train.batch_size,
        help=f"Batch size for training (default: {config.Train.batch_size})"
    )

    parser.add_argument(
        "--device",
        type=str,
        default=config.device,
        help=f"Device to use (default: {config.device})"
    )

    parser.add_argument(
        "--yolo-model",
        type=str,
        default=config.Train.yolo_base_model,
        help=f"YOLO base model (default: {config.Train.yolo_base_model})"
    )

    parser.add_argument(
        "--cache-dir",
        type=str,
        default=config.cache_dir,
        help=f"Cache directory (default: {config.cache_dir})"
    )

    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable caching"
    )

    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output"
    )

    return parser.parse_args(argv)


def train_model(args, config: Config):
    """Train the model with given arguments and config"""
    args = parse_args(args, config)

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level, format='%(asctime)s - %(levelname)s - %(message)s')

    config.batch_size = args.batch_size
    config.device = args.device
    config.yolo_base_model = args.yolo_model
    config.cache_dir = args.cache_dir
    config.use_cache = not args.no_cache

    logger.info("Training configuration:")
    logger.info("  Data: %s", {args.data})
    logger.info("  Model save path: %s", {args.output})
    logger.info("  Epochs: %s", {args.epochs})
    logger.info("  Device: %s", {config.device})
    logger.info("  Batch size: %s", {config.batch_size})
    logger.info("  YOLO model: %s", {config.yolo_base_model})
    logger.info("  Cache: %s", {'Enabled' if config.use_cache else 'Disabled'})
    logger.info("  Cache dir: %s", {config.cache_dir})

    model = YOLO(config.Train.yolo_base_model)
    model.train(
        data=args.data,
        epochs=args.epochs,
        batch=config.Train.batch_size,
        device=config.device,
        conf=config.conf,
    )

    model.save(args.output)
    logger.info("Model saved to: %s", {args.output})

    logger.info("Training completed successfully!")
