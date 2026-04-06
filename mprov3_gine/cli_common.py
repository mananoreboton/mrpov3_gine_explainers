"""Shared argparse pieces for train.py and classify.py."""

from __future__ import annotations

import argparse

from mprov3_gine_explainer_defaults import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_DROPOUT,
    DEFAULT_HIDDEN_CHANNELS,
    DEFAULT_NUM_FOLDS,
    DEFAULT_NUM_LAYERS,
    DEFAULT_OUT_CLASSES,
    DEFAULT_RESULTS_ROOT,
    DEFAULT_TEST_SPLIT_FILE,
    DEFAULT_TRAIN_SPLIT_FILE,
    DEFAULT_TRAINING_CHECKPOINT_FILENAME,
    DEFAULT_VAL_SPLIT_FILE,
)


def add_data_and_results_roots(
    parser: argparse.ArgumentParser,
    *,
    results_help: str,
    data_root_help: str = "Path to raw MPro snapshot (Splits/, Info.csv); default: DEFAULT_DATA_ROOT",
) -> None:
    parser.add_argument(
        "--data_root",
        type=str,
        default=None,
        help=data_root_help,
    )
    parser.add_argument(
        "--results_root",
        type=str,
        default=None,
        help=f"Root for outputs (default: {DEFAULT_RESULTS_ROOT}); {results_help}",
    )


def add_split_and_fold_args(
    parser: argparse.ArgumentParser,
    *,
    fold_indices_help: str,
) -> None:
    parser.add_argument(
        "--train_split_file",
        type=str,
        default=DEFAULT_TRAIN_SPLIT_FILE,
        help=f"Train split file in Splits/ (default: {DEFAULT_TRAIN_SPLIT_FILE})",
    )
    parser.add_argument(
        "--val_split_file",
        type=str,
        default=DEFAULT_VAL_SPLIT_FILE,
        help=f"Val split file in Splits/ (default: {DEFAULT_VAL_SPLIT_FILE})",
    )
    parser.add_argument(
        "--test_split_file",
        type=str,
        default=DEFAULT_TEST_SPLIT_FILE,
        help=f"Test split file in Splits/ (default: {DEFAULT_TEST_SPLIT_FILE})",
    )
    parser.add_argument(
        "--num_folds",
        type=int,
        default=DEFAULT_NUM_FOLDS,
        help=f"Number of folds (default: {DEFAULT_NUM_FOLDS})",
    )
    parser.add_argument(
        "--fold_indices",
        type=int,
        nargs="+",
        default=None,
        metavar="K",
        help=fold_indices_help,
    )


def add_checkpoint_arg(parser: argparse.ArgumentParser, *, help: str) -> None:
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=DEFAULT_TRAINING_CHECKPOINT_FILENAME,
        help=help,
    )


def add_batch_size_arg(parser: argparse.ArgumentParser, *, for_classification: bool) -> None:
    if for_classification:
        parser.add_argument(
            "--batch_size",
            type=int,
            default=DEFAULT_BATCH_SIZE,
            help="Batch size for test-set classification",
        )
    else:
        parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE)


def add_model_loader_args(
    parser: argparse.ArgumentParser,
    *,
    for_classification: bool,
    include_num_classes: bool = True,
) -> None:
    match_kw: dict = {}
    if for_classification:
        match_kw = {"help": "Must match trained model"}
    parser.add_argument(
        "--hidden", type=int, default=DEFAULT_HIDDEN_CHANNELS, **match_kw
    )
    parser.add_argument(
        "--num_layers", type=int, default=DEFAULT_NUM_LAYERS, **match_kw
    )
    parser.add_argument(
        "--dropout", type=float, default=DEFAULT_DROPOUT, **match_kw
    )
    if include_num_classes:
        num_cls_help = (
            f"Number of classes (Category, default: {DEFAULT_OUT_CLASSES})"
            if not for_classification
            else f"Number of classes (default: {DEFAULT_OUT_CLASSES})"
        )
        parser.add_argument(
            "--num_classes",
            type=int,
            default=DEFAULT_OUT_CLASSES,
            help=num_cls_help,
        )
