"""
Local test run script for multitask model.

Run from the project root. Choose one encoder (required):
    python -m multitask.local_test_runs.run --universal-sentence-encoder   # default
    python -m multitask.local_test_runs.run --openai   # requires OPENAI_API_KEY
    python -m multitask.local_test_runs.run --pytorch-frozen   # PyTorch frozen encoder (e.g. Llama)
    python -m multitask.local_test_runs.run --custom-embeddings --train-embeddings X --val-embeddings Y --test-embeddings Z -d DIM

Use --use-existing to load an existing model instead of training:
    python -m multitask.local_test_runs.run --universal-sentence-encoder --use-existing
    python -m multitask.local_test_runs.run --openai --use-existing
"""

from multitask import ModelConfig, TrainingConfig, EncoderConfig, MultiTaskModel, Trainer
from multitask import load_model
from multitask.config import EncoderInputType
from multitask.models.encoders import PyTorchFrozenEncoder
from multitask.local_test_runs.helpers import dataframe_to_dataset, embeddings_and_labels_to_dataset

import pandas as pd
import tensorflow as tf
import numpy as np
import os
import shutil
import argparse
import sys

# -----------------------------------------------------------------------------
# Project paths
# -----------------------------------------------------------------------------
script_dir = os.path.dirname(os.path.abspath(__file__))


def _find_project_root() -> str:
    """Walk up from script dir to find repo root (directory that contains 'multitask' package)."""
    d = script_dir
    while d != os.path.dirname(d):
        if os.path.isdir(os.path.join(d, "multitask")):
            return d
        d = os.path.dirname(d)
    return script_dir


project_root = _find_project_root()
tfhub_cache = os.path.join(project_root, "tfhub_cache")
data_path = os.path.join(project_root, "multitask", "data")

# -----------------------------------------------------------------------------
# Task configuration
# -----------------------------------------------------------------------------
TASKS = {
    'disagree_agree_bi': {'type': 'binary', 'num_classes': 2},
    'attacking_respectful_bi': {'type': 'binary', 'num_classes': 2},
    'emotion_fact_bi': {'type': 'binary', 'num_classes': 2},
}

DEFAULT_OPENAI_MODEL = "openai:text-embedding-3-small"
DEFAULT_PYTORCH_FROZEN_MODEL = "knowledgator/Llama-encoder-1.0B"


def _get_encoder_config(
    encoder_type: str, 
    model_name: str | None = None,
    embedding_dim: int | None = None
) -> EncoderConfig:
    """Build EncoderConfig for the chosen encoder."""
    if encoder_type == 'universal-sentence-encoder':
        return EncoderConfig(
            encoder_identifier='https://tfhub.dev/google/universal-sentence-encoder/4',
            trainable_encoder=False,
        )
    if encoder_type == 'openai':
        raw = model_name or DEFAULT_OPENAI_MODEL
        identifier = raw if raw.startswith("openai:") else f"openai:{raw}"
        return EncoderConfig(
            encoder_identifier=identifier,
            trainable_encoder=False,
            api_key=os.environ.get('OPENAI_API_KEY'),
        )
    if encoder_type == 'pytorch-frozen':
        name = model_name or DEFAULT_PYTORCH_FROZEN_MODEL
        return EncoderConfig(
            encoder=PyTorchFrozenEncoder(name),
            encoder_input_type=EncoderInputType.RAW_STRING,
            trainable_encoder=False,
        )
    if encoder_type == 'custom-embeddings':
        return EncoderConfig(
            embedding_dim=embedding_dim,
            trainable_encoder=False
        )
    raise ValueError(f"Unknown encoder: {encoder_type}")


def _build_model_config(
    encoder_type: str, 
    model_name: str | None = None,
    embedding_dim: int | None = None
) -> ModelConfig:
    """Build ModelConfig from encoder type and task definitions."""
    encoder_config = _get_encoder_config(
        encoder_type,
        model_name=model_name, 
        embedding_dim=embedding_dim
    )
    
    return ModelConfig(
        task_structure=[[3]],
        task_names=list(TASKS.keys()),
        task_types=[t['type'] for t in TASKS.values()],
        num_classes_per_task=[t['num_classes'] for t in TASKS.values()],
        encoder_config=encoder_config,
    )


def get_model_paths(encoder_type: str):
    """
    Resolve model directory and required file paths for the encoder.
    Returns (model_path, weights_path, config_path, metadata_path, model_exists).
    """
    models_dir = os.path.join(project_root, 'models')
    os.makedirs(models_dir, exist_ok=True)
    subdirs = {'openai': 'my_model_openai', 'pytorch-frozen': 'my_model_pytorch_frozen'}
    subdir = subdirs.get(encoder_type, 'my_model_use')
    model_path = os.path.join(models_dir, subdir)
    weights_path = os.path.join(model_path, 'weights.weights.h5')
    config_path = os.path.join(model_path, 'config.json')
    metadata_path = os.path.join(model_path, 'metadata.json')
    model_exists = all(os.path.exists(p) for p in [weights_path, config_path, metadata_path])
    return model_path, weights_path, config_path, metadata_path, model_exists


def load_train_val_data(task_names, batch_size=32):
    """Load train/val CSVs and return (train_dataset, val_dataset)."""
    print("Loading data...")
    data = pd.read_csv(os.path.join(data_path, 'no_prop_overlap_train.csv'))
    data_val = pd.read_csv(os.path.join(data_path, 'no_prop_overlap_val.csv'))
    print("Converting DataFrames to datasets...")
    train_ds = dataframe_to_dataset(data, task_names, batch_size=batch_size, shuffle=True)
    val_ds = dataframe_to_dataset(data_val, task_names, batch_size=batch_size, shuffle=False)
    return train_ds, val_ds


def verify_openai_encoder(base_model, embeddings_only: bool):
    """
    Verify OpenAI encoder: check API key, fetch test embeddings, save to file,
    print stats, and run a full model forward pass.
    """
    print("\n" + "=" * 60)
    print("Verifying OpenAI encoder embeddings...")
    print("=" * 60)

    api_key = os.environ.get('OPENAI_API_KEY')
    if not api_key:
        print("WARNING: OPENAI_API_KEY not found in environment!")
        print("Set it with: export OPENAI_API_KEY='sk-...'")
    else:
        print(f"✓ OPENAI_API_KEY found (length: {len(api_key)})")

    test_texts = tf.constant(
        ["This is a test sentence for embedding verification."],
        dtype=tf.string,
    )

    if not base_model.uses_encoder or base_model.encoder is None:
        print("WARNING: Model does not appear to use an encoder!")
        print("=" * 60 + "\n")
        return

    print(f"\nEncoder type: {base_model.encoder_type}")
    print(f"Encoder input type: {base_model.encoder_input_type}")
    print(f"Encoder trainable: {base_model.encoder.trainable}")
    print(f"Expected embedding dim: {base_model.encoder.embedding_dim}")

    if hasattr(base_model.encoder, 'client'):
        try:
            print("Initializing OpenAI client...")
            _ = base_model.encoder.client
            print("✓ OpenAI client initialized")
        except Exception as e:
            print(f"✗ Failed to initialize OpenAI client: {e}")
            raise

    try:
        if embeddings_only:
            # Generate embeddings for all no_prop_overlap_*.csv files
            print("\nGenerating embeddings for all no_prop_overlap_*.csv files...")
            csv_files = sorted(glob.glob(os.path.join(data_path, "no_prop_overlap_*.csv")))
            
            if not csv_files:
                print(f"  No CSV files found in {data_path}")
                return
            
            for csv_path in csv_files:
                file_name = Path(csv_path).name
                # Extract split name (train, val, test)
                split = file_name.replace("no_prop_overlap_", "").replace(".csv", "")
                
                print(f"\n  Processing {file_name}...")
                
                # Load texts
                texts = _load_texts_from_csv(csv_path)
                if not texts:
                    print(f"    ✗ No texts found in {file_name}")
                    continue
                
                print(f"    Loaded {len(texts)} texts")
                
                # Get embeddings
                print(f"    Generating embeddings...")
                embeddings = _batch_embeddings(texts, base_model.encoder, batch_size=50)
                
                # Save embeddings
                output_path = os.path.join(data_path, f"openai_embeddings_{split}.npy")
                np.save(output_path, embeddings)
                print(f"    ✓ Saved to {output_path}")
                print(f"      Shape: {embeddings.shape}")
            
            print("\n✓ All embeddings generated successfully!")
            return
        
        # Single test embedding (original behavior)
        test_texts = tf.constant(
            ["This is a test sentence for embedding verification."],
            dtype=tf.string,
        )
        
        print("\nCalling OpenAI API to get embeddings...")
        embeddings = base_model.encoder(test_texts, training=False)

        embeddings_path = os.path.join(script_dir, "openai_embeddings.npy")
        np.save(embeddings_path, embeddings.numpy())
        print(f"✓ Saved embeddings to {embeddings_path}")

        print("✓ Successfully received embeddings")
        print(f"\nEmbedding shape: {embeddings.shape}")

        if embeddings_only:
            return

        sample_values = embeddings.numpy()[0][:10]
        print("\nSample embedding values (first 10 dims):")
        print(f"  {sample_values}")
        print("\nEmbedding statistics:")
        print(f"  Min: {tf.reduce_min(embeddings).numpy():.6f}")
        print(f"  Max: {tf.reduce_max(embeddings).numpy():.6f}")
        print(f"  Mean: {tf.reduce_mean(embeddings).numpy():.6f}")
        print(f"  Std: {tf.math.reduce_std(embeddings).numpy():.6f}")

        if embeddings.shape[1] == base_model.encoder.embedding_dim:
            print(f"\n✓ Embedding dimension matches expected: {embeddings.shape[1]}")
        else:
            print("\n✗ WARNING: Embedding dimension mismatch!")
            print(f"  Expected: {base_model.encoder.embedding_dim}, Got: {embeddings.shape[1]}")

        print("\nTesting full model forward pass...")
        model_output = base_model(test_texts, training=False)
        print(f"Model output keys: {list(model_output.keys())}")
        for task_name, output in model_output.items():
            print(f"  {task_name}: shape {output.shape}, dtype {output.dtype}")

        print("\n✓ All checks passed!")
    except Exception as e:
        print("\n✗ ERROR: Failed to get embeddings from OpenAI encoder")
        print(f"  Error type: {type(e).__name__}")
        print(f"  Error message: {str(e)}")
        if "API key" in str(e) or "authentication" in str(e).lower():
            print("\n  This might be an API key issue. Check:")
            print("  1. OPENAI_API_KEY is set correctly")
            print("  2. API key is valid and has credits")
            print("  3. Network connection is working")
        raise

    print("=" * 60 + "\n")
    # print(f"OpenAI encoding is currently disabled. Uncomment code to enable.")
    return


def create_base_model(config):
    """
    Create MultiTaskModel(config). Handles TF Hub cache ValueError and re-raises.
    """
    try:
        return MultiTaskModel(config)
    except ValueError as e:
        if 'saved_model.pb' in str(e) and 'saved_model.pbtxt' in str(e):
            print("\nTensorFlow Hub cache error: cached encoder is corrupted or incomplete.")
            print(f"Clear cache and re-run:  rm -rf {tfhub_cache}")
            print("Or system cache:  rm -rf /var/folders/*/T/tfhub_modules  # macOS")
            raise SystemExit(1) from e
        raise


def run_training(model, train_dataset, val_dataset, num_tasks, model_path):
    """Train, save, and reload model; return (model, thresholds)."""
    training_config = TrainingConfig()
    trainer = Trainer(model, training_config)
    print("Training model...")
    history, thresholds = trainer.fit(
        train_dataset, val_dataset, num_tasks=num_tasks, optimize_thresholds=True
    )
    print("Saving model...")
    model.save(model_path, thresholds=thresholds)
    print("Loading model for predictions...")
    return load_model(model_path)


def run_predictions(
    model,
    custom_embeddings: dict | None = None,
    batch_size: int = 32,
):
    """Load test data, run model.predict, return predictions dict."""
    if custom_embeddings is not None:
        test_emb = tf.constant(custom_embeddings['test'], dtype=tf.float32)
        dataset = (
            tf.data.Dataset.from_tensor_slices(test_emb)
            .batch(batch_size)
            .prefetch(tf.data.AUTOTUNE)
        )
    else:
        data_test = pd.read_csv(os.path.join(data_path, 'no_prop_overlap_test.csv'))
        texts_list = data_test['text'].astype(str).tolist()
        texts_tensor = tf.constant(texts_list, dtype=tf.string)
        dataset = tf.data.Dataset.from_tensor_slices(texts_tensor).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return model.predict(dataset)

def parse_custom_embeddings(
    train_fp: str, val_fp: str, test_fp: str
) -> dict[str, np.ndarray]:
    """Load custom embeddings from .npy files for train, val, and test.

    Each file must be a 2D array of shape (n_samples, embedding_dim). Row order
    must match the corresponding CSV (train/val/test) so labels can be aligned.

    Returns:
        Dict with keys 'train', 'val', 'test', each value a np.ndarray
        of shape (n_samples, embedding_dim), dtype float32.
    """
    out = {}
    for key, path in [('train', train_fp), ('val', val_fp), ('test', test_fp)]:
        arr = np.load(path)
        if arr.ndim != 2:
            raise ValueError(
                f"Custom embeddings '{path}' must be 2D (n_samples, embedding_dim), got shape {arr.shape}"
            )
        out[key] = np.asarray(arr, dtype=np.float32)
    return out


# -----------------------------------------------------------------------------
# Main entry
# -----------------------------------------------------------------------------
def main(
    encoder_type: str, 
    use_existing: bool, 
    model_name: str | None = None, 
    clear_cache: bool = False,
    embeddings_only: bool = False,
    custom_embeddings_train: str | None = None,
    custom_embeddings_val: str | None = None,
    custom_embeddings_test: str | None = None,
    embedding_dim: int | None = None
):
    """
    Run the training/prediction pipeline for the chosen encoder.
    """
    custom_embeddings = None
    use_custom_embeddings = (
        encoder_type == 'custom-embeddings'
        and custom_embeddings_train
        and custom_embeddings_val
        and custom_embeddings_test
    )
    if encoder_type == 'custom-embeddings':
        if not (custom_embeddings_train and custom_embeddings_val and custom_embeddings_test):
            print(
                "Error: --custom-embeddings requires --train-embeddings, --val-embeddings, and --test-embeddings.",
                file=sys.stderr,
            )
            sys.exit(1)
        if embedding_dim is None:
            print("Error: --embedding-dim (-d) is required when using --custom-embeddings.", file=sys.stderr)
            sys.exit(1)
    if use_custom_embeddings:
        custom_embeddings = parse_custom_embeddings(
            custom_embeddings_train,
            custom_embeddings_val,
            custom_embeddings_test,
        )

    if encoder_type == 'universal-sentence-encoder':
        if clear_cache and os.path.isdir(tfhub_cache):
            shutil.rmtree(tfhub_cache)
            print(f"Cleared TFHub cache at {tfhub_cache}")

        os.makedirs(tfhub_cache, exist_ok=True)
        os.environ.setdefault('TFHUB_CACHE_DIR', tfhub_cache)

    config = _build_model_config(encoder_type, model_name=model_name, embedding_dim=embedding_dim)
    task_names = config.task_names
    num_tasks = len(task_names)

    model_path, weights_path, config_path, metadata_path, model_exists = get_model_paths(encoder_type)

    if use_existing:
        if not model_exists:
            print(f"Error: Model not found at {model_path}")
            print("Required files:", weights_path, config_path, metadata_path)
            sys.exit(1)
        print(f"Loading existing model from {model_path}...")
        return load_model(model_path), custom_embeddings

    if embeddings_only:
        if encoder_type != 'openai':
            print("Warning: --embeddings-only is only supported with --openai. Ignoring.")
        else:
            print("Creating model...")
            model = create_base_model(config)
            verify_openai_encoder(model, embeddings_only=True)
            return None, custom_embeddings

    if use_custom_embeddings:
        # Combine custom embeddings with task names
        print("Loading data...")
        data = pd.read_csv(os.path.join(data_path, 'no_prop_overlap_train.csv'))
        data_val = pd.read_csv(os.path.join(data_path, 'no_prop_overlap_val.csv'))
        train_dataset = embeddings_and_labels_to_dataset(
            custom_embeddings['train'],
            data,
            task_names,
            batch_size=32,
            shuffle=True,
        )
        val_dataset = embeddings_and_labels_to_dataset(
            custom_embeddings['val'],
            data_val,
            task_names,
            batch_size=32,
            shuffle=False,
        )
    else:
        train_dataset, val_dataset = load_train_val_data(task_names)

    if model_exists:
        print(f"Note: Existing model at {model_path} will be overwritten.")

    print("Creating model...")
    model = create_base_model(config)
    if encoder_type == 'openai':
        verify_openai_encoder(model, embeddings_only)

    return run_training(model, train_dataset, val_dataset, num_tasks, model_path), custom_embeddings


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=(
            'Train or load a multitask model. Choose one encoder: --universal-sentence-encoder, '
            '--openai, --pytorch-frozen, or --custom-embeddings.'
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""

        Examples:
        python -m multitask.local_test_runs.run --universal-sentence-encoder
        python -m multitask.local_test_runs.run --openai
        python -m multitask.local_test_runs.run --openai --use-existing
        python -m multitask.local_test_runs.run --openai --model-name "openai:text-embedding-3-small"
        python -m multitask.local_test_runs.run --pytorch-frozen
        python -m multitask.local_test_runs.run --pytorch-frozen --model-name "knowledgator/Llama-encoder-1.0B"
        python -m multitask.local_test_runs.run --custom-embeddings --train-embeddings train.npy --val-embeddings val.npy --test-embeddings test.npy -d 1536
        """,
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        '--universal-sentence-encoder',
        action='store_true',
        help='Use Universal Sentence Encoder from TensorFlow Hub (default encoder).',
    )
    group.add_argument(
        '--openai',
        action='store_true',
        help='Use OpenAI text-embedding-3-small (requires OPENAI_API_KEY).',
    )
    group.add_argument(
        '--pytorch-frozen',
        action='store_true',
        help='Use PyTorch frozen encoder (knowledgator/Llama-encoder-1.0B).',
    )
    group.add_argument(
        '--custom-embeddings',
        action='store_true',
        help='Use custom embeddings from .npy files (requires --train-embeddings, --val-embeddings, --test-embeddings, -d).',
    )
    parser.add_argument(
        '--use-existing',
        action='store_true',
        help='Load an existing model instead of training. Exit with error if not found.',
    )
    parser.add_argument(
        '--model-name',
        type=str,
        default=None,
        metavar='NAME',
        help='Encoder model (--openai: e.g. "openai:text-embedding-3-small" or "text-embedding-3-small"; '
             '--pytorch-frozen: e.g. "knowledgator/Llama-encoder-1.0B"). Uses script default if omitted.',
    )
    parser.add_argument(
        '--clear-cache',
        action='store_true',
        help='Clear TFHub cache before running (only applies with --universal-sentence-encoder). Sometimes needed if file is corrupted or training doesn\'t complete.',
    )
    parser.add_argument(
        '--embeddings-only',
        action='store_true',
        help='Only get embeddings. Don\'t run training or predictions.'
    )
    parser.add_argument(
        '--train-embeddings',
        dest='custom_embeddings_train',
        type=str,
        default=None,
        metavar='PATH',
        help='Path to training set embeddings (.npy), same row order as train CSV',
    )
    parser.add_argument(
        '--val-embeddings',
        dest='custom_embeddings_val',
        type=str,
        default=None,
        metavar='PATH',
        help='Path to validation set embeddings (.npy), same row order as val CSV',
    )
    parser.add_argument(
        '--test-embeddings',
        dest='custom_embeddings_test',
        type=str,
        default=None,
        metavar='PATH',
        help='Path to test set embeddings (.npy), same row order as test CSV (for prediction)',
    )
    parser.add_argument(
        '-d',
        '--embedding-dim',
        dest='embedding_dim',
        type=int,
        default=None,
        help='Embedding dimension for custom embeddings'
    )
    args = parser.parse_args()

    encoder_type = (
        'openai' if args.openai
        else 'pytorch-frozen' if args.pytorch_frozen
        else 'universal-sentence-encoder' if args.universal_sentence_encoder
        else 'custom-embeddings' if args.custom_embeddings
        else 'universal-sentence-encoder'
    )
    
    use_custom_embeddings = (
        encoder_type == 'custom-embeddings'
        and args.custom_embeddings_train is not None
        and args.custom_embeddings_val is not None
        and args.custom_embeddings_test is not None
    )

    result, custom_embeddings = main(
        encoder_type=encoder_type,
        use_existing=args.use_existing,
        model_name=args.model_name,
        clear_cache=args.clear_cache,
        embeddings_only=args.embeddings_only,
        custom_embeddings_train=args.custom_embeddings_train,
        custom_embeddings_val=args.custom_embeddings_val,
        custom_embeddings_test=args.custom_embeddings_test,
        embedding_dim=args.embedding_dim,
    )

    if result is not None:
        model, thresholds = result
        print("Making predictions...")
        predictions = run_predictions(
            model,
            custom_embeddings
        )
        print("Predictions completed!")
        print(f"Prediction shapes: {[f'{k}: {v.shape}' for k, v in predictions.items()]}")
