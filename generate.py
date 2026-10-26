"""Generate a TinyStories continuation from a saved training checkpoint."""

import argparse
import os

# Avoid reserving all GPU memory before JAX initializes the device.
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')

import jax
import jax.numpy as jnp
from transformers import PreTrainedTokenizerFast

from hyperlinear import config_classes
from hyperlinear.data import tinystories
from hyperlinear.model import transformer
from hyperlinear.training.experiment import Experiment


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--ckpt-path', required=True, help='Training checkpoint directory')
  parser.add_argument('--prompt', required=True, help='Text to continue')
  parser.add_argument('--max-new-tokens', type=int, default=40)
  parser.add_argument('--temperature', type=float, default=0.8)
  parser.add_argument('--top-k', type=int, default=40, help='Use 0 to disable top-k')
  parser.add_argument('--seed', type=int, default=0)
  return parser.parse_args()


def main() -> None:
  args = parse_args()
  if args.max_new_tokens < 0:
    raise ValueError('--max-new-tokens must be non-negative')
  if args.temperature < 0:
    raise ValueError('--temperature must be non-negative')
  if args.top_k < 0:
    raise ValueError('--top-k must be non-negative')

  config, state = Experiment.load(args.ckpt_path)
  if not isinstance(config.dataset, config_classes.TinystoriesConfig):
    raise ValueError('This script currently supports TinyStories checkpoints only')

  tokenizer = PreTrainedTokenizerFast(
    tokenizer_file=os.path.join(config.dataset.tokenizer_path, 'tokenizer.json')
  )
  tokenizer.add_special_tokens(tinystories.SPECIAL_TOKENS)

  model = transformer.Transformer(
    n_vocab=len(tokenizer),
    d_model=config.model.d_model,
    d_heads=config.model.d_heads,
    n_layers=config.model.n_layers,
    feedforward_config=config.model.feedforward_config,
    dtype=config.model.dtype,
    remat=config.model.remat,
  )

  max_context = config.dataset.seq_len
  if max_context < 2:
    raise ValueError('Checkpoint seq_len must be at least 2')

  @jax.jit
  def predict_next(
    params: dict, input_ids: jax.Array, position: jax.Array
  ) -> jax.Array:
    logits = model.apply(params, input_ids)
    return logits[0, position, :].astype(jnp.float32)

  rng = jax.random.key(args.seed)
  generated = tokenizer.encode(args.prompt, add_special_tokens=False)
  if not generated:
    raise ValueError('Prompt produced no tokens; try a longer prompt')

  for _ in range(args.max_new_tokens):
    # Training prepends EOS as BOS. Keep the newest context if it exceeds seq_len.
    context = generated[-(max_context - 1) :]
    model_input = [tokenizer.eos_token_id, *context]
    position = len(model_input) - 1
    model_input.extend([tokenizer.pad_token_id] * (max_context - len(model_input)))
    input_array = jnp.asarray([model_input], dtype=jnp.int32)
    logits = predict_next(
      state.params, input_array, jnp.asarray(position, dtype=jnp.int32)
    )

    if args.temperature == 0:
      next_token = int(jnp.argmax(logits))
    else:
      logits = logits / args.temperature
      if 0 < args.top_k < logits.shape[-1]:
        top_logits, top_ids = jax.lax.top_k(logits, args.top_k)
        rng, sample_rng = jax.random.split(rng)
        selected = jax.random.categorical(sample_rng, top_logits)
        next_token = int(top_ids[selected])
      else:
        rng, sample_rng = jax.random.split(rng)
        next_token = int(jax.random.categorical(sample_rng, logits))

    if next_token == tokenizer.eos_token_id:
      break
    generated.append(next_token)

  print(tokenizer.decode(generated, skip_special_tokens=True))


if __name__ == '__main__':
  main()
