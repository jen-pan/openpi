import dataclasses
import functools
import logging
import platform
from typing import Any
from rich import print
import os, logging

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"          # hush XLA/TF INFO
logging.getLogger("jax").setLevel(logging.ERROR) 

# Use a user-specific directory in /tmp for JAX's persistent compilation cache.
# This avoids permission or quota issues that can arise if multiple users share
# the same machine or if the default ~/.cache path is quota-restricted.
cache_dir = f"/tmp/jax_cache_jrpan"
# Ensure the directory exists and is writable.
os.makedirs(cache_dir, exist_ok=True)
# Expose it both to JAX via the env-var *before* importing JAX, and later via
# jax.config.update.
os.environ["JAX_COMPILATION_CACHE_DIR"] = cache_dir
CACHE_DIR = cache_dir

import jax
import jax.experimental
import jax.numpy as jnp
import numpy as np
import optax
import tqdm_loggable.auto as tqdm
import wandb

import etils.epath as epath
import flax.nnx as nnx
from flax.training import common_utils
import flax.traverse_util as traverse_util

import openpi.models.model as _model
import openpi.shared.array_typing as at
import openpi.shared.nnx_utils as nnx_utils
import openpi.training.checkpoints as _checkpoints
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.training.optimizer as _optimizer
import openpi.training.sharding as sharding
import openpi.training.utils as training_utils
import openpi.training.weight_loaders as _weight_loaders


def init_logging():
    """Custom logging format for better readability."""
    level_mapping = {"DEBUG": "D", "INFO": "I", "WARNING": "W", "ERROR": "E", "CRITICAL": "C"}

    class CustomFormatter(logging.Formatter):
        def format(self, record):
            record.levelname = level_mapping.get(record.levelname, record.levelname)
            return super().format(record)

    formatter = CustomFormatter(
        fmt="%(asctime)s.%(msecs)03d [%(levelname)s] %(message)-80s (%(process)d:%(filename)s:%(lineno)s)",
        datefmt="%H:%M:%S",
    )

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers[0].setFormatter(formatter)


def init_wandb(config: _config.TrainConfig, *, resuming: bool, log_code: bool = False, enabled: bool = True):
    if not enabled:
        wandb.init(mode="disabled")
        return

    ckpt_dir = config.checkpoint_dir
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory {ckpt_dir} does not exist.")
    if resuming:
        run_id = (ckpt_dir / "wandb_id.txt").read_text().strip()
        wandb.init(id=run_id, resume="must", project=config.project_name)
    else:
        wandb.init(
            name=config.exp_name,
            config=dataclasses.asdict(config),
            project=config.project_name,
        )
        (ckpt_dir / "wandb_id.txt").write_text(wandb.run.id)

    if log_code:
        wandb.run.log_code(epath.Path(__file__).parent.parent)


def _load_weights_and_validate(loader: _weight_loaders.WeightLoader, params_shape: at.Params) -> at.Params:
    """Loads and validates the weights. Returns a loaded subset of the weights."""
    loaded_params = loader.load(params_shape)
    at.check_pytree_equality(expected=params_shape, got=loaded_params, check_shapes=True, check_dtypes=True)

    # Remove jax.ShapeDtypeStruct from the loaded params. This makes sure that only the loaded params are returned.
    return traverse_util.unflatten_dict(
        {k: v for k, v in traverse_util.flatten_dict(loaded_params).items() if not isinstance(v, jax.ShapeDtypeStruct)}
    )


@at.typecheck
def init_train_state(
    config: _config.TrainConfig, init_rng: at.KeyArrayLike, mesh: jax.sharding.Mesh, *, resume: bool
) -> tuple[training_utils.TrainState, Any]:
    tx = _optimizer.create_optimizer(config.optimizer, config.lr_schedule, weight_decay_mask=None)

    def init(rng: at.KeyArrayLike, partial_params: at.Params | None = None) -> training_utils.TrainState:
        rng, model_rng = jax.random.split(rng)
        # initialize the model (and its parameters).
        model = config.model.create(model_rng)

        # Merge the partial params into the model.
        if partial_params is not None:
            graphdef, state = nnx.split(model)
            # This will produce an error if the partial params are not a subset of the state.
            state.replace_by_pure_dict(partial_params)
            model = nnx.merge(graphdef, state)

        params = nnx.state(model)
        # Convert frozen params to bfloat16.
        params = nnx_utils.state_map(params, config.freeze_filter, lambda p: p.replace(p.value.astype(jnp.bfloat16)))

        return training_utils.TrainState(
            step=0,
            params=params,
            model_def=nnx.graphdef(model),
            tx=tx,
            opt_state=tx.init(params.filter(config.trainable_filter)),
            ema_decay=config.ema_decay,
            ema_params=None if config.ema_decay is None else params,
        )

    train_state_shape = jax.eval_shape(init, init_rng)
    state_sharding = sharding.fsdp_sharding(train_state_shape, mesh, log=True)

    if resume:
        return train_state_shape, state_sharding

    partial_params = _load_weights_and_validate(config.weight_loader, train_state_shape.params.to_pure_dict())
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    # Initialize the train state and mix in the partial params.
    train_state = jax.jit(
        init,
        donate_argnums=(1,),  # donate the partial params buffer.
        in_shardings=replicated_sharding,
        out_shardings=state_sharding,
    )(init_rng, partial_params)

    return train_state, state_sharding


@at.typecheck
def train_step(
    config: _config.TrainConfig,
    rng: at.KeyArrayLike,
    state: training_utils.TrainState,
    batch: tuple[_model.Observation, _model.Actions],
) -> tuple[training_utils.TrainState, dict[str, at.Array]]:
    model = nnx.merge(state.model_def, state.params)
    model.train()

    @at.typecheck
    def loss_fn(
        model: _model.BaseModel, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions
    ):
        chunked_loss = model.compute_loss(rng, observation, actions, train=True)
        return jnp.mean(chunked_loss) * config.action_loss_weight

    train_rng = jax.random.fold_in(rng, state.step)
    observation, actions = batch

    # Filter out frozen params.
    diff_state = nnx.DiffState(0, config.trainable_filter)
    loss, grads = nnx.value_and_grad(loss_fn, argnums=diff_state)(model, train_rng, observation, actions)
    params = state.params.filter(config.trainable_filter)
    updates, new_opt_state = state.tx.update(grads, state.opt_state, params)
    new_params = optax.apply_updates(params, updates)

    # Update the model in place and return the new full state.
    nnx.update(model, new_params)
    new_params = nnx.state(model)

    new_state = dataclasses.replace(state, step=state.step + 1, params=new_params, opt_state=new_opt_state)
    if state.ema_decay is not None:
        new_state = dataclasses.replace(
            new_state,
            ema_params=jax.tree.map(
                lambda old, new: state.ema_decay * old + (1 - state.ema_decay) * new, state.ema_params, new_params
            ),
        )
    # Filter out params that aren't kernels.
    kernel_params = nnx.state(
        model,
        nnx.All(
            nnx.Param,
            nnx.Not(nnx_utils.PathRegex(".*/(bias|scale|pos_embedding|input_embedding)")),
            lambda _, x: x.value.ndim > 1,
        ),
    )
    info = {
        "loss": loss,
        "grad_norm": optax.global_norm(grads),
        "param_norm": optax.global_norm(kernel_params),
    }
    return new_state, info


@at.typecheck
def train_step_subtask(
	config: _config.TrainConfig,
	rng: at.KeyArrayLike,
	state: training_utils.TrainState,
	observation: _model.Observation,
) -> tuple[training_utils.TrainState, dict[str, at.Array]]:
	model = nnx.merge(state.model_def, state.params)
	model.train()

	def loss_fn(model: _model.BaseModel, rng: at.KeyArrayLike, observation: _model.Observation):
		result = model.compute_subtask_loss(rng, observation, train=True)
		loss, _, _ = result
		return jnp.mean(loss) * config.subtask_loss_weight

	train_rng = jax.random.fold_in(rng, state.step)

	diff_state = nnx.DiffState(0, config.trainable_filter)
	loss, grads = nnx.value_and_grad(loss_fn, argnums=diff_state)(model, train_rng, observation)
	params = state.params.filter(config.trainable_filter)
	updates, new_opt_state = state.tx.update(grads, state.opt_state, params)
	new_params = optax.apply_updates(params, updates)
	nnx.update(model, new_params)
	new_params = nnx.state(model)
	new_state = dataclasses.replace(state, step=state.step + 1, params=new_params, opt_state=new_opt_state)
	if state.ema_decay is not None:
		new_state = dataclasses.replace(
			new_state,
			ema_params=jax.tree.map(
				lambda old, new: state.ema_decay * old + (1 - state.ema_decay) * new, state.ema_params, new_params
			),
		)
	kernel_params = nnx.state(
		model,
		nnx.All(
			nnx.Param,
			nnx.Not(nnx_utils.PathRegex(".*/(bias|scale|pos_embedding|input_embedding)")),
			lambda _, x: x.value.ndim > 1,
		),
	)
	info = {"loss": loss, "grad_norm": optax.global_norm(grads), "param_norm": optax.global_norm(kernel_params)}
	return new_state, info


@at.typecheck
def eval_step(
    config: _config.TrainConfig,
    rng: at.KeyArrayLike,
    state: training_utils.TrainState,
    batch: tuple[_model.Observation, _model.Actions],
) -> dict[str, at.Array]:
    """Run a forward pass without gradient updates and return metrics."""
    # Use EMA parameters for evaluation if available.
    model_params = state.ema_params if state.ema_params is not None else state.params
    model = nnx.merge(state.model_def, model_params)

    @at.typecheck
    def loss_fn(model: _model.BaseModel, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions):
        chunked_loss = model.compute_loss(rng, observation, actions, train=False)
        return jnp.mean(chunked_loss)

    observation, actions = batch
    loss = loss_fn(model, rng, observation, actions)

    return {"eval/action_loss": loss}

def decode_subtask_predictions(batch, predicted_tokens, target_mask):
    try:
        import numpy as np
        from openpi.models import tokenizer as _tokenizer        
        paligemma_tokenizer = _tokenizer.PaligemmaTokenizer()
        observation, _ = batch
        if observation.subtask_target is None:
            return ["NO_SUBTASK_TARGET"], ["NO_SUBTASK_TARGET"]
        
        if predicted_tokens is not None and target_mask is not None:
            num_samples_to_decode = min(20, predicted_tokens.shape[0])
            predicted_tokens_cpu = jax.device_get(predicted_tokens[:num_samples_to_decode])
            target_mask_cpu = jax.device_get(target_mask[:num_samples_to_decode])
            subtask_target_cpu = jax.device_get(observation.subtask_target[:num_samples_to_decode])
            subtask_target_mask_cpu = jax.device_get(observation.subtask_target_mask[:num_samples_to_decode])
            
            decoded_predictions = []
            decoded_targets = []
            
            for i in range(num_samples_to_decode):
                # Decode predictions
                mask_np = np.array(target_mask_cpu[i], dtype=bool)
                valid_tokens = predicted_tokens_cpu[i][mask_np]
                valid_tokens = valid_tokens[valid_tokens != 0]
                
                if len(valid_tokens) > 0:
                    decoded_text = paligemma_tokenizer._tokenizer.decode(valid_tokens.tolist())
                    decoded_predictions.append(decoded_text)
                else:
                    decoded_predictions.append("EMPTY_PREDICTION")
                
                # Decode targets
                target_mask_np = np.array(subtask_target_mask_cpu[i], dtype=bool)
                valid_target_tokens = subtask_target_cpu[i][target_mask_np]
                valid_target_tokens = valid_target_tokens[valid_target_tokens != 0]
                
                if len(valid_target_tokens) > 0:
                    decoded_target = paligemma_tokenizer._tokenizer.decode(valid_target_tokens.tolist())
                    decoded_targets.append(decoded_target)
                else:
                    decoded_targets.append("EMPTY_TARGET")
                    
            return decoded_predictions, decoded_targets
        else:
            return ["PREDICTED_TOKENS_ARE_NONE"], ["PREDICTED_TOKENS_ARE_NONE"]
        
    except Exception as e:
        return [f"DECODE_ERROR: {str(e)}"], [f"DECODE_ERROR: {str(e)}"]


def eval_step_subtask(
    config: _config.TrainConfig,
    rng: at.KeyArrayLike,
    state: training_utils.TrainState,
    batch: tuple[_model.Observation, _model.Actions],
) -> tuple[dict[str, at.Array], at.Array | None, at.Array | None]:
    model_params = state.ema_params if state.ema_params is not None else state.params
    model = nnx.merge(state.model_def, model_params)
    
    observation, _ = batch
    loss, predicted_tokens, target_mask = model.compute_subtask_loss(rng, observation, train=False)
    loss = jnp.mean(loss)
    return {"eval/subtask_loss": loss}, predicted_tokens, target_mask

def main(config: _config.TrainConfig):
    init_logging()
    logging.info(f"Running on: {platform.node()}")
    print("[bold green]TRAIN CONFIG[/]", config)
    print("[bold green]device_count[/]", jax.device_count())
    if config.action_batch_size % jax.device_count() != 0:
        raise ValueError(
            f"Batch size {config.action_batch_size} must be divisible by the number of devices {jax.device_count()}."
        )
    if config.subtask_batch_size % jax.device_count() != 0:
        raise ValueError(
            f"Subtask batch size {config.subtask_batch_size} must be divisible by the number of devices {jax.device_count()}."
        )
    print("[bold green]action_per_device_batch_size[/]", config.action_batch_size // jax.device_count())
    print("[bold green]subtask_per_device_batch_size[/]", config.subtask_batch_size // jax.device_count())
    jax.config.update("jax_compilation_cache_dir", CACHE_DIR)

    rng = jax.random.key(config.seed)
    train_rng, init_rng = jax.random.split(rng)

    mesh = sharding.make_mesh(config.fsdp_devices)
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    checkpoint_manager, resuming = _checkpoints.initialize_checkpoint_dir(
        config.checkpoint_dir,
        keep_period=config.keep_period,
        overwrite=config.overwrite,
        resume=config.resume,
    )
    init_wandb(config, resuming=resuming, enabled=config.wandb_enabled)

    # =========================== action dataloader ===========================
    if config.action_data is not None:
        print("[bold green]ACTION DATALOADER[/]", config.action_data)
        action_loader = _data_loader.create_data_loader(
            config,
            sharding=data_sharding,
            shuffle=True,
            task="action_pred"
        )
        action_eval_loader = _data_loader.create_data_loader(
            config,
            sharding=data_sharding,
            shuffle=False,
            is_eval=True,
            task="action_pred"
        )

        action_iter = iter(action_loader)
        action_eval_iter = iter(action_eval_loader)
    else:
        action_iter = None
        action_loader = None
        action_eval_iter = None
        action_eval_loader = None

    # =========================== subtask dataloader ===========================
    if config.subtask_data is not None:
        print("[bold green]SUBTASK DATALOADER[/]", config.subtask_data)
        subtask_loader = _data_loader.create_data_loader(
            config,
            sharding=data_sharding,
            shuffle=True,
            skip_norm_stats=True,  # no normalization for subtask data
            task="subtask_pred"
        )
        subtask_eval_loader = _data_loader.create_data_loader(
            config,
            sharding=data_sharding,
            shuffle=False,
            is_eval=True,
            skip_norm_stats=True,
            task="subtask_pred"
        )
        subtask_iter = iter(subtask_loader)
        subtask_eval_iter = iter(subtask_eval_loader)
    else:
        subtask_loader = None
        subtask_iter = None
        subtask_eval_loader = None
        subtask_eval_iter = None

    if action_iter is None and subtask_iter is None:
        raise ValueError("At least one of action_data or subtask_data must be provided")

    # =========================== logging and sanity checking ===========================
    if action_iter is not None:
        initial_action_batch = next(action_iter)
        logging.info(f"Initialized data loader:\n{training_utils.array_tree_to_info(initial_action_batch)}")

        action_images_to_log = [
            wandb.Image(np.concatenate([np.array(initial_action_batch[0].images[key][i]) for key in _model.ACTION_PRED_IMAGE_KEYS], axis=1))
            for i in range(min(5, len(next(iter(initial_action_batch[0].images.values())))))
        ]
        wandb.log({"action_camera_views": action_images_to_log}, step=0)
    
    if subtask_iter is not None:
        initial_subtask_batch = next(subtask_iter)
        logging.info(f"Initialized subtask data loader:\n{training_utils.array_tree_to_info(initial_subtask_batch)}")
        
        subtask_images_to_log = [
            wandb.Image(np.concatenate([np.array(initial_subtask_batch[0].images[key][i]) for key in _model.SUBTASK_PRED_IMAGE_KEYS], axis=1))
            for i in range(min(5, len(next(iter(initial_subtask_batch[0].images.values())))))
        ]
        wandb.log({"subtask_camera_views": subtask_images_to_log}, step=0)

    # =========================== initialize train state ===========================
    train_state, train_state_sharding = init_train_state(config, init_rng, mesh, resume=resuming)
    jax.block_until_ready(train_state)
    logging.info(f"Initialized train state:\n{training_utils.array_tree_to_info(train_state.params)}")

    if resuming:
        train_state = _checkpoints.restore_state(checkpoint_manager, train_state)

    ptrain_step = jax.jit(
        functools.partial(train_step, config),
        in_shardings=(replicated_sharding, train_state_sharding, data_sharding),
        out_shardings=(train_state_sharding, replicated_sharding),
        donate_argnums=(1,),
    )
    ptrain_step_subtask = jax.jit(
		functools.partial(train_step_subtask, config),
		in_shardings=(replicated_sharding, train_state_sharding, data_sharding),
		out_shardings=(train_state_sharding, replicated_sharding),
		donate_argnums=(1,),
	)
    peval_step = jax.jit(
        functools.partial(eval_step, config),
        in_shardings=(replicated_sharding, train_state_sharding, data_sharding),
        out_shardings=replicated_sharding,
    )
    peval_step_subtask = jax.jit(
        functools.partial(eval_step_subtask, config),
        in_shardings=(replicated_sharding, train_state_sharding, data_sharding),
        out_shardings=replicated_sharding,
    )

    eval_rng = jax.random.key(config.seed + 1)
    start_step = int(train_state.step)
    pbar = tqdm.tqdm(
        range(start_step, config.num_train_steps),
        initial=start_step,
        total=config.num_train_steps,
        dynamic_ncols=True,
    )

    action_infos = []
    subtask_infos = []

    # =========================== train loop ===========================
    for step in pbar:
        with sharding.set_mesh(mesh):
            # Use ratio-based scheduling for subtask steps
            should_do_subtask = (subtask_iter is not None and 
                                step % config.subtask_step_ratio == 0)
            
            if should_do_subtask:
                batch = next(subtask_iter)
                observation, _ = batch
                train_state, info = ptrain_step_subtask(train_rng, train_state, observation)
                subtask_info = {f"subtask/{k}": v for k, v in info.items()}
                subtask_infos.append(subtask_info)
            elif action_iter is not None:
                batch = next(action_iter)
                train_state, info = ptrain_step(train_rng, train_state, batch)
                action_info = {f"action/{k}": v for k, v in info.items()}
                action_infos.append(action_info)
            else:
                raise RuntimeError("No data available for training step")
        
    # =========================== logging ===========================
        if step % config.log_interval == 0:
            combined_info = {}
            info_parts = []
            
            if action_infos:
                stacked_action = common_utils.stack_forest(action_infos)
                reduced_action = jax.device_get(jax.tree.map(jnp.mean, stacked_action))
                combined_info.update(reduced_action)
                action_str = ", ".join(f"{k}={v:.4f}" for k, v in reduced_action.items())
                info_parts.append(action_str)
                
            if subtask_infos:
                stacked_subtask = common_utils.stack_forest(subtask_infos)
                reduced_subtask = jax.device_get(jax.tree.map(jnp.mean, stacked_subtask))
                combined_info.update(reduced_subtask)
                subtask_str = ", ".join(f"{k}={v:.4f}" for k, v in reduced_subtask.items())
                info_parts.append(subtask_str)
            
            if combined_info:
                info_str = ", ".join(info_parts)
                pbar.write(f"Step {step}: {info_str}")
                wandb.log(combined_info, step=step)
                
            action_infos = []
            subtask_infos = []

    # =========================== evaluation ===========================
        if step % config.eval_interval == 0 and step != start_step:
            reduced_eval = {}

            if action_eval_iter is not None:
                eval_infos = []
                num_eval_batches = len(action_eval_loader._data_loader.torch_loader)  # type: ignore[attr-defined]
                logging.info(f"[ACTION EVAL] Starting action evaluation with {num_eval_batches} batches")
                
                batch_idx = 0
                while True:
                    if num_eval_batches is not None and batch_idx >= num_eval_batches:
                        break
                    e_batch = next(action_eval_iter)

                    eval_info = peval_step(eval_rng, train_state, e_batch)
                    eval_infos.append(eval_info)
                    batch_idx += 1

                stacked_eval = common_utils.stack_forest(eval_infos)
                reduced_eval = jax.device_get(jax.tree.map(jnp.mean, stacked_eval))
            
            if subtask_eval_iter is not None:
                subtask_eval_infos = []
                num_subtask_eval_batches = len(subtask_eval_loader._data_loader.torch_loader)  # type: ignore[attr-defined]
                logging.info(f"[SUBTASK EVAL] Starting subtask evaluation with {num_subtask_eval_batches} batches")
                
                batch_idx = 0
                while True:
                    if num_subtask_eval_batches is not None and batch_idx >= num_subtask_eval_batches:
                        break
                    se_batch = next(subtask_eval_iter)

                    subtask_eval_info, predicted_tokens, target_mask = peval_step_subtask(eval_rng, train_state, se_batch)
                    subtask_eval_infos.append(subtask_eval_info)
                    
                    # Decode predictions from first batch for sanity checking (outside JIT)
                    if batch_idx == 0 and jax.process_index() == 0:
                        try:
                            decoded_predictions, decoded_targets = decode_subtask_predictions(se_batch, predicted_tokens, target_mask)
                            for i, (pred, target) in enumerate(zip(decoded_predictions, decoded_targets)):
                                print(f"------EVAL SUBTASK SAMPLE {i}:")
                                print(f"  PRED: {pred}")
                                print(f"  TARGET: {target}")
                        except Exception as e:
                            print(f"------EVAL SUBTASK DECODE ERROR: {e}")
                    
                    batch_idx += 1

                stacked_subtask_eval = common_utils.stack_forest(subtask_eval_infos)
                reduced_subtask_eval = jax.device_get(jax.tree.map(jnp.mean, stacked_subtask_eval))
                reduced_eval.update(reduced_subtask_eval)
            
            if reduced_eval:
                info_str = ", ".join(f"{k}={v:.4f}" for k, v in reduced_eval.items())
                pbar.write(f"[EVAL] Step {step}: {info_str}")
                wandb.log(reduced_eval, step=step)

        # =========================== checkpointing ===========================
        if (step % config.save_interval == 0 and step > start_step) or step == config.num_train_steps - 1:
            # Use action data loader for checkpointing if available, otherwise use subtask loader. action data loader actually
            # has norm stats, so we need to pass it in if we use the action data loader. subtask data loader doesn't have norm stats.
            checkpoint_data_loader = action_loader if action_loader is not None else subtask_loader
            _checkpoints.save_state(checkpoint_manager, train_state, checkpoint_data_loader, step)

    logging.info("Waiting for checkpoint manager to finish")
    checkpoint_manager.wait_until_finished()

if __name__ == "__main__":
    main(_config.cli())
