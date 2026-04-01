default:
  just --list

run config:
    uv run python ./src/scripts/run_model.py {{config}}

tensorboard log_dir:
    uv run tensorboard --logdir {{log_dir}}

plot log_data:
    uv run python ./src/scripts/plot_from_training_data.py {{log_data}}
