default:
  just --list

run config:
    uv run python ./src/scripts/run_model.py {{config}}
