# M-VSWiM

A virtual solar wind monitor for Mercury using Gaussian Process Regression. Inspired by [vSWIM](https://github.com/abbyazari/vSWIM).

We utilise many spacecraft which sampled solar wind data within the vicinity of Mercury's orbit.
- Solar Orbiter
- Parker Solar Probe
- Helios 1 & 2

These spacecraft offer a relatively long-term and continuous measurement of the solar wind, within which we can quantify the performance of a given model architecture, before applying it in practice.

> [!TIP]
> For portability, we **strongly** recommend and use [uv](https://docs.astral.sh/uv/) to manage dependencies and versions. If you do not wish to use uv, dependencies can be found within [pyproject.toml](./pyproject.toml) and a `requirements.txt` file can be made with `pip compile pyproject.toml -o requirements.txt`.

## Fit the model to data
<details>
<summary>Using <a href='https://github.com/astral-sh/uv'>uv</a></summary>

```shell
# uv run python src/scripts/run_model.py <config>

e.g.
uv run python src/scripts/run_model.py src/scripts/config/example-config.py
```
</details>
<details>
<summary>Using <a href='https://github.com/casey/just'>just</a></summary>

Requires <a href='https://github.com/astral-sh/uv'>uv</a>

```shell
# just train <config>

e.g.
just run src/scripts/config/example-config.py
```
</details>
<details>
<summary>Manual environment</a></summary>

NOT RECOMMENDED FOR REPRODUCIBILITY

If you would rather, you may manually create an Python environment (i.e. via venv, pyenv, cond, or similar). See [pyproject.toml](./pyproject.toml) for dependancy details.

```shell
# uv run python src/scripts/run_model.py <config>

e.g.
uv run python src/scripts/run_model.py src/scripts/config/example-config.py
```
</details>
