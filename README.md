# AgentRisk · SemEval 2027

This experiment evaluates how Qwen3-0.6B rates the severity, harm domains and risk drivers of a proposed agent action. The repository includes the prompts, dataset splits, saved predictions and scoring code.

## Check the saved results

From this folder, run the following command with Python 3.9 or later. It verifies the saved request hashes and recalculates the metrics using the Python standard library.

```sh
python main.py check
```

Use `python main.py export` to print the scores as CSV or `python main.py runs` to list the experiment presets.

## Run the model on Windows

The bundled local setup targets Windows x64 with an NVIDIA GPU and a compatible driver. Model weights and llama.cpp binaries live under `model/` and `runtime/llama/`. Git ignores those large downloads. After cloning the repository, restore them once from this folder with the following command. The download is about 1.3 GB.

```sh
python setup_model.py
```

If the weights and runtime are already present, you can run the model directly. For example, this command runs a full split and writes the predictions to `results.json` under a new ID. Choose a fresh ID for each run you want to keep.

```sh
python main.py run direct_0 --save-as direct_0_repeat
```

Use `python main.py run --help` for single-case and runtime options. On Windows, `run.cmd check` is a shortcut for `python main.py check`.

## Files and sources

| Path | Contents |
| --- | --- |
| `main.py` | Inference, scoring, export and verification |
| `agentrisk.json` | Cases, evaluation splits, prompts and model settings |
| `results.json` | Saved predictions, request hashes and summary metrics |
| `data/` | Original training and development data and split manifest |
| `model/LICENSE` | Upstream model license |

[AgentRisk dataset](https://huggingface.co/datasets/AgentRisk/agentrisk-data-2027) · [Qwen3-0.6B GGUF](https://huggingface.co/Qwen/Qwen3-0.6B-GGUF) · [llama.cpp build b12110](https://github.com/ggml-org/llama.cpp/releases/tag/b12110)
