# Logit Lens Visualizer

Implements the [logit lens](https://www.lesswrong.com/posts/AcKRB8wDpdaN6v6ru/interpreting-gpt-the-logit-lens) 
technique for inspecting how a transformer builds up its predictions layer by layer.

Runs on [Pythia-14M](https://huggingface.co/EleutherAI/pythia-14m) via TransformerLens.

## Setup

Requires [uv](https://docs.astral.sh/uv/).

git clone "to add"
cd "to add"
uv sync
uv run jupyter notebook

## Usage

Open `logit_lens.ipynb` and run all cells. 
You can type any prompt into the interactive widget to visualize predictions at each layer.

## What it shows

Each cell in the heatmap shows the top predicted token at a given layer and position — 
you can watch the model "converge" on its final prediction as depth increases.
```

---

**`.gitignore`**

Make sure you have one — at minimum:
```
.venv/
__pycache__/
*.pyc
.ipynb_checkpoints/