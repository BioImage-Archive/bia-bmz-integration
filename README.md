Create conda env with cuda:

```bash
conda env create -f environment.yml -p ./env
conda activate ./env
```

Create venv without cuda for dev:

```bash 
poetry shell
poetry install
```
