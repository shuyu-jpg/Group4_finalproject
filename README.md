## How to run

```bash
# Create virtual environment and activate
virtualenv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
pip install -r requirements.txt --no-deps
```

```bash
# Launch Jupyter Notebook for data exploration and model training
jupyter notebook \
    --notebook-dir="." \
    --ip=0.0.0.0 --port=3225
```

```bash
# Optional: Debug a specific script (e.g., test.py)
python -m debugpy --listen 4444 test.py
```

```bash
# Start FastAPI app with model registry and dashboard
uvicorn main:app --reload --port 8000
```

## Enable push to git

```bash
nano ~/.gitconfig
```

```bash
code ~/.gitconfig
```

