# Code Teacher Model

This project is a code-teaching AI model that answers programming questions and generates code, similar to GitHub Copilot. It uses HuggingFace Transformers and is trained on a custom Q&A dataset.

## Setup Instructions

### 1. Clone the Repository

Clone or download the project to your local machine.

### 2. Create and Activate a Python Virtual Environment

```bash
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On macOS/Linux:
source .venv/bin/activate
```

### 3. Install Python and Dependencies

```bash
pip install -r code-teacher-model/requirements.txt
```

### 4. Prepare the Dataset

- Place your Q&A dataset in `code-teacher-model/data/`.
- Supported formats: JSON, CSV, or TXT (see `utils/preprocess.py` for details).
- Example command to preprocess:

  ```bash
  python code-teacher-model/utils/preprocess.py --input data/your_dataset.json --output data/processed_dataset.json
  ```

### 5. Model Files

- Pretrained or checkpointed model files should be placed in `code-teacher-model/model/checkpoints/`.
- If starting from scratch, training will create this directory automatically.

### 6. Training the Model

```bash
python code-teacher-model/model/train.py
```

### 7. Inference (Ask Questions)

- Use `code-teacher-model/model/interface.py` to interact with the model:

```bash
python code-teacher-model/model/interface.py
```

### 8. Configuration

- Edit `code-teacher-model/config/config.yaml` for model and training settings.

### 9. Utilities

- Data preprocessing scripts are in `code-teacher-model/utils/preprocess.py`.
- You can integrate more public code datasets for advanced capabilities.
- If you encounter issues, check Python version and package compatibility.

```
Q: How do I print something in python?
A: Use the print function. Example:
print('Hello, World!')
```

## Contact

For questions or contributions, open an issue or pull request on GitHub.
