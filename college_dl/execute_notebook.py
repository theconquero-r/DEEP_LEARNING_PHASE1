
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
import os

notebook_path = '/home/conqueror/college dl/titanic_survival_prediction.ipynb'

# Check if the notebook exists
if not os.path.exists(notebook_path):
    print(f"Error: Notebook not found at {notebook_path}")
else:
    with open(notebook_path) as f:
        nb = nbformat.read(f, as_version=4)

    # Create an executor
    ep = ExecutePreprocessor(timeout=600, kernel_name='python3')

    try:
        # Execute the notebook
        ep.preprocess(nb, {'metadata': {'path': os.path.dirname(notebook_path)}})
        print(f"Notebook '{notebook_path}' executed successfully.")

        # Save the executed notebook
        with open(notebook_path, 'w', encoding='utf-8') as f:
            nbformat.write(nb, f)
        print(f"Executed notebook saved to '{notebook_path}'.")

    except Exception as e:
        print(f"Error executing notebook: {e}")
