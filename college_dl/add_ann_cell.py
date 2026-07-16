
import nbformat
from nbformat.v4 import new_notebook, new_code_cell
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from sklearn.metrics import accuracy_score

# Define the path to the existing notebook and the dataset
notebook_path = '/home/conqueror/college dl/titanic_survival_prediction.ipynb'
data_path = '/home/conqueror/college dl/TitanicSurvivalDataNumeric.pkl'

# --- Code for the new cell ---
new_cell_code = """
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from sklearn.metrics import accuracy_score

# Load the dataset
try:
    data = pd.read_pickle('/home/conqueror/college dl/TitanicSurvivalDataNumeric.pkl')
    print("Dataset loaded successfully.")
except FileNotFoundError:
    print("Error: TitanicSurvivalDataNumeric.pkl not found. Please ensure the file is in the correct directory.")
    exit()

# Separate features (X) and target (y)
X = data.drop('Survived', axis=1)
y = data['Survived']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define the ANN architecture
model = Sequential([
    Input(shape=(X_train_scaled.shape[1],)),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid') # Output layer for binary classification
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
print("Training the ANN model...")
history = model.fit(X_train_scaled, y_train, epochs=50, batch_size=32, verbose=0)
print("Model training complete.")

# Evaluate the model
loss, accuracy = model.evaluate(X_test_scaled, y_test, verbose=0)
print(f"Test Accuracy: {accuracy:.4f}")

# Make predictions (optional, for demonstration)
y_pred_proba = model.predict(X_test_scaled)
y_pred = (y_pred_proba > 0.5).astype(int)
print("Sample predictions (first 5):")
print(y_pred[:5].flatten())
"""

# Read the existing notebook
try:
    with open(notebook_path, 'r', encoding='utf-8') as f:
        notebook = nbformat.read(f, as_version=4)
except FileNotFoundError:
    print(f"Notebook not found at {notebook_path}. Creating a new one.")
    notebook = new_notebook()
except Exception as e:
    print(f"Error reading notebook: {e}. Creating a new one.")
    notebook = new_notebook()

# Create a new code cell
new_cell = new_code_cell(new_cell_code)

# Add the new cell to the notebook
notebook.cells.append(new_cell)

# Write the modified notebook back to the file
try:
    with open(notebook_path, 'w', encoding='utf-8') as f:
        nbformat.write(notebook, f)
    print(f"Successfully added ANN cell to {notebook_path}")
except Exception as e:
    print(f"Error writing notebook: {e}")
