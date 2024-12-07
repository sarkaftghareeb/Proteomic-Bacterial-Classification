import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Set random seed for reproducibility
np.random.seed(42)

# Parameters for synthetic data generation
num_samples = 500  # Number of samples (bacterial strains)
num_features = 128  # Number of m/z features per sample
m_z_range = (3, 30)  # m/z range from 3 kDa to 30 kDa

# Generate synthetic mass-to-charge ratios (m/z values)
m_z_values = np.linspace(m_z_range[0], m_z_range[1], num_features)

# Simulate intensity values (random values for simplicity)
def generate_intensity():
    # Generate intensity values based on a normal distribution
    return np.random.normal(loc=1000, scale=500, size=num_features)

# Create synthetic dataset
data = []
labels = []

for _ in range(num_samples):
    # Simulate bacterial species
    if np.random.rand() > 0.7:  # 30% chance for N. meningitidis (class 0)
        labels.append(0)
        intensity = generate_intensity() + np.random.normal(200, 100, size=num_features)  # Simulate a distinct feature
    elif np.random.rand() > 0.5:  # 20% chance for N. gonorrhoeae (class 1)
        labels.append(1)
        intensity = generate_intensity() + np.random.normal(100, 50, size=num_features)  # Slightly different feature
    else:  # 50% chance for other species (class 2)
        labels.append(2)
        intensity = generate_intensity()  # Random intensity for "other" species
    
    data.append(intensity)

# Convert to a DataFrame (only after making sure data is populated)
if len(data) > 0:
    df = pd.DataFrame(data, columns=[f"m/z_{i}" for i in range(num_features)])
    df['class'] = labels  # Add class labels to the dataframe
else:
    print("Error: Data list is empty!")

# Show first few rows of the synthetic dataset
print(df.head())
########### bacteria classificatio


from sklearn.preprocessing import MinMaxScaler
import numpy as np

# Initialize the MinMaxScaler
scaler = MinMaxScaler()

# Normalize the features (m/z values and intensities)
X = df.drop(columns=['class']).values  # Features (m/z values)
y = df['class'].values  # Labels (bacterial species)

# Normalize data between 0 and 1
X_scaled = scaler.fit_transform(X)

# Split data into training, validation, and test sets (80%, 10%, 10%)
from sklearn.model_selection import train_test_split
X_train, X_temp, y_train, y_temp = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

#####Artificial Neural Network (ANN) Setup and Training:
    
pip install tensorflow

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

# Build the MLP ANN model
model = Sequential()
model.add(Dense(64, input_dim=X_train.shape[1], activation='relu'))  # First hidden layer (64 neurons)
model.add(Dense(32, activation='relu'))  # Second hidden layer (32 neurons)
model.add(Dense(3, activation='softmax'))  # Output layer (3 classes: N. meningitidis, N. gonorrhoeae, Other)

# Compile the model
model.compile(optimizer=Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=50, batch_size=32, verbose=1)
####Dimensionality Reduction and Feature Selection:

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Apply PCA for dimensionality reduction (2 components for visualization)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Visualize the PCA results
plt.figure(figsize=(8, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis', label='Samples')
plt.colorbar(label='Class')
plt.title('PCA of Proteomic Data')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.show()

# Extract weights from the trained model for feature selection
weights = model.get_weights()[0]  # Weights from the first hidden layer
importance = np.abs(weights).sum(axis=1)  # Sum the absolute values of weights for each feature
top_features = np.argsort(importance)[-30:]  # Select top 30 features based on their importance
#######Model Validation:
from sklearn.metrics import accuracy_score, roc_curve, auc

# Predict on the test set
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)  # Get the class with the highest probability

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred_classes)
print(f"Accuracy on Test Set: {accuracy * 100:.2f}%")

# ROC Curve for multi-class classification
fpr, tpr, thresholds = roc_curve(y_test, y_pred[:, 0], pos_label=0)  # For class 0 (N. meningitidis)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='blue', label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()
#######

!git status
!git push
