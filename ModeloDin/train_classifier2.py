import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib  # Para guardar el modelo entrenado

# Ruta a la carpeta con los datos de entrenamiento
data_folder = "output_data"

# Lista para almacenar las secuencias y sus respectivas etiquetas
sequences = []
labels = []

# Iterar sobre las carpetas de señas
for sign_folder in os.listdir(data_folder):
    sign_path = os.path.join(data_folder, sign_folder)
    if os.path.isdir(sign_path):
        # Iterar sobre los archivos de secuencias en la carpeta de la señal
        for sequence_file in os.listdir(sign_path):
            if sequence_file.endswith("_keypoints.npy"):
                # Cargar la secuencia desde el archivo
                sequence_path = os.path.join(sign_path, sequence_file)
                sequence = np.load(sequence_path)

                # Aplanar la secuencia para asegurarse de que tenga forma bidimensional
                sequence = sequence.flatten()

                # Agregar la secuencia y su etiqueta a las listas
                sequences.append(sequence)
                labels.append(sign_folder)
# Obtener la longitud máxima de las secuencias
max_sequence_length = max(len(seq) for seq in sequences)

# Aplanar y rellenar/truncar cada secuencia para que tenga la misma longitud
for i, sequence in enumerate(sequences):
    flattened_sequence = sequence.flatten()
    padded_sequence = np.pad(flattened_sequence, (0, max_sequence_length - len(flattened_sequence)))
    sequences[i] = padded_sequence
# Convertir las listas a arrays de numpy
X = np.array(sequences)
y = np.array(labels)

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Inicializar el clasificador RandomForest
clf = RandomForestClassifier(n_estimators=100, random_state=42)

# Entrenar el clasificador
clf.fit(X_train, y_train)

# Predecir en los datos de prueba
y_pred = clf.predict(X_test)

# Calcular la precisión del modelo
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')

# Guardar el modelo entrenado para su uso futuro
model_filename = "sign_language_rf_model3.joblib"
joblib.dump(clf, model_filename)
print(f'Model saved as {model_filename}')
