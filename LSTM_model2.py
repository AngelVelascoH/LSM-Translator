import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Define paths
output_folder = "output_data"

# Inicializar listas para almacenar datos y etiquetas
X_data = []  # Features
y_data = []  # Labels

# Mapear nombres de clases a números
class_to_label = {}
label_counter = 0

# Iterar sobre las carpetas de salida
for sign_folder in os.listdir(output_folder):
    sign_path = os.path.join(output_folder, sign_folder)
    if os.path.isdir(sign_path):
        # Añadir la clase al diccionario si no está presente
        if sign_folder not in class_to_label:
            class_to_label[sign_folder] = label_counter
            label_counter += 1

        # Obtener la etiqueta para esta clase
        label = class_to_label[sign_folder]

        # Iterar sobre los archivos en cada carpeta de salida
        for filename in os.listdir(sign_path):
            if filename.endswith("_keypoints.npy"):
                # Cargar el archivo de puntos clave
                keypoints_path = os.path.join(sign_path, filename)
                keypoints = np.load(keypoints_path)

                # Añadir los keypoints a las características (features)
                X_data.append(keypoints)

                # Añadir la etiqueta a las etiquetas (labels)
                y_data.append(label)

# Convertir las listas a matrices numpy
X_data = np.array(X_data)
y_data = np.array(y_data)

# Dividir los datos en conjuntos de entrenamiento y prueba
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.2, random_state=42)



# Crear el modelo LSTM
model = Sequential()
model.add(LSTM(50, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dense(1, activation='sigmoid'))

# Resto del código...

# Normalizar los datos (opcional, dependiendo del modelo)
# Puedes usar MinMaxScaler, StandardScaler, etc.



# Compilar el modelo
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])  # Cambiar 'binary_crossentropy' según el número de clases

# Entrenar el modelo
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# Evaluar el modelo
accuracy = model.evaluate(X_test, y_test)[1]
print(f'Accuracy: {accuracy * 100:.2f}%')
