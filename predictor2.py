import cv2
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import joblib
import mediapipe as mp

# Cargar el modelo entrenado
model_filename = "sign_language_rf_model.joblib"
clf = joblib.load(model_filename)

# Inicializar mediapipe hands module
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

# Inicializar OpenCV VideoCapture
cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)  # Usa 0 para la cámara predeterminada, puedes ajustar esto según tu configuración

# Definir la longitud de la secuencia
max_sequence_length = 60

# Lista para almacenar las secuencias de keypoints en tiempo real
realtime_sequences = []

# Inicializar la palabra actual y la última detectada
current_word = ""
last_detected_word = ""

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convertir el frame a RGB para mediapipe
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detectar keypoints en la mano
    results = hands.process(frame_rgb)
    if results.multi_hand_landmarks:
        hand_keypoints = results.multi_hand_landmarks[0].landmark
        keypoints_array = np.array([[point.x, point.y, point.z] for point in hand_keypoints]).flatten()

        # Añadir keypoints a la lista de la secuencia
        realtime_sequences.append(keypoints_array)

        # Cuando alcanzamos la longitud máxima de la secuencia, hacemos una predicción
        if len(realtime_sequences) == max_sequence_length:
            # Convertir la secuencia a un array de numpy
            input_sequence = np.array(realtime_sequences)

            # Hacer la predicción
            predicted_label = clf.predict(input_sequence.reshape(1, -1))[0]

            # Actualizar la palabra actual
            current_word = predicted_label

            # Mostrar la predicción en la parte inferior de la pantalla
            cv2.putText(frame, f"Current Word: {current_word}", (10, frame.shape[0] - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Limpiar la lista de la secuencia para la siguiente iteración
            realtime_sequences = []

    # Mostrar el frame
    cv2.imshow("Real-time Prediction", frame)

    # Actualizar la palabra detectada si cambia
    if current_word != last_detected_word:
        last_detected_word = current_word

    # Salir si se presiona la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar recursos
cap.release()
cv2.destroyAllWindows()