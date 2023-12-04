import cv2
import numpy as np
import joblib
import mediapipe as mp
import tkinter as tk
from tkinter import scrolledtext, font
from PIL import Image, ImageTk

# Cargar el modelo entrenado
model_filename = "sign_language_rf_model3.joblib"
clf = joblib.load(model_filename)

# Inicializar mediapipe hands module
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

# Inicializar OpenCV VideoCapture
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # Usa 0 para la cámara predeterminada, puedes ajustar esto según tu configuración

# Definir la longitud de la secuencia
max_sequence_length = 60

# Lista para almacenar las secuencias de keypoints en tiempo real
realtime_sequences = []

# Inicializar la palabra actual y la última detectada
current_word = ""
last_detected_word = ""

# Función para actualizar el área de texto
def update_text_area():
    global current_word
    global last_detected_word

    if current_word != "" and current_word != last_detected_word:
        last_detected_word = current_word
        text_area.insert(tk.END, f"{last_detected_word}\n")
        text_area.yview(tk.END)

# Función principal para el procesamiento de video
def process_video():
    global current_word
    global realtime_sequences
    global last_detected_word

    ret, frame = cap.read()
    if not ret:
        return

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
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(frame)
    img = ImageTk.PhotoImage(img)
    video_canvas.create_image(0, 0, anchor=tk.NW, image=img)
    video_canvas.img = img  # Para evitar que la imagen se borre por el recolector de basura

    # Actualizar la palabra detectada si cambia
    update_text_area()

    # Llamar a la función nuevamente después de 1 ms
    root.after(1, process_video)

# Configurar la interfaz gráfica
root = tk.Tk()
root.title("Real-time Sign Language Prediction")

# Frame de video
video_canvas = tk.Canvas(root, width=640, height=480)
video_canvas.pack(side=tk.LEFT)

# Área de texto para mostrar las palabras detectadas
text_area = scrolledtext.ScrolledText(root, wrap=tk.WORD, width=20, height=10, font=("Helvetica", 28))
text_area.pack(side=tk.RIGHT)

# Llamar a la función de procesamiento de video al inicio
process_video()

# Salir si se cierra la ventana
root.protocol("WM_DELETE_WINDOW", lambda: [root.destroy(), cap.release(), cv2.destroyAllWindows()])

# Iniciar el bucle principal de la interfaz gráfica
root.mainloop()
