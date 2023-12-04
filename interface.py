import cv2
import mediapipe as mp
import pickle
import numpy as np
import tkinter as tk
from tkinter import Label, Button, Frame, Canvas
from tkinter import ttk
from PIL import Image, ImageTk
import sv_ttk
import pyttsx3
import joblib


model_dict = pickle.load(open("./model.p", "rb"))
model = model_dict["model"]
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.6)
engine = pyttsx3.init()
labels_dict = {
    0: "A",
    1: "B",
    2: "C",
    3: "D",
    4: "E",
    5: "F",
    6: "G",
    7: "H",
    8: "I",
    9: "J",
    10: "K",
    11: "L",
    12: "M",
    13: "N",
    14: "ENIE",
    15: "O",
    16: "P",
    17: "Q",
    18: "R",
    19: "S",
    20: "T",
    21: "U",
    22: "V",
    23: "W",
    24: "X",
    25: "Y",
    26: "Z",
}

global reading 
reading = False

current_letter = None
current_time = 0
detected_string = ""
spelling = False

realtime_sequences = []

# Inicializar la palabra actual y la última detectada
current_word = ""
last_detected_word = ""
max_sequence_length = 60

model_filename = "sign_language_rf_model2.joblib"
clf = joblib.load(model_filename)



# Crear una imagen en negro
blank_image = np.zeros((480, 640, 3), np.uint8)

def read_text():
    global detected_string
    engine.say(detected_string)
    engine.runAndWait()

def switch():
    global spelling
    if(spelling == False):
        spelling = True
    else:
        spelling = False
    stop_camera()
    start_camera()

def start_camera():
    global cap
    global reading
    global spelling
    if(reading):
        pass
    else:
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        reading = True
        
        if(spelling):
            capture_video()
        else:
            capture_video2()

def stop_camera():
    global cap
    global reading
    
    reading = False
    video_label.config(image=blank_image)  # Mostrar imagen en negro
    letter_canvas.delete("all")
    cap.release()
    cv2.destroyAllWindows()

def clear_text():
    global detected_string,current_time,current_letter
    detected_string = ""
    current_letter = None
    current_time = 0
    letter_canvas.delete("all")
    text_area.delete("1.0",tk.END)

def capture_video():
    global current_letter,current_time,detected_string
    predictedChar = ""
    x_center = letter_canvas.winfo_width() / 2

    y_center = letter_canvas.winfo_height() / 2
    
    if reading:
        ret, frame = cap.read()
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            

            for idx, hand_handedness in enumerate(results.multi_handedness):
                label = hand_handedness.classification[0].label
                if label == 'Right' and len(results.multi_handedness) != 2:
                    # Ajustar landmarks para simular efecto espejo
                    for hand_landmarks in results.multi_hand_landmarks:
                        for landmark in hand_landmarks.landmark:
                            landmark.x = 1 - landmark.x

        if results.multi_hand_landmarks:
            
            data_aux = []
            x_ = []
            y_ = []

            for hand_landmarks in results.multi_hand_landmarks:
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    x_.append(x)
                    y_.append(y)

                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    data_aux.append(x - min(x_))
                    data_aux.append(y - min(y_))

           
            if len(results.multi_handedness) != 2 and len(results.multi_hand_landmarks) == 1:
                letter_canvas.delete("all")
                prediction = model.predict([np.asarray(data_aux)])

                confidence = model.predict_proba([np.asarray(data_aux)])
                confidence = confidence.max()
                print(confidence)
                if confidence > 0.4:
                    predicted_letter = labels_dict[int(prediction[0])]
                    if predicted_letter == current_letter:
                        current_time += 1
                    else:
                        current_letter = predicted_letter
                        current_time = 1

                    if current_time > 5:  # 30 frames son aproximadamente 3 segundos
                        detected_string += predicted_letter
                        current_time = 0
                        text_area.delete("1.0", tk.END)
                        text_area.insert(tk.END, detected_string)
                        text_area.tag_add("big", "1.0", "end")
                        
                    
                    # crear un espacio si no se detecta mano por 3 segundos
                   


                    predictedChar = predicted_letter + "\n" + str(confidence) + "%"
            else:
                current_time += 1 
                if(current_time > 20):
                    print("Espacio.---")
                    current_time = 0

                    current_letter = None
                    space = "Espacio..."
                    letter_canvas.delete("all")
                    letter_canvas.create_text(
                    x_center, y_center, text=space, font=("Arial", 30), fill="white"
                    )
                    detected_string += " "
                    text_area.delete("1.0", tk.END)
                    text_area.insert(tk.END, detected_string)
                    text_area.tag_add("big", "1.0", tk.END)

            letter_canvas.create_text(
            x_center, y_center, text=predictedChar, font=("Arial", 30), fill="white"
            )

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame)
        img = ImageTk.PhotoImage(image=img)
        video_label.img = img
        video_label.config(image=img)
        video_label.after(10, capture_video)

def capture_video2():
    x_center = letter_canvas.winfo_width() / 2

    y_center = letter_canvas.winfo_height() / 2
    global current_letter,realtime_sequences,last_detected_word,detected_string
    if reading:
        ret, frame = cap.read()
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
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
           
           
            detected_string += current_word + " "
            text_area.delete("1.0", tk.END)
            text_area.insert(tk.END, detected_string)
            text_area.tag_add("big", "1.0", tk.END)
            letter_canvas.create_text(
                x_center, y_center, text=current_word, font=("Arial", 30), fill="white"
            )
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame)
            img = ImageTk.PhotoImage(image=img)
            video_label.img = img
            video_label.config(image=img)

            video_label.after(10, capture_video2)

        

root = tk.Tk()
root.title("LSM ESCOM TT2022")

# Crear un frame para dividir la ventana en dos partes
main_frame = Frame(root)
main_frame.pack(fill=tk.BOTH, expand=True)

# Dividir el frame en dos columnas
left_frame = Frame(main_frame, width=320, height=480)
left_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")

right_frame = Frame(main_frame, width=320, height=480)
right_frame.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")

main_frame.grid_rowconfigure(0, weight=1)
main_frame.grid_columnconfigure(0, weight=1)
main_frame.grid_columnconfigure(1, weight=1)

# Canvas para mostrar un rectángulo sutil (75%)
letter_canvas = Canvas(right_frame, bg="black")
letter_canvas.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

# Crear un frame para los botones (25%)
button_frame = Frame(right_frame)
button_frame.pack(fill=tk.BOTH, expand=True)

# Botones
start_button = ttk.Button(button_frame, text="Start Camera", command=start_camera,style="Accent.TButton")
start_button.grid(row=0, column=0, padx=20, pady=10, sticky="nsew")

stop_button = ttk.Button(button_frame, text="Stop Camera", command=stop_camera)
stop_button.grid(row=1, column=0, padx=20, pady=10, sticky="nsew")

clear_button = ttk.Button(button_frame, text="Clear", command=clear_text,style="Accent.TButton")
clear_button.grid(row=2, column=0, padx=20, pady=10, sticky="nsew")
# Crear un frame para el text area (25%)
text_frame = Frame(right_frame)
text_frame.pack(fill=tk.BOTH, expand=True)

# Botón para reproducir texto
play_button = ttk.Button(button_frame, text="Reproducir", command=read_text, style="Accent.TButton")
play_button.grid(row=1, column=1, padx=20, pady=10, sticky="nsew")

switch_button = ttk.Button(button_frame, text="Switch", command=switch,style="Accent.TButton")
switch_button.grid(row=0, column=1, padx=20, pady=10, sticky="nsew")

# Text area
text_area = tk.Text(text_frame, height=10)
text_area.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

text_area.tag_configure("big", font=("Arial", 20, "bold"))

# Etiqueta para mostrar el video de la cámara (50%)
video_label = Label(left_frame)
video_label.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

# Mostrar la imagen en negro cuando se inicia la aplicación
blank_image = Image.fromarray(blank_image)
blank_image = ImageTk.PhotoImage(image=blank_image)
video_label.img = blank_image
video_label.config(image=blank_image)

cap = None
sv_ttk.set_theme("dark")
global x_center 


root.mainloop()
