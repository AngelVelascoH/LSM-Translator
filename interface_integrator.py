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

top_word_choices = []

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.3)
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
    14: "Ñ",
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
max_sequence_length = 30

model_filename = "sign_language_rf_model5.joblib"
clf = joblib.load(model_filename)

# Crear una imagen en negro
blank_image = np.zeros((430, 590, 3), np.uint8)


def select_word(word):
    x_center = letter_canvas.winfo_width() / 2
    y_center = letter_canvas.winfo_height() / 2
    global detected_string,last_detected_word
    if word != "" and word != last_detected_word:
        if word == "banio":
            word = "baño"
        letter_canvas.delete("all")
        last_detected_word = word
        detected_string += word + " "
        text_area.delete("1.0", tk.END)
        text_area.insert(tk.END, detected_string)
        text_area.tag_add("big", "1.0", tk.END)
        letter_canvas.create_text(
                    x_center, y_center, text=word, font=("Arial", 30), fill="white"
                )


def update_top_word_choices_ui():
    global top_word_choices, realtime_sequences

    # Limpiar el frame de los botones anteriores
    for widget in top_words_frame.winfo_children():
        widget.destroy()

    # Crear un botón para cada palabra en top_word_choices
    for word, probability in top_word_choices:
        word_button = ttk.Button(top_words_frame, text=f"{word} ({probability * 100:.2f}%)", 
                                 command=lambda w=word: select_word(w))
        word_button.pack(pady=5)  # Añade un pequeño espacio vertical entre los botones
    realtime_sequences = []

def read_text():
    global detected_string
    engine.say(detected_string)
    engine.runAndWait()


def switch():
    global spelling
    if spelling == False:
        spelling = True
    else:
        spelling = False
    


def start_camera():
    global cap
    global reading
    global spelling
    if reading:
        pass
    else:
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        reading = True
        capture_video_combined()


def stop_camera():
    global cap
    global reading

    reading = False
    video_label.config(image=blank_image)  # Mostrar imagen en negro
    letter_canvas.delete("all")
    cap.release()
    cv2.destroyAllWindows()


def clear_text():
    global detected_string, current_time, current_letter
    detected_string = ""
    current_letter = None
    current_time = 0
    letter_canvas.delete("all")
    text_area.delete("1.0", tk.END)

def delete_letter():
    global detected_string
    if len(detected_string) > 0:
        detected_string = detected_string[:-1]
        update_text_area()

def delete_word():
    global detected_string
    words = detected_string.split()
    if len(words) > 0:
        detected_string = " ".join(words[:-1])
        detected_string = detected_string + " "
        update_text_area()

def update_text_area():
    text_area.delete("1.0", tk.END)
    text_area.insert(tk.END, detected_string)
    text_area.tag_add("big", "1.0", tk.END)


def capture_video_combined():
    global current_letter, current_time, detected_string, current_word, last_detected_word, realtime_sequences
    x_center = letter_canvas.winfo_width() / 2
    y_center = letter_canvas.winfo_height() / 2

    if reading:
        ret, frame = cap.read()
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        if spelling:
            handle_spelling(results)
        else:
            handle_sign_language(results)

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame)
        img = ImageTk.PhotoImage(image=img)
        video_label.img = img
        video_label.config(image=img)

        video_label.after(10, capture_video_combined)


def handle_spelling(results):
    global current_letter, current_time, detected_string
    predicted_char = ""
    x_center = letter_canvas.winfo_width() / 2
    y_center = letter_canvas.winfo_height() / 2

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
                   


                    predicted_char = predicted_letter + "\n" + str(confidence) + "%"
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
            x_center, y_center, text=predicted_char, font=("Arial", 30), fill="white"
            )

def handle_sign_language(results):
    global current_word, realtime_sequences, detected_string,last_detected_word, top_word_choices
    x_center = letter_canvas.winfo_width() / 2
    y_center = letter_canvas.winfo_height() / 2

    if results.multi_handedness:
        print(f"realtime_sequences: {len(realtime_sequences)}")
      

        if results.multi_hand_landmarks:
            hand_keypoints = results.multi_hand_landmarks[0].landmark
            keypoints_array = np.array(
                [[point.x, point.y, point.z] for point in hand_keypoints]
            ).flatten()

            # Añadir keypoints a la lista de la secuencia
            realtime_sequences.append(keypoints_array)

            # Cuando alcanzamos la longitud máxima de la secuencia, hacemos una predicción
            if len(realtime_sequences) == max_sequence_length:
                # Convertir la secuencia a un array de numpy
                input_sequence = np.array(realtime_sequences)
                probabilities = clf.predict_proba(input_sequence.reshape(1, -1))[0]
                top_indices = np.argsort(probabilities)[-3:][::-1]
                top_word_choices = [(clf.classes_[i], probabilities[i]) for i in top_indices]
                if probabilities[top_indices[0]] > 0.4:
                    current_word = clf.classes_[top_indices[0]]
                    select_word(current_word)
                    realtime_sequences = []
                else:
                    update_top_word_choices_ui()

                # Hacer la predicción
                """
                predicted_label = clf.predict(input_sequence.reshape(1, -1))[0]

                # Actualizar la palabra actual
                current_word = predicted_label

                # Mostrar la predicción en la parte inferior de la pantalla
            

                # Limpiar la lista de la secuencia para la siguiente iteración
                realtime_sequences = []
            if current_word != "" and current_word != last_detected_word:
                last_detected_word = current_word
                detected_string += current_word + " "
                letter_canvas.delete("all")

                text_area.delete("1.0", tk.END)
                text_area.insert(tk.END, detected_string)
                text_area.tag_add("big", "1.0", tk.END)
                letter_canvas.create_text(
                    x_center, y_center, text=current_word, font=("Arial", 30), fill="white"
                )
               """ 

    else:
        realtime_sequences = []
        print("No hand landmarks detected")


root = tk.Tk()
root.geometry("1280x720")

root.title("LSM ESCOM TT2022")

# Crear un frame para dividir la ventana en dos partes
main_frame = Frame(root)
main_frame.pack(fill=tk.BOTH, expand=True)

# Dividir el frame en dos columnas
left_frame = Frame(main_frame, width=640, height=720)
left_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")

right_frame = Frame(main_frame, width=640, height=720)
right_frame.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")

main_frame.grid_rowconfigure(0, weight=1)
main_frame.grid_columnconfigure(0, weight=1)
main_frame.grid_columnconfigure(1, weight=1)

# Canvas para mostrar un rectángulo sutil (75%)
letter_canvas = Canvas(right_frame, bg="black")
letter_canvas.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

button_text_frame = Frame(right_frame)
button_text_frame.pack(fill=tk.BOTH, expand=True)

# Botones
start_button = ttk.Button(button_text_frame, text="Iniciar Cámara", command=start_camera, style="Accent.TButton")
start_button.grid(row=0, column=0, padx=20, pady=10, sticky="nsew")

stop_button = ttk.Button(button_text_frame, text="Detener Cámara", command=stop_camera)
stop_button.grid(row=1, column=0, padx=20, pady=10, sticky="nsew")

clear_button = ttk.Button(button_text_frame, text="Limpiar", command=clear_text, style="Accent.TButton")
clear_button.grid(row=2, column=0, padx=20, pady=10, sticky="nsew")

play_button = ttk.Button(button_text_frame, text="Reproducir", command=read_text, style="Accent.TButton")
play_button.grid(row=3, column=0, padx=20, pady=10, sticky="nsew")

switch_button = ttk.Button(button_text_frame, text="Cambiar modo", command=switch, style="Accent.TButton")
switch_button.grid(row=4, column=0, padx=20, pady=10, sticky="nsew")

delete_letter_button = ttk.Button(button_text_frame, text="Borrar letra", command=delete_letter, style="Accent.TButton")
delete_letter_button.grid(row=5, column=0, padx=20, pady=10, sticky="nsew")

delete_word_button = ttk.Button(button_text_frame, text="Borrar palabra", command=delete_word, style="Accent.TButton")
delete_word_button.grid(row=6, column=0, padx=20, pady=10, sticky="nsew")


top_words_frame = Frame(left_frame)
top_words_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)


# Text area
text_area = tk.Text(button_text_frame, width=30, height=10, font=("Arial", 16))
text_area.grid(row=0, column=1, rowspan=5, padx=10, pady=10, sticky="nsew")

text_area.tag_configure("big", font=("Arial", 18, "bold"))

# ...




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
