from tkinter import *
from PIL import Image, ImageTk
import tensorflow as tf                                   
from tensorflow.keras.models import load_model           
import google.generativeai as genai
import os

spider_classes = [
    "Black Widow",
    "Blue Tarantula",
    "Bold Jumper",
    "Brown Grass Spider",
    "Brown Recluse Spider",
    "Deinopis Spider",
    "Golden Orb Weaver",
    "Hobo Spider",
    "Huntsman Spider",
    "Ladybird Mimic Spider",
    "Peacock Spider",
    "Red Knee Tarantula",
    "Spiny-backed Orb-weaver",
    "White Kneed Tarantula",
    "Yellow Garden Spider"
]


def interface(spider, description, image_path):
    root = Tk()  # create root window
    root.title("Basic GUI Layout")  # title of the GUI window
    root.maxsize(900, 600)  # specify the max size the window can expand to
    root.config(bg="skyblue")  # specify background color

    # Create Label in our window
    imagen = Image.open(image_path)  # open image
    imagen = imagen.resize((200, 200), Image.LANCZOS)  # resize image

    # Create left and right frames
    left_frame = Frame(root, width=400, height=400, bg='grey')
    left_frame.grid(row=0, column=0, padx=5, pady=5)

    right_frame = Frame(root, width=400, height=400, bg='white')
    right_frame.grid(row=0, column=1, padx=10, pady=5)

    # Create frames and labels in left_frame
    Label(left_frame, text="Spider").grid(row=0, column=0, padx=5, pady=5)

    # load image to be "edited"
    image = ImageTk.PhotoImage(imagen)
    original_image = image
    Label(left_frame, image=original_image).grid(row=1, column=0, padx=5, pady=5)

    # Display image in right_frame
    #Label(right_frame, image=image).grid(row=0,column=0, padx=5, pady=5)

    title_label = Label(right_frame, text=spider, font=("Helvetica", 16), bg='white')
    title_label.grid(row=0, column=0, padx=5, pady=5)

    paragraph_label = Label(right_frame, text=description, wraplength=400, justify="left", bg='white')
    paragraph_label.grid(row=1, column=0, padx=5, pady=5)

    root.mainloop()
   
def predict_image(image_path, model, class_names, img_size=(256, 256)):
    import tensorflow as tf
    from tensorflow.keras.preprocessing import image

    # Cargar y procesar la imagen
    img = image.load_img(image_path, target_size=img_size)
    img_array = image.img_to_array(img) / 255.0  # Normalizar
    img_array = tf.expand_dims(img_array, axis=0)  # Agregar dimensión batch

    # Realizar predicción
    predictions = model.predict(img_array)
    predicted_class_idx = tf.argmax(predictions[0]).numpy()
    predicted_class = class_names[predicted_class_idx]

    return {
        "Predicted Class": predicted_class,
        "Confidence Scores": {class_names[i]: round(predictions[0][i], 4) for i in range(len(class_names))}
    }



if __name__ == "__main__":
    image_path_test = input("Ingrese la ruta de la imagen a predecir: ")
    model = load_model('./modelo2_entrenado.h5')

    if 'model' in locals() and 'spider_classes' in locals():
        result = predict_image(image_path_test, model=model, class_names=spider_classes)
        print("Predicción para la imagen:", image_path_test)
        print("Clase predicha:", result["Predicted Class"])
        print("Confianza por clase:", result["Confidence Scores"])
    else:
        print("Imagen no disponible")

    genai.configure(api_key="AIzaSyDnEyMsGx9FlNXb1fRNF6RidXz6JH6Xgac")
    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content("Give me a little short information of: the lethality and what to do in case of a" + result["Predicted Class"])
    print(response.text)

    interface(spider=result["Predicted Class"], description=response.text, image_path=image_path_test)   
