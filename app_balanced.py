
import streamlit as st
from PIL import Image, ImageFilter
import numpy as np
import joblib
import matplotlib.pyplot as plt
from streamlit_drawable_canvas import st_canvas  # Använd rätt import för drawable canvas


# Ladda den tränade modellen
best_model_balanced = joblib.load('best_model_balanced.pkl')

# Funktion för att förbereda och bearbeta bilden
def prepare_image(image):
    # Omvandla bilden till gråskala
    image = image.convert('L')
    # Lägg till en liten oskärpa för att hjälpa modellen att fokusera på siffran
    image = image.filter(ImageFilter.GaussianBlur(1))  # Lägg till en mild oskärpa
    # Ändra storlek till 28x28 pixlar (som MNIST-formatet)
    image = image.resize((28, 28))
    # Omvandla till en array och normalisera pixelvärden
    image_array = np.array(image)
    image_array = np.array(image).reshape(1, 784)  # Omvandla till en vektor med 784 pixlar
    image_array = image_array / 255.0  # Skala mellan 0 och 1
    
    return image_array

# Streamlit app
st.title('Streamlit MNIST Prediktera en Siffra')
st.write('Välj ett alternativ nedan för att få en siffra predikterad.')

# Alternativ 1: Ladda upp en bild
option = st.selectbox("Välj ett alternativ:", ["Ladda upp bild", "Rita en siffra", "Ta en bild med kameran"])

if option == "Ladda upp bild":
    uploaded_image = st.file_uploader("Välj en bild att ladda upp", type=["png", "jpg", "jpeg"])

    if uploaded_image is not None:
        # Visa den uppladdade bilden
        image = Image.open(uploaded_image)
        st.image(image, caption='Uppladdad bild.', use_column_width=True)

        # Förbered bilden och visa den för att säkerställa att den ser bra ut
        image_array = prepare_image(image)
        st.image(image_array.reshape(28, 28), caption='Förberedd bild för förutsägelse', use_column_width=True)

        # Gör en förutsägelse
        prediction = best_model_balanced.predict(image_array)
        st.write(f"Förutsägelsen är: {prediction[0]}")
        


elif option == "Rita en siffra":
    # Skapa en ritpanel med hjälp av st_canvas (från streamlit)
    st.write("Rita en siffra i ritytan nedan:")
    canvas_result = st_canvas(
        fill_color="white", 
        width=280, 
        height=280, 
        drawing_mode="freedraw", 
        key="canvas"
    )

    if canvas_result.image_data is not None:
        # Förbered bilden som ritas
        
        image = Image.fromarray(canvas_result.image_data)
        st.image(image, caption="Ritad bild", use_column_width=True)
        
        # Prediktera siffra
        image_array = prepare_image(image)
        prediction = best_model_balanced.predict(image_array)
        st.write(f"Förutsägelsen är: {prediction[0]}")
        


elif option == "Ta en bild med kameran":
    # Ta bild via datorns kamera
    st.write("Ta en bild med din kamera för att prediktera en siffra:")

    camera_image = st.camera_input("Kamera")

    if camera_image is not None:
        # Visa den tagna bilden
        image = Image.open(camera_image)
        st.image(image, caption='Tagen bild.', use_column_width=True)

        # Förbered och gör en prediktion
        image_array = prepare_image(image)
        prediction = best_model_balanced.predict(image_array)
        st.write(f"Förutsägelsen är: {prediction[0]}")

