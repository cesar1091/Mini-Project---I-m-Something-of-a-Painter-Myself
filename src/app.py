import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow_addons.layers import SpectralNormalization
import os
import time

# Page Setup
st.set_page_config(
    page_title="Image Generator and Classifier 256x256",
    page_icon="🖼️",
    layout="wide"
)

# Main title
st.title("🖼️ Image Generator and Classifier (256x256x3)")

# --- Loading models ---
@st.cache_resource
def load_generator_model():
    try:
        modelo = load_model('../models/generator.keras')  # Ajusta la ruta si es necesario
        # Verificar que el modelo genera imágenes de 256x256x3
        output_shape = modelo.output_shape[1:]
        if output_shape != (256, 256, 3):
            st.warning(f"The generator produces images of size {output_shape}, not 256x256x3")
        return modelo
    except Exception as e:
        st.error(f"Error loading generator model: {e}")
        return None

@st.cache_resource
def load_discriminator_model():
    try:
        custom_objects = {'SpectralNormalization': SpectralNormalization}
        modelo = load_model('../models/discriminator.keras', custom_objects=custom_objects)  
        # Verify that the discriminator expects 256x256x3
        input_shape = modelo.input_shape[1:]
        if input_shape != (256, 256, 3):
            st.warning(f"The discriminator expects images of size {input_shape}, not 256x256x3")
        return modelo
    except Exception as e:
        st.error(f"Error loading discriminator model: {e}")
        return None

# Loading models
with st.spinner("Loading models..."):
    generador = load_generator_model()
    discriminador = load_discriminator_model()

# Verificar que los modelos se cargaron correctamente
if generador is None or discriminador is None:
    st.error("One or both models could not be loaded. Check the .keras files.")
    st.stop()

# Pestañas
tab1, tab2 = st.tabs(["Generate Images", "Classify Images"])

# --- Pestaña 1: Generate Images (256x256x3) ---
with tab1:
    st.header("Image generator (256x256)")
    st.write("Use the generator model to create new 256x256 pixel images.")
    
    # Obtener la dimensión del ruido del modelo generador
    try:
        noise_dim = generador.input_shape[1]
        st.write(f"The generator expects a noise vector of dimension: {noise_dim}")
    except:
        noise_dim = 100  # Valor por defecto si no se puede determinar
        st.warning(f"The input dimension could not be determined. Using the default value: {noise_dim}")
    
    # Controles para la generación
    col1, col2 = st.columns(2)
    with col1:
        seed = st.number_input("Seed for generation", value=42, min_value=0)
        generate_btn = st.button("Generate Image 256x256")
    
    # Función para generar imagen
    def generate_image_256(noise_dim, seed):
        np.random.seed(seed)
        noise = np.random.randn(1, noise_dim)
        generated_image = generador.predict(noise, verbose=0)
        
        # Asumimos que el generador produce imágenes en el rango [-1, 1]
        generated_image = (generated_image + 1) * 127.5  # Escalar a [0, 255]
        generated_image = np.clip(generated_image, 0, 255).astype(np.uint8)
        
        return generated_image[0]  # Remover dimensión del batch
    
    # Generar y mostrar imagen
    if generate_btn:
        with st.spinner(f"Generating 256x256x3 image with seed {seed}..."):
            try:
                # Llamar al modelo generador
                generated_image = generate_image_256(noise_dim, seed)
                
                # Mostrar la imagen con su tamaño
                st.image(generated_image, caption=f"Generated image {generated_image.shape}", use_column_width=True)
                
                # Mostrar información del tensor
                with st.expander("Technical details of the generated image"):
                    st.write(f"Shape: {generated_image.shape}")
                    st.write(f"Minimum value: {generated_image.min()}")
                    st.write(f"Maximum value: {generated_image.max()}")
                    st.write(f"Data type: {generated_image.dtype}")
                
                # Opción para descargar
                img_pil = Image.fromarray(generated_image)
                st.download_button(
                    label="Download image 256x256",
                    data=img_pil.tobytes(),
                    file_name=f"image_256_{seed}.png",
                    mime="image/png"
                )
            except Exception as e:
                st.error(f"Error generating image: {e}")

# --- Pestaña 2: Clasificar Imágenes (256x256x3) ---
with tab2:
    st.header("Image Classifier (256x256)")
    st.write("Upload an image to classify as real or generated (fake). The model expects 256x256 pixel images.")
    
    # Función para preprocesar imagen para 256x256x3
    def preprocess_image_256(imagen):
        # Redimensionar a 256x256
        imagen = imagen.resize((256, 256))
        imagen_array = np.array(imagen)
        
        # Manejar diferentes formatos de imagen:
        if len(imagen_array.shape) == 2:  # Escala de grises
            imagen_array = np.stack((imagen_array,)*3, axis=-1)
        elif imagen_array.shape[2] == 4:  # RGBA
            imagen_array = imagen_array[:, :, :3]
        elif imagen_array.shape[2] == 1:  # 1 canal
            imagen_array = np.concatenate([imagen_array]*3, axis=-1)
        
        # Normaliz2 a [-1, 1] si el modelo fue entrenado así
        imagen_array = (imagen_array / 127.5) - 1.0
        
        return np.expand_dims(imagen_array, axis=0)  # Añadir dimensión del batch
    
    # Función para clasificar imagen
    def classify_image_256(imagen):
        # Preprocess image
        imagen_procesada = preprocess_image_256(imagen)
        
        # Classify image
        prediccion = discriminador.predict(imagen_procesada, verbose=0)[0][0]
        es_real = prediccion > 0.5
        confianza = prediccion if es_real else 1 - prediccion
        return es_real, confianza
    
    # Subir imagen
    uploaded_file = st.file_uploader(
        "Choose an image (it will be resized to 256x256)...", 
        type=["jpg", "jpeg", "png", "bmp"],
        key="classifier"
    )
    
    if uploaded_file is not None:
        # Mostrar imagen subida
        try:
            imagen = Image.open(uploaded_file)
            
            # Mostrar información de la imagen original
            with st.expander("Original image information"):
                st.write(f"Original size: {imagen.size}")
                st.write(f"Mode: {imagen.mode}")
            
            # Mostrar preview redimensionada
            st.image(imagen, caption="Uploaded image (preview)", use_column_width=True)
            
            # Clasificar
            with st.spinner("Classifying image (resizing to 256x256x3)..."):
                es_real, confianza = classify_image_256(imagen)
            
            # Mostrar resultados
            st.subheader("Qualifying results")
            if es_real:
                st.success(f"✅ The image is REAL with {confianza*100:.2f}% confidence")
            else:
                st.error(f"❌ The image is FALSE (generated) with {(1-confianza)*100:.2f}% confidence")
            
            # Mostrar preprocesamiento
            with st.expander("Preprocessing details"):
                imagen_procesada = preprocess_image_256(imagen)[0]
                st.write(f"Shape after preprocessing: {imagen_procesada.shape}")
                st.write(f"Range of values: {imagen_procesada.min():.2f} to {imagen_procesada.max():.2f}")
                st.image((imagen_procesada + 1) * 127.5, caption="Preprocessed image (visualization)", clamp=True)
                
        except Exception as e:
            st.error(f"Error processing image: {e}")

# Información en el sidebar
st.sidebar.markdown("""
### Technical Specifications:
- **Generator Model**:
- Input: Noise vector of dimension {}
- Output: 256x256x3 image in range [-1, 1]

- **Discriminator Model**:
- Input: 256x256x3 image
- Output: Probability of being real

### Instructions:
1. **Generate Images**:
- Adjust the seed to change the generated image
- Images are generated in 256x256 pixels

2. **Classify Images**:
- Any uploaded images will be resized to 256x256
- They will be converted to RGB if necessary
""".format(generador.input_shape[1] if generador else "?"))

# Verificación final de shapes
st.sidebar.markdown("""
### Shape Verification:
- Input generator: {}
- Output generator: {}
- Input discriminator: {}
""".format(
    generador.input_shape if generador else "N/A",
    generador.output_shape if generador else "N/A",
    discriminador.input_shape if discriminador else "N/A"
))