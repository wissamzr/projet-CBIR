import streamlit as st
import numpy as np
import cv2
import os
import sys

# Ajouter le chemin vers le répertoire FeatureExtraction en utilisant des chemins absolus
current_dir = os.path.abspath(os.path.dirname(__file__))
sys.path.append(current_dir)

# Importer les modules après avoir ajouté le chemin
from descriptor import glcm, bitdesc
from distances import manhattan, euclidean, chebyshev, canberra, retrieve_similar_images

def load_signature_db(descriptor_type):
    if descriptor_type == 'glcm':
        return np.load('glcm_signatures.npy', allow_pickle=True)
    else:
        return np.load('bitdesc_signatures.npy', allow_pickle=True)

def get_absolute_path(relative_path):
    base_path = os.path.abspath("./dataset")
    return os.path.join(base_path, relative_path)

def main():
    st.title('Image Similarity Finder')
    uploaded_file = st.file_uploader('Upload an image', type=['jpg', 'jpeg', 'png'])

    descriptor = st.radio('Select Descriptor', ['glcm', 'bitdesc'])

    distance_metric = st.radio('Select Distance Metric', ['manhattan', 'euclidean', 'chebyshev', 'canberra'])

    num_results = st.number_input('Number of similar images to display', min_value=1, value=5, step=1)

    if uploaded_file is not None:
        try:
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)  # Load image in color
            if image is None:
                st.error("Failed to read the uploaded image.")
            else:
                st.image(image, channels="BGR")

                gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convertir en niveaux de gris
                if descriptor == 'glcm':
                    query_features = glcm(gray_image)
                else:
                    query_features = bitdesc(gray_image)

                st.write(f"Extracted features: {query_features}, Shape: {np.array(query_features).shape}")

                signature_db = load_signature_db(descriptor)

                st.write(f"Signature database shape: {signature_db.shape}")

                similar_images = retrieve_similar_images(signature_db, query_features, distance_metric, num_results)

                st.write(f'Top {num_results} similar images:')
                for img_path, dist, label in similar_images:
                    st.write(f'Image Path: {img_path}, Distance: {dist}, Label: {label}')
                    absolute_img_path = get_absolute_path(img_path)  # Convert to absolute path
                    if not os.path.isfile(absolute_img_path):
                        st.write(f"Failed to load image: {absolute_img_path}")
                        continue
                    img = cv2.imread(absolute_img_path)
                    if img is not None:
                        st.image(img, channels="BGR")
                    else:
                        st.write("Failed to load image:", absolute_img_path)
        except Exception as e:
            st.error(f"An error occurred: {e}")

if __name__ == '__main__':
    main()
