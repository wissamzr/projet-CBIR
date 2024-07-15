import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops

def glcm(image):
    if len(image.shape) != 2:
        raise ValueError("L'image doit être un tableau 2D")
    
    co_matrix = graycomatrix(image, [1], [0, np.pi/4, np.pi/2, 3*np.pi/4], symmetric=True, normed=True)
    dissimilarity = graycoprops(co_matrix, 'dissimilarity').mean()
    contrast = graycoprops(co_matrix, 'contrast').mean()
    correlation = graycoprops(co_matrix, 'correlation').mean()
    energy = graycoprops(co_matrix, 'energy').mean()
    asm = graycoprops(co_matrix, 'ASM').mean()
    homogeneity = graycoprops(co_matrix, 'homogeneity').mean()
    return [dissimilarity, contrast, correlation, energy, asm, homogeneity]

def bitdesc(image):
    # Placeholder for bio_taxo function, replace with actual implementation if available
    return [np.mean(image), np.std(image)]  # Example placeholder, replace with actual implementation
