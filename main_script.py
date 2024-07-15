import cv2
import os
import numpy as np
import sys

# Ajouter le chemin vers le répertoire FeatureExtraction en utilisant des chemins absolus
current_dir = os.path.abspath(os.path.dirname(__file__))
sys.path.append(current_dir)

# Importer les modules après avoir ajouté le chemin
from descriptor import glcm, bitdesc

def process_datasets(root_folder, descriptor_func, output_file):
    print('Function calling')
    all_features = []
    count = 1

    for root, dirs, files in os.walk(root_folder):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                relative_path = os.path.relpath(os.path.join(root, file), root_folder)
                image_rel_path = os.path.join(root, file)
                folder_name = os.path.basename(os.path.dirname(image_rel_path))
                img = cv2.imread(image_rel_path, 0)
                if img is not None:
                    extraction = descriptor_func(img)
                    extraction = extraction + [folder_name, image_rel_path]
                    all_features.append(extraction)
                    print(f'Processed {count}/{len(files)} files')
                    count += 1

    print('Extraction completed!')
    print('Now creating signatures DB ...')
    signatures = np.array(all_features)
    print(f"Signature's shape: {signatures.shape}")
    np.save(output_file, signatures)
    print('Signature successfully stored!')

if __name__ == '__main__':
    # Process GLCM features
    process_datasets('./images', glcm, 'glcm_signatures.npy')
    # Process Bitdesc features
    process_datasets('./images', bitdesc, 'bitdesc_signatures.npy')
