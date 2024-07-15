from descriptor import glcm, bitdesc
import numpy as np
import os, cv2

def process_datasets(root_folder):
    print('Function calling')
    all_features = []
    count = 1
    
    for root, dirs, files in os.walk(root_folder):
        # print(f'Root: {root}')
        # print(f'Dirs: {dirs}')
        # print(f'Files: {files}')
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                # Construct the relative path
                # Get relative path without root_folder
                relative_path = os.path.relpath(os.path.join(root, file), root_folder)
                # Get relative path with root_folder
                image_rel_path = os.path.join(root, file)
                # Get Image folder name
                folder_name = os.path.basename(os.path.dirname(image_rel_path))
                # Feature extraction
                extraction = glcm(image_rel_path)
                extraction = extraction + [folder_name, image_rel_path]
                all_features.append(extraction)
                # print(f'{int(((count/len(files))) * 100)} % extracted')
            count +=1
    print('Extraction completed!')
    print('Now creating signatures DB ...')
    signatures = np.array(all_features)
    print(f"Signature's shape: {signatures.shape}")
    np.save('signatures.npy', signatures)
    print('Signature successfully stored!')
                
        

if __name__ == '__main__':        
    process_datasets('../Cbir_datasets')
    # img = cv2.imread('../Cbir_datasets/Wildfire/nofire/nofire_0560.jpg')
    # cv2.imshow('image', img)
    # cv2.waitKey(0)
    