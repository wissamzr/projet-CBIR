import numpy as np
from scipy.spatial import distance

def manhattan(v1, v2):
    v1 = np.array(v1).astype('float')
    v2 = np.array(v2).astype('float')
    dist = np.sum(np.abs(v1 - v2))
    return dist

def euclidean(v1, v2):
    v1 = np.array(v1).astype('float')
    v2 = np.array(v2).astype('float')
    dist = np.sqrt(np.sum((v1 - v2) ** 2))
    return dist

def chebyshev(v1, v2):
    v1 = np.array(v1).astype('float')
    v2 = np.array(v2).astype('float')
    dist = np.max(np.abs(v1 - v2))
    return dist

def canberra(v1, v2):
    return distance.canberra(v1, v2)

def retrieve_similar_images(features_db, query_features, distance_metric, num_results):
    distances = []
    for instance in features_db:
        features, label, img_path = instance[:-2], instance[-2], instance[-1]
        if distance_metric == 'manhattan':
            dist = manhattan(query_features, features)
        elif distance_metric == 'euclidean':
            dist = euclidean(query_features, features)
        elif distance_metric == 'chebyshev':
            dist = chebyshev(query_features, features)
        elif distance_metric == 'canberra':
            dist = canberra(query_features, features)
        else:
            raise ValueError("Unknown distance metric")
        distances.append((img_path, dist, label))
    distances.sort(key=lambda x: x[1])
    return distances[:num_results]
