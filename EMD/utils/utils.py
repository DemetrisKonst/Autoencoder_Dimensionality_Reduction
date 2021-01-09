import math
import numpy as np

def check_total_image_values(dataset, queryset):
    prev_sum = None
    for image in dataset:
        image_sum = np.sum(image)
        if prev_sum is not None:
            if prev_sum != image_sum:
                return False
        else:
            prev_sum = image_sum

    for image in queryset:
        image_sum = np.sum(image)
        if prev_sum != image_sum:
            return False

    return True

def normalize_set(set):
    set = set.astype(np.float64)
    for i in range(len(set)):
        image_sum = np.sum(set[i])
        set[i] = set[i] / (image_sum*1.0)

    return set

def sort_alongside(distances, neighbors):
    zipped = zip(distances, neighbors)
    std = sorted(zipped)

    tuples = zip(*std)
    distances, neighbors = [list(tuple) for tuple in tuples]

    return distances, neighbors

def kNN(dataset, queryset, N, metric):
    neighbors_arr = []
    distances_arr = []

    for i in range(len(queryset)):
        print(i)
        neighbors = [0 for i in range(N)]
        distances = [math.inf for i in range(N)]
        for j in range(len(dataset)):
            dist = metric(dataset[j], queryset[i])

            if (dist < distances[N-1]):
                distances[N-1] = dist
                neighbors[N-1] = j
                distances, neighbors = sort_alongside(distances, neighbors)


        neighbors_arr.append(neighbors)
        distances_arr.append(distances)

    return neighbors_arr, distances_arr

def evaluate(dlabels, qlabels, neighbors, N):
    correct_count = 0
    for i in range(len(neighbors)):
        qlabel = qlabels[i]

        for neighbor in neighbors[i]:
            nlabel = dlabels[neighbor]
            if nlabel == qlabel:
                correct_count += 1


    return correct_count / (len(neighbors)*N)
