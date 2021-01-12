#pragma once

namespace utils {
  /*
  Function which evaluates a metric based on the labels of the k Nearest Neighbors returned
  by a brute force approach. Essentially, accuracy = same_label_neighbors / total_neighbors
  */
  template <typename T>
  double evaluate(std::vector<std::vector<std::pair<int, Item<T>*>>> neighbors, interface::Labelset<T> dataset_labels, interface::Labelset<T> queryset_labels, int N){
    int correct_count = 0;

    for (int i = 0; i < queryset_labels.number_of_labels; i++){
      T qlabel = queryset_labels.labels[i];

      for (int j = 0; j < N; j++){
        T nlabel = dataset_labels.labels[neighbors[i][j].second->id];
        if (qlabel == nlabel)
          correct_count++;
      }
    }

    return correct_count / (queryset_labels.number_of_labels * N * 1.0);
  }
}
