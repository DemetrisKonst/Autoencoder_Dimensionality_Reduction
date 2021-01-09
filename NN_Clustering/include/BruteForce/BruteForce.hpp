#include <utility>
#include <limits>
#include <ctime>
#include <cmath>

#include "../core/item.hpp"
#include "../metrics/metrics.hpp"
#include "../utils/search.hpp"
#include "../LSH/LSHFun.hpp"

/*
The following class implements a simple brute force implementation of kNN
*/
template <typename T>
class BruteForce {
private:
  // Number of items in training set
  int imageCount;
  // Data dimension
  int dimension;
  // Array of item pointers
  Item<T>** items;

public:
  BruteForce (interface::Data<T>& ds) {
    // Initialize data
    imageCount = ds.n;
    dimension = ds.dimension;
    items = ds.items;
  }

  ~BruteForce () {

  }

  /*
  Each neighbor is represented as a pair of <distanceToQuery, neighborItem*>
  The following function returns a vector of these pairs
  */
  std::vector<std::pair<int, Item<T>*>> kNN (T* query, int N) {
    // At first initalize the vector itself
    std::vector<std::pair<int, Item<T>*>> d;
    // Then initialize each pair with distance -> (max integer) and a null item
    for (int i = 0; i < N; i++)
      d.push_back(std::make_pair(std::numeric_limits<int>::max(), new Item<T>()));


    // For each item...
    for (int i = 0; i < imageCount; i++) {
      // Calculate its distance from the query item
      int distance = metrics::ManhattanDistance<T>(query, items[i]->data, dimension);

      /*
      If the distance is less than the last pair's in the vector,
      replace the pair with the new distance and the current item.
      Then, sort the vector by ascending order based on distance.
      This is done so that whenever we find a good neighbor candidate,
      we replace the least similar neighbor in the vector
      */
      if (distance < d[N-1].first) {
        d[N-1].first = distance;
        if (d[N-1].second->null)
          delete d[N-1].second;
        d[N-1].second = items[i];
        std::sort(d.begin(), d.end(), utils::comparePairs<T>);
      }
    }

    return d;
  }

  /*
  The following function creates a SearchOutput object.
  This object contains all information required to create the output file.
  This is used only on ANN (not clustering) and also requires that its LSH or Hypercube
  counterpart will be executed (so as to compare times and distances).
  */
  void buildOutput (interface::output::SearchOutput& output, interface::Dataset<T>& query, bool original) {
    // A vector containing the ids of all query items
    std::vector<int> queryIdVec;
    // A vector containing the nearest neighbor ids of all query items
    std::vector<int> nnIdVec;
    // The true distances of the nearest neighbor of each query item
    std::vector<double> timeVec;
    double total_time = 0.0;


    // For each query item...
    for (int i = 0; i < query.number_of_images; i++) {
      // Execute kNN and calculate time elapsed
      clock_t begin = clock();

      std::vector<std::pair<int, Item<T>*>> kNNRes = kNN(query.images[i], 1);

      clock_t end = clock();
      double elapsed = double(end - begin) / CLOCKS_PER_SEC;
      total_time += elapsed;

      // Push the relevant data to the vectors
      queryIdVec.push_back(i);
      nnIdVec.push_back( (int) kNNRes[0].second->id);
      timeVec.push_back(elapsed);

      if ((i+1)%1000 == 0)
        std::cout << "BF: " << i+1 << " query items..." << '\n';
    }

    // Set the following SearchOutput's attributes to the vectors created above
    if (original){
      output.query_id = queryIdVec;
      output.true_neighbors_id = nnIdVec;
      output.true_time = timeVec;
      output.true_total_time = total_time;
    }else{
      output.reduced_neighbors_id = nnIdVec;
      output.reduced_time = timeVec;
      output.reduced_total_time = total_time;
    }
  }

  std::vector<std::vector<std::pair<int, Item<T>*>>> getNeighbors (interface::Dataset<T>& query, int N, int metric){
    std::vector<std::vector<std::pair<int, Item<T>*>>> neighbors;

    for (int i = 0; i < query.number_of_images; i++){
      std::vector<std::pair<int, Item<T>*>> kNNRes = kNN(query.images[i], N);
      neighbors.push_back(kNNRes);

      if ((i+1)%1000 == 0)
        std::cout << "BF: " << i+1 << " query items..." << '\n';
    }

    return neighbors;
  }
};
