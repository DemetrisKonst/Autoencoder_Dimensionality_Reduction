#ifndef _INTERFACE
#define _INTERFACE

#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <cstring>
#include <sys/stat.h>
#include <unistd.h>
#include <fstream>
#include <regex>
#include <arpa/inet.h>
#include <vector>
#include <utility>
#include <algorithm>
#include <limits>

#include "interface_utils.hpp"
#include "../core/item.hpp"


/* namespace regarding interface utilities */
namespace interface
{

  /* struct used to group the IO (input - output) values */
  typedef struct IOFiles
  {
    std::string input_file = "";
    std::string query_file = "";
    std::string reduced_input_file = "";
    std::string reduced_query_file = "";
    std::string output_file = "";
  } IOFiles;

  /* struct used to group the IO (input - output) values for MH */
  typedef struct IOFilesMH
  {
    std::string input_file = "";
    std::string query_file = "";
    std::string input_labels_file = "";
    std::string query_labels_file = "";
    std::string output_file = "";
  } IOFilesMH;

  /* struct used to group the IOC (input - output - configuration) values */
  typedef struct IOCFiles
  {
    std::string input_file = "";
    std::string reduced_input_file = "";
    std::string labels_file = "";
    std::string configuration_file = "";
    std::string output_file = "";
  } IOCFiles;

  /* struct used to define the contents of a dataset */
  template <typename T>
  struct Dataset
  {
    uint32_t magic_number = 0;
    uint32_t number_of_images = 0;
    uint32_t rows_per_image = 0;
    uint32_t columns_per_image = 0;
    T** images = NULL;
  };

  /* struct used to define a set containing labels for classes */
  template <typename T>
  struct Labelset
  {
    uint32_t magic_number = 0;
    uint32_t number_of_labels = 0;
    T* labels = NULL;
  };

  /* struct used to define the contents of a file containing pre-assigned clusters */
  typedef struct Clusterset
  {
    uint16_t K = 10;
    std::vector<uint32_t> sizes;
    std::vector<std::vector<uint32_t>> images_in_clusters;
  } Clusterset;

  /* struct used to move around the Dataset */
  template <typename T>
  struct Data
  {
    uint32_t n = 0;
    uint16_t dimension = 0;
    Item<T>** items = NULL;

    Data(Dataset<T>& dataset)
    {
      /* initialize the values of the Data */
      n = dataset.number_of_images;
      dimension = dataset.rows_per_image * dataset.columns_per_image;
      items = new Item<T>*[n];

      /* create each item separately */
      for (int i = 0; i < n; i++)
      {
        items[i] = new Item<T>(i, dataset.images[i], false, false);
      }
    }

    ~Data(void)
    {
      /* free each item separately */
      for (int i = 0; i < n; i++)
      {
        delete items[i];
      }

      /* now delete the items array */
      delete[] items;
      items = NULL;

    }
  };


  /* function used to read input from the dataset */
  template <typename T>
  int ParseDataset(const std::string& filename, Dataset<T>& dataset, ExitCode& status, bool values_in_big_endian=false)
  {
    /* create an ifstream item to open and navigate the file */
    std::ifstream input_file(filename, std::ios::binary);

    /* make sure that the file successfully opened */
    if (!input_file.is_open())
    {
      status = INVALID_DATASET_PATH;
      return 0;
    }

    /* temp variables to read input from the file */
    uint32_t temp_big_endian;

    /* read the magic number, number of images and their dimensions */
    input_file.read((char *) &(temp_big_endian), sizeof(temp_big_endian));
    dataset.magic_number = ntohl(temp_big_endian);

    input_file.read((char *) &(temp_big_endian), sizeof(temp_big_endian));
    dataset.number_of_images = ntohl(temp_big_endian);

    input_file.read((char *) &(temp_big_endian), sizeof(temp_big_endian));
    dataset.rows_per_image = ntohl(temp_big_endian);

    input_file.read((char *) &(temp_big_endian), sizeof(temp_big_endian));
    dataset.columns_per_image = ntohl(temp_big_endian);


    /* compute the "area" of an image to avoid computing it again */
    uint32_t area = dataset.rows_per_image * dataset.columns_per_image;

    /* initialize the images (pixels array) */
    dataset.images = new T*[dataset.number_of_images];

    /* iterate through the array to allocate space for the pixels of each image, while reading the image at the same time */
    for (int i = 0; i < dataset.number_of_images; i++)
    {
      dataset.images[i] = new T[area];
      /* add the data points */
      for (uint32_t data_point = 0; data_point < area; data_point++)
      {
        input_file.read((char *) &dataset.images[i][data_point], sizeof(T));
        if (values_in_big_endian)
        {
          dataset.images[i][data_point] = ntohs(dataset.images[i][data_point]);
        }
      }
    }

    /* everything is done, close the file and return */
    input_file.close();
    return 1;
  }


  /* function used to parse the set containing labels */
  template <typename T>
  int ParseLabelset(const std::string& filename, Labelset<T>& labelset, ExitCode& status, bool values_in_big_endian=false)
  {
    /* create an ifstream item to open and navigate the file */
    std::ifstream input_file(filename, std::ios::binary);

    /* make sure that the file successfully opened */
    if (!input_file.is_open())
    {
      status = INVALID_LABELSET_PATH;
      return 0;
    }

    /* temp variables to read input from the file */
    uint32_t temp_big_endian;

    /* read the magic number and the number of labels */
    input_file.read((char *) &(temp_big_endian), sizeof(temp_big_endian));
    labelset.magic_number = ntohl(temp_big_endian);

    input_file.read((char *) &(temp_big_endian), sizeof(temp_big_endian));
    labelset.number_of_labels = ntohl(temp_big_endian);

    /* initialize the images (pixels array) */
    labelset.labels = new T[labelset.number_of_labels];

    /* iterate through the array to allocate space for the pixels of each image, while reading the image at the same time */
    for (int label = 0; label < labelset.number_of_labels; label++)
    {
      input_file.read((char *) &labelset.labels[label], sizeof(T));
      if (values_in_big_endian)
      {
        labelset.labels[label] = ntohs(labelset.labels[label]);
      }
    }

    /* everything is done, close the file and return */
    input_file.close();
    return 1;
  }


  /* function used to parse a file containing pre-assigned clusters per image */
  int ParseClusterset(const std::string& filename, Clusterset& clusterset, const uint16_t& K, ExitCode& status)
  {
    /* create in ifstream object to read the input from the cluster set file */
    std::ifstream clusters_file(filename);

    /* make sure that the file successfully opened */
    if (!clusters_file.is_open())
    {
      status = INVALID_CLUSTERSET_PATH;
      return 0;
    }

    /* initialize the vectors of the clusterset */
    clusterset.K = K;

    /* create a variable to read lines from the file */
    std::string line = "ALFZ FYGE";

    /* create a regex to match all integer values */
    std::regex reg("[0-9]+");

    /* for every cluster */
    for (size_t i = 0; i < K; i++)
    {
      /* smatch object */
      std::smatch matches;

      /* match all numbers and remove the first number which is the cluster number */
      std::getline(clusters_file, line);
      std::regex_search(line, matches, reg);
      line = matches.suffix().str();

      /* match again to get the size, and remove it */
      std::regex_search(line, matches, reg);
      uint32_t size = stoi(matches[0]);
      clusterset.sizes.push_back(size);
      line = matches.suffix().str();


      /* now start matching for the image IDs */
      std::vector<uint32_t> images;
      while(std::regex_search(line, matches, reg))
      {
        images.push_back(stoi(matches[0]));
        line = matches.suffix().str();
      }

      /* perform a sanity check */
      if (size != images.size())
      {
        status = INVALID_SIZE_IN_CLUSTERSET;
        return 0;
      }

      /* add the labels to the clusterset object */
      clusterset.images_in_clusters.push_back(images);
    }

    /* everythind is done, close the file and return */
    clusters_file.close();
    return 1;
  }


  /* function to free up the memory that the dataset used */
  template <typename T>
  void freeDataset(Dataset<T>& dataset)
  {
    /* free the data points one-by-one */
    for (int i = 0; i < dataset.number_of_images; i++)
    {
      delete[] dataset.images[i];
    }

    /* free the array used for the images */
    delete[] dataset.images;
    dataset.images = NULL;
  }

  /* function to free up the memory that the labelset used */
  template <typename T>
  void freeLabelset(Labelset<T>& labelset)
  {
    delete[] labelset.labels;
    labelset.labels = NULL;
  }


  /* namespace regarding output utilities */
  namespace output
  {
    /* struct used to group the output data */
    typedef struct SearchOutput
    {
      uint32_t n;
      // IDs of query items
      std::vector<int> query_id;

      // IDs of NNs from query
      std::vector<int> reduced_neighbors_id;
      std::vector<int> lsh_neighbors_id;
      std::vector<int> true_neighbors_id;

      // Distances of NNs from query
      std::vector<double> reduced_distance;
      std::vector<double> lsh_distance;
      std::vector<double> true_distance;

      // Singular time to find NN for one query
      std::vector<double> reduced_time;
      std::vector<double> lsh_time;
      std::vector<double> true_time;

      // Total times to find NNs for all queries
      double reduced_total_time;
      double lsh_total_time;
      double true_total_time;

      // Approximation factors for LSH and Reduced space
      double lsh_approx;
      double reduced_approx;
    } SearchOutput;

    /* function to write the output of LSH/HC to a file */
    int writeOutput(const std::string& outfile_name, SearchOutput& output, ExitCode& status)
    {
      /* create in ifstream object to open the output file */
      std::ofstream outfile;
      outfile.open(outfile_name, std::ios::out | std::ios::trunc);
      /* make sure that the file successfully opened */
      if (!outfile.is_open())
      {
        status = INVALID_OUTFILE_PATH;
        return 0;
      }
      /* main loop to write the result for each query in the query set */
      for (int i = 0; i < output.n; i++)
      {
        outfile << "Query: " << output.query_id[i] + 1 << std::endl;


        outfile << "Nearest neighbor Reduced: " << output.reduced_neighbors_id[i] << std::endl;
        outfile << "Nearest neighbor LSH: " << output.lsh_neighbors_id[i] << std::endl;
        outfile << "Nearest neighbor True: " << output.true_neighbors_id[i] << std::endl;

        outfile << "distanceReduced: " << output.reduced_distance[i] << std::endl;
        outfile << "distanceLSH: " << output.lsh_distance[i] << std::endl;
        outfile << "distanceTrue: " << output.true_distance[i] << std::endl;

        outfile << "tReduced: " << output.reduced_time[i] << std::endl;
        outfile << "tLSH: " << output.lsh_time[i] << std::endl;
        outfile << "tTrue: " << output.true_time[i] << std::endl;

        outfile << std::endl;
      }

      outfile << "Reduced Total Time: " << output.reduced_total_time << std::endl;
      outfile << "LSH Total Time: " << output.lsh_total_time << std::endl;
      outfile << "True Total Time: " << output.true_total_time << std::endl;

      outfile << "Approximation Factor Reduced: " << output.reduced_approx << std::endl;
      outfile << "Approximation Factor LSH: " << output.lsh_approx << std::endl;

      outfile.close();
      return 1;
    }
  }

}


#endif
