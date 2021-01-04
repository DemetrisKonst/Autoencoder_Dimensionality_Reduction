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
#include <arpa/inet.h>
#include <vector>
#include <utility>
#include <algorithm>
#include <limits>

#include "interface_utils.h"
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

  /* struct used to group the IOC (input - output - configuration) values */
  typedef struct IOCFiles
  {
    std::string input_file = "";
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
      status = INVALID_INFILE_PATH;
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
      status = INVALID_INFILE_PATH;
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


  /* namespace regarding output utilities */
  namespace output
  {
    /* struct used to group the output data */
    typedef struct KNNOutput
    {
      uint32_t n;
      std::string method;
      std::vector<int> query_id;    // item id
      std::vector<std::vector<int>> n_neighbors_id;     // knn neighbors item id
      std::vector<std::vector<double>> approx_distance; // knn distance
      std::vector<std::vector<double>> true_distance;   // real distance
      std::vector<double> approx_time;  //time to complete kNN for 1 query
      std::vector<double> true_time;
      std::vector<std::vector<int>> r_near_neighbors_id; // item id of range search neighbors
    } KNNOutput;

    /* function to write the output of LSH/HC to a file */
    int writeOutput(const std::string& outfile_name, KNNOutput& output, ExitCode& status)
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
        /* log the number in the query set of the i-th query image */
        outfile << "Query: " << output.query_id[i] + 1 << std::endl;

        /* log info for N Nearest-Neighbors */
        uint32_t N = output.n_neighbors_id[i].size();

        for (int j = 0; j < N; j++)
        {
          outfile << "Nearest neighbor-" << j + 1 << ": " << output.n_neighbors_id[i][j] << std::endl;
          outfile << "distance" << output.method << ": " << output.approx_distance[i][j] << std::endl;
          outfile << "distanceTrue: " << output.true_distance[i][j] << std::endl;
        }

        /* write the info for the execution times */
        outfile << "t" << output.method << ": " << output.approx_time[i] << std::endl;

        outfile << "tTrue: " << output.true_time[i] << std::endl;

        /* log information about R-near neighbors */
        outfile << "R-near neighbors" << std::endl;

        uint32_t neighbors = output.r_near_neighbors_id[i].size();

        for (int j = 0; j < neighbors; j++)
        {
          outfile << output.r_near_neighbors_id[i][j] << std::endl;
        }

        /* put a newline between each query */
        outfile << std::endl;
      }

      /* everything is done, close the file and return */
      outfile.close();
      return 1;
    }
  }

}


#endif
