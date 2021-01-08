#ifndef _CLUSTERING_INTERFACE
#define _CLUSTERING_INTERFACE

#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <cstring>
#include <sys/stat.h>
#include <unistd.h>
#include <fstream>
#include <regex>
#include <iomanip>

#include "interface.hpp"


/* namespace regarding interface utilities */
namespace interface
{
  /* namespace regarding command line parameters utilities */
  namespace input
  {
    /* namespace regarding the clustering algorithms */
    namespace clustering
    {
      /* struct used to define the clustering input from the configuration file */
      typedef struct ClusteringConfig
      {
        uint16_t clusters_K = 10;
      } ClusteringConfig;

      /* function to parse command line parameters */
      int ClusteringParseInput(const int& argc, const char* argv[], IOCFiles& files, ExitCode& status)
      {
        /* first check that at least some parameters have been given */
        if (argc == 1)
        {
          status = NO_INPUT_CLUSTERING;
          return 0;
        }

        /* exactly 11 parameters should be given in the command line */
        if (argc != 11)
        {
          /* now check if the option "--help" has been given */
          if (!strcmp(argv[1], "--help"))
          {
            status = HELP_MSG_CLUSTERING;
            return 0;
          }

          /* if we get here it means the option "help" was not given, so act accordingly */
          status = INVALID_INPUT_CLUSTERING;
          return 0;
        }

        /* use flags to make sure that the command line input is valid */
        bool flag_d = false;
        bool flag_i = false;
        bool flag_n = false;
        bool flag_c = false;
        bool flag_o = false;

        /* iterate through the arguments to find the values */
        for (size_t i = 1; i < argc; i += 2)
        {
          /* get the option provided by the user */
          std::string option(argv[i]);
          std::string value(argv[i+1]);

          /* enumerate the command line parameters */
          if (option == "-d")
          {
            if (!FileExists(value))
            {
              status = INVALID_INFILE_PATH;
              return 0;
            }
            flag_d = true;
            files.input_file = value;
          }
          else if (option == "-i")
          {
            if (!FileExists(value))
            {
              status = INVALID_RED_INFILE_PATH;
              return 0;
            }
            flag_i = true;
            files.reduced_input_file = value;
          }
          else if (option == "-n")
          {
            if (!FileExists(value))
            {
              status = INVALID_PRED_LABELS_PATH;
              return 0;
            }
            flag_n = true;
            files.labels_file = value;
          }
          else if (option == "-c")
          {
            std::string value(argv[i + 1]);
            if (!FileExists(value))
            {
              status = INVALID_CONFIG_PATH;
              return 0;
            }
            flag_c = true;
            files.configuration_file = value;
          }
          else if (option == "-o")
          {
            if (!FileIsAccessible(value))
            {
              status = INVALID_OUTFILE_PATH;
              return 0;
            }
            flag_o = true;
            files.output_file = value;
          }
          else
          {
            status = INVALID_INPUT_CLUSTERING;
            return 0;
          }
        }

        /* check if the input file (dataset) was not given */
        if (!flag_d || !flag_i || !flag_n || !flag_c || !flag_o)
        {
          status = NO_INPUT_CLUSTERING;
          return 0;
        }

        /* everything went ok, return */
        return 1;
      }

      /* function to parse configuration file */
      int ClusteringParseConfigFile(const std::string& filename, ClusteringConfig& config, ExitCode& status)
      {
        /* create in ifstream object to read the input from the configuration file */
        std::ifstream config_file(filename);

        /* make sure that the file successfully opened */
        if (!config_file.is_open())
        {
          status = INVALID_CONFIG_PATH;
          return 0;
        }

        /* create a variable to read lines from the file */
        std::string line = "";

        /* create a regex to match all integer values */
        std::regex reg("(-?[0-9]+)");

        /* create an std::smatch object to match regexes */
        std::smatch matches;


        /* field 1: number_of_clusters: <int>                    // K of K-means */
        std::getline(config_file, line);
        /* search matches */
        std::regex_search(line, matches, reg);

        /* check if a value for the number of clusters has been given */
        if (!matches.empty())
        {
          int value_as_int = stoi(matches.str(1));
          /* check for the correct range of values */
          if (value_as_int <= 1 || value_as_int >= 256)
          {
            status = CONFIG_INVALID_CLUSTERS;
            return 0;
          }
          config.clusters_K = value_as_int;
        }
        /* else, raise an error because this field is mandatory */
        else
        {
          status = CONFIG_NO_CLUSTERS;
          return 0;
        }

        /* everythind is done, close the file and return */
        config_file.close();
        return 1;
      }
    }
  }

  /* namespace regarding output file utilities */
  namespace output
  {
    /* namespace regarding clustering */
    namespace clustering
    {
      /* struct used to group all the clustering output information */
      template <typename T>
      struct ClusteringOutput
      {
        uint16_t K = 10;
        uint16_t d = 0;
        std::string header = "";
        std::vector<int> cluster_sizes;
        std::vector<T*> centroids;
        double clustering_time = 0.0;
        bool skip_until_silhouette = false;
        double* cluster_silhouettes = NULL;
        double total_silhouette = 0.0;
        double value_of_objective_function = 0.0;
      };

      /* function that writes the desired output to the outfile */
      template <typename T>
      int writeOutput(const std::string& outfile_name, const ClusteringOutput<T>& output, ExitCode& status)
      {
        /* create in ifstream object to open the output file and append information to it */
        std::ofstream outfile;
        outfile.open(outfile_name, std::ios::out | std::ios::app);

        /* make sure that the file successfully opened */
        if (!outfile.is_open())
        {
          status = INVALID_OUTFILE_PATH;
          return 0;
        }

        /* log header */
        outfile << output.header << std::endl;

        /* if the clusters information is to not be skipped */
        if (!output.skip_until_silhouette)
        {
          /* log information about clusters */
          for (int c = 0; c < output.K; c++)
          {
            outfile << "Cluster-" << c + 1 << " {size: " << output.cluster_sizes[c] << " centroid: [";
            for (int j = 0; j < output.d; j++)
            {
              outfile << +output.centroids[c][j];
              if (j < output.d - 1)
              {
                outfile << " ";
              }
            }
            outfile << "]}" << std::endl;
          }

          /* log information about clustering time */
          outfile << "clustering_time: " << output.clustering_time << std::endl;
        }

        /* log information about silhouettes */
        outfile << "Silhouette: [";
        for (int c = 0; c < output.K; c++)
        {
          outfile << output.cluster_silhouettes[c] << ", ";
        }
        outfile << output.total_silhouette << "]" << std::endl;

        /* log the value of the objective funtion */
        outfile << "Value of Objective Function: " << output.value_of_objective_function << std::endl;

        /* write also 2 newlines if this is not the last output to be appended in the file */
        if (!output.skip_until_silhouette)
        {
          outfile << "\n\n";
        }

        /* everything is done, close the outfile and return */
        outfile.close();
        return 1;
      }

    }
  }
}


#endif
