#ifndef _MH_INTERFACE
#define _MH_INTERFACE

#include <cstdlib>
#include <cstdint>

#include "interface.hpp"

/* namespace regarding interface utilities */
namespace interface
{

  /* namespace regarding command line parameters utilities */
  namespace input
  {

    /* namespace regarding MANHATTAN */
    namespace MH
    {
      /* function to parse command line parameters */
      int MHParseInput(const int& argc, const char* argv[], IOFilesMH& files, ExitCode& status)
      {
        /* first check that at least some parameters have been given */
        if (argc == 1)
        {
          status = NO_INPUT;
          return 0;
        }

        /* now check if the option "--help" has been given, or if the input length implies a mistake */
        if (argc % 2 == 0)
        {
          /* make sure that the option "--help" was given */
          if (!strcmp(argv[1], "--help"))
          {
            status = HELP_MSG_MANHATTAN;
            return 0;
          }

          /* if we get here it means the option was not given, so act accordingly */
          status = INVALID_INPUT_MANHATTAN;
          return 0;
        }

        /* keep flags to make sure that all the needed outputs are provided */
        bool flag_d = false;
        bool flag_q = false;
        bool flag_l1 = false;
        bool flag_l2 = false;
        bool flag_o = false;

        /* start iterating the arguments array */
        for (int i = 1; i < argc; i += 2)
        {
          /* get the parameter and the value */
          std::string option(argv[i]);
          std::string value(argv[i+1]);

          if (option == "-d")
          {
            flag_d = true;

            if (!FileExists(value))
            {
              status = INVALID_INFILE_PATH;
              return 0;
            }
            files.input_file = value;
          }
          else if (option == "-q")
          {
            flag_q = true;
            if (!FileExists(value))
            {
              status = INVALID_QFILE_PATH;
              return 0;
            }
            files.query_file = value;
          }
          else if (option == "-l1")
          {
            flag_l1 = true;
            if (!FileExists(value))
            {
              status = INVALID_RED_INFILE_PATH;
              return 0;
            }
            files.input_labels_file = value;
          }
          else if (option == "-l2")
          {
            flag_l2 = true;
            if (!FileExists(value))
            {
              status = INVALID_RED_QFILE_PATH;
              return 0;
            }
            files.query_labels_file = value;
          }
          else if (option == "-o")
          {
            flag_o = true;
            if (!FileIsAccessible(value))
            {
              status = INVALID_OUTFILE_PATH;
              return 0;
            }
            files.output_file = value;
          }
          else
          {
            status = INVALID_INPUT_LSH;
            return 0;
          }

        }

        /* check if the input file (dataset) was not given */
        if (!flag_d || !flag_q || !flag_l1 || !flag_l2 || !flag_o)
        {
          status = NO_INPUT;
          return 0;
        }

        /* everything went ok, return */
        return 1;
      }

    }
  }

  namespace output
  {

    namespace MH
    {

      int writeOutput(const std::string& outfile_name, double average, ExitCode& status)
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

        outfile << "Average Correct Search Results MANHATTAN: " << average << std::endl;

        outfile.close();
        return 1;
      }

    }

  }
}


#endif
