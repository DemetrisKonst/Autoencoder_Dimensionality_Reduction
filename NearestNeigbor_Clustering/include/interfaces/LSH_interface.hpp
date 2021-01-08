#ifndef _LSH_INTERFACE
#define _LSH_INTERFACE

#include <cstdlib>
#include <cstdint>

#include "interface.hpp"

/* namespace regarding interface utilities */
namespace interface
{

  /* namespace regarding command line parameters utilities */
  namespace input
  {

    /* namespace regarding LSH */
    namespace LSH
    {
      /* struct used to define the LSH input (from the command line), the rest (epsilon, c) can be found in a configfile */
      typedef struct LSHInput
      {
        uint8_t k = 4;
        uint8_t L = 5;
        uint32_t N = 1;
        double R = 10000.0;
      } LSHInput;

      /* function to parse command line parameters */
      int LSHParseInput(const int& argc, const char* argv[], LSHInput& input, IOFiles& files, ExitCode& status)
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
            status = HELP_MSG_LSH;
            return 0;
          }

          /* if we get here it means the option was not given, so act accordingly */
          status = INVALID_INPUT_LSH;
          return 0;
        }

        /* keep flags to make sure that all the needed outputs are provided */
        bool flag_d = false;
        bool flag_i = false;
        bool flag_q = false;
        bool flag_s = false;
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
          else if (option == "-i")
          {
            flag_i = true;
            if (!FileExists(value))
            {
              status = INVALID_RED_INFILE_PATH;
              return 0;
            }
            files.reduced_input_file = value;
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
          else if (option == "-s")
          {
            flag_s = true;
            if (!FileExists(value))
            {
              status = INVALID_RED_QFILE_PATH;
              return 0;
            }
            files.reduced_query_file = value;
          }
          else if (option == "-k")
          {
            int value_as_int = stoi(value);
            if (value_as_int <= 0 || value_as_int >= 32)
            {
              status = INVALID_K;
              return 0;
            }
            input.k = value_as_int;
          }
          else if (option == "-L")
          {
            int value_as_int = stoi(value);
            if (value_as_int <= 0)
            {
              status = INVALID_L;
              return 0;
            }
            input.L = value_as_int;
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
        if (!flag_d || !flag_i || !flag_q || !flag_s || !flag_o)
        {
          status = NO_INPUT;
          return 0;
        }

        /* everything went ok, return */
        return 1;
      }


    }
  }
}


#endif
