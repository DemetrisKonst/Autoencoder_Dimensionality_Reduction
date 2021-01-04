#include <iostream>
#include <chrono>
#include <utility>

#include "../include/interfaces/LSH_interface.h"
#include "../include/BruteForce/BruteForce.hpp"
#include "../include/LSH/LSH.hpp"

int main(int argc, char const *argv[]) {

  /* define useful variables */
  int success = 0;
  bool response = true;
  interface::ExitCode status;
  interface::Dataset<uint8_t> original_dataset;
  interface::Dataset<uint16_t> reduced_dataset;
  interface::Dataset<uint8_t> original_queryset;
  interface::Dataset<uint16_t> reduced_queryset;
  interface::IOFiles files;
  interface::input::LSH::LSHInput lsh_input;
  interface::output::KNNOutput output;


  /* parse LSH input */
  success = interface::input::LSH::LSHParseInput(argc, argv, lsh_input, files, status);
  /* check for potential errors or violations */
  if (success != 1) {
    interface::output::PrintErrorMessageAndExit(status);
  }

  std::cout << "peos1" << std::endl;
  /* parse original dataset */
  success = interface::ParseDataset(files.input_file, original_dataset, status);
  if (success != 1) {
    interface::output::PrintErrorMessageAndExit(status);
  }
  interface::Data<uint8_t> original_data(original_dataset);

  std::cout << "peos2" << std::endl;
  /* parse reduced dataset */
  success = interface::ParseDataset(files.reduced_input_file, reduced_dataset, status);
  if (success != 1) {
    interface::output::PrintErrorMessageAndExit(status);
  }
  interface::Data<uint16_t> reduced_data(reduced_dataset);

  std::cout << "peos3" << std::endl;
  /* parse original queryset */
  success = interface::ParseDataset(files.query_file, original_queryset, status);
  if (success != 1) {
    interface::output::PrintErrorMessageAndExit(status);
  }
  interface::Data<uint8_t> original_queries(original_queryset);

  std::cout << "peos4" << std::endl;
  /* parse reduced queryset */
  success = interface::ParseDataset(files.reduced_query_file, reduced_queryset, status);
  if (success != 1) {
    interface::output::PrintErrorMessageAndExit(status);
  }
  interface::Data<uint16_t> reduced_queries(reduced_queryset);


  std::cout << "Number of images in original dataset: " << original_data.n << std::endl;
  std::cout << "Number of images in reduced dataset: " << reduced_data.n << std::endl;
  std::cout << "Number of images in original queryset: " << original_queries.n << std::endl;
  std::cout << "Number of images in reduced queryset: " << reduced_queries.n << std::endl;

  std::cout << std::endl;
  std::cout << "random entry from original dataset: " << +original_dataset.images[69][400] << '\n';
  std::cout << "random entry from reduced dataset: " << +reduced_dataset.images[69][3] << '\n';


  // ----------------------------      COMMENTED      ---------------------------- //
  // // Initialize Brute Force
  // BruteForce<uint8_t> bf = BruteForce<uint8_t>(data);
  //
  // // Calculate the window size (or set it to a default value)
  // double averageItemDistance = utils::averageDistance<uint8_t>(0.05, data.items, data.n, data.dimension);
  // int windowConstant = 1;
  // int windowSize = (int) windowConstant*averageItemDistance;
  // // int windowSize = 40000;
  // // std::cout << "Window Size: " << windowSize << '\n';
  //
  // // Initialize LSH
  // LSH<uint8_t> lsh = LSH<uint8_t>(lsh_input, data, windowSize);
  //
  //
  // /* get the query set and and output file, in case they are not provided by the command line parameters */
  // interface::ScanInput(files, status, false, files.query_file.empty(), files.output_file.empty());
  //
  //
  // /* keep iterating while there is a new queryset to perform queries on */
  // while (response) {
  //
  //   /* parse the query set */
  //   success = interface::ParseDataset(files.query_file, queries, status);
  //   /* check for potential errors or violations */
  //   if (success != 1) {
  //     interface::output::PrintErrorMessageAndExit(status);
  //   }
  //
  //   /* start building the output object */
  //   output.n = queries.number_of_images;
  //   output.method = "LSH";
  //   /* perform the queries for the brute force algorithm */
  //   bf.buildOutput(output, queries, lsh_input.N);
  //   /* perform the queries for the LSH algorithm */
  //   lsh.buildOutput(output, queries, lsh_input.N, lsh_input.R);
  //
  //   /* write the results to the specified output file */
  //   interface::output::writeOutput(files.output_file, output, status);
  //
  //   /* free the memory for the current query set */
  //   interface::freeDataset(queries);
  //
  //   /* ask the user if he/she/it (it's 2020, we don't judge) wants to repeat the experiment */
  //   std::cout << "Would you like to to repeat the experiment with a different query set and output file? (y/n)\n";
  //   /* variable to store the answer */
  //   std::string answer;
  //   std::cin >> answer;
  //
  //   /* check the response of the user */
  //   response = (answer == "y") || (answer == "Y") || (answer == "Yes") || (answer == "YES") || (answer == "yes");
  //
  //   /* if a positive response was given */
  //   if (response) {
  //     /* get the names of the new files */
  //     interface::ScanInput(files, status, false, true, true);
  //   }
  //
  // }
  // ----------------------------      COMMENTED      ---------------------------- //




  /* free the datasets and return, as we have finished */
  interface::freeDataset(original_dataset);
  interface::freeDataset(reduced_dataset);
  interface::freeDataset(original_queryset);
  interface::freeDataset(reduced_queryset);

  return 0;
}
