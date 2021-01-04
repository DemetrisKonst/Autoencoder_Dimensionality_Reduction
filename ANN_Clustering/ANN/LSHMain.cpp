#include <iostream>
#include <chrono>
#include <utility>

#include "../include/interfaces/LSH_interface.h"
#include "../include/BruteForce/BruteForce.hpp"
#include "../include/LSH/LSH.hpp"
#include "../include/utils/ANN.hpp"

int main(int argc, char const *argv[]) {

  /* define useful variables */
  int success = 0;
  interface::ExitCode status;
  interface::Dataset<uint8_t> original_dataset;
  interface::Dataset<uint16_t> reduced_dataset;
  interface::Dataset<uint8_t> original_queryset;
  interface::Dataset<uint16_t> reduced_queryset;
  interface::IOFiles files;
  interface::input::LSH::LSHInput lsh_input;
  interface::output::SearchOutput output;


  /* parse LSH input */
  success = interface::input::LSH::LSHParseInput(argc, argv, lsh_input, files, status);
  /* check for potential errors or violations */
  if (success != 1) {
    interface::output::PrintErrorMessageAndExit(status);
  }

  /* parse original dataset */
  success = interface::ParseDataset(files.input_file, original_dataset, status);
  if (success != 1) {
    interface::output::PrintErrorMessageAndExit(status);
  }
  interface::Data<uint8_t> original_data(original_dataset);

  /* parse reduced dataset */
  success = interface::ParseDataset(files.reduced_input_file, reduced_dataset, status, true);
  if (success != 1) {
    interface::output::PrintErrorMessageAndExit(status);
  }
  interface::Data<uint16_t> reduced_data(reduced_dataset);

  /* parse original queryset */
  success = interface::ParseDataset(files.query_file, original_queryset, status);
  if (success != 1) {
    interface::output::PrintErrorMessageAndExit(status);
  }
  interface::Data<uint8_t> original_queries(original_queryset);

  /* parse reduced queryset */
  success = interface::ParseDataset(files.reduced_query_file, reduced_queryset, status, true);
  if (success != 1) {
    interface::output::PrintErrorMessageAndExit(status);
  }
  interface::Data<uint16_t> reduced_queries(reduced_queryset);

  // Initialize Brute Force for original and reduced datasets
  BruteForce<uint8_t> original_bf = BruteForce<uint8_t>(original_data);
  BruteForce<uint16_t> reduced_bf = BruteForce<uint16_t>(reduced_data);

  // Calculate the window size for LSH (or set it to a default value)
  double averageItemDistance = utils::averageDistance<uint8_t>(0.05, original_data.items, original_data.n, original_data.dimension);
  int windowConstant = 1;
  int windowSize = (int) windowConstant*averageItemDistance;

  // Initialize LSH for original
  LSH<uint8_t> original_lsh = LSH<uint8_t>(lsh_input, original_data, windowSize);

  /* start building the output object */
  output.n = original_queryset.number_of_images;
  /* perform the queries for the brute force algorithms */
  original_bf.buildOutput(output, original_queryset, true);
  reduced_bf.buildOutput(output, reduced_queryset, false);
  /* perform the queries for the LSH algorithm */
  original_lsh.buildOutput(output, original_queryset);
  /* calculate the original distances and the approximation factors */
  utils::calculateDistances<uint8_t>(output, original_data, original_queries);
  /* write the results to the specified output file */
  interface::output::writeOutput(files.output_file, output, status);

  /* free the datasets and return, as we have finished */
  interface::freeDataset(original_dataset);
  interface::freeDataset(reduced_dataset);
  interface::freeDataset(original_queryset);
  interface::freeDataset(reduced_queryset);

  return 0;
}
