#include <iostream>
#include <chrono>
#include <utility>

#include "./include/interfaces/mh_interface.hpp"
#include "./include/BruteForce/BruteForce.hpp"
#include "./include/utils/manhattan.hpp"

int main(int argc, char const *argv[]) {

  /* define useful variables */
  int success = 0;
  interface::ExitCode status;
  interface::Dataset<uint8_t> dataset;
  interface::Dataset<uint8_t> queryset;
  interface::Labelset<uint8_t> dataset_labels;
  interface::Labelset<uint8_t> queryset_labels;
  interface::IOFilesMH files;
  // interface::output::SearchOutput output;


  /* parse LSH input */
  success = interface::input::MH::MHParseInput(argc, argv, files, status);
  /* check for potential errors or violations */
  if (success != 1) {
    interface::output::PrintErrorMessageAndExit(status);
  }

  /* parse dataset */
  success = interface::ParseDataset(files.input_file, dataset, status);
  if (success != 1) {
    interface::output::PrintErrorMessageAndExit(status);
  }
  interface::Data<uint8_t> data(dataset);

  /* parse queryset */
  success = interface::ParseDataset(files.query_file, queryset, status);
  if (success != 1) {
    interface::output::PrintErrorMessageAndExit(status);
  }
  interface::Data<uint8_t> queries(queryset);

  /* parse dataset labels */
  success = interface::ParseLabelset(files.input_labels_file, dataset_labels, status);
  if (success != 1) {
    interface::output::PrintErrorMessageAndExit(status);
  }

  /* parse queryset labels */
  success = interface::ParseLabelset(files.query_labels_file, queryset_labels, status);
  if (success != 1) {
    interface::output::PrintErrorMessageAndExit(status);
  }

  // Initialize Brute Force for original and reduced datasets
  BruteForce<uint8_t> bf = BruteForce<uint8_t>(data);
  std::vector<std::vector<std::pair<int, Item<uint8_t>*>>> mneighbors = bf.getNeighbors(queryset, 10, 1);

  double mavg = utils::evaluate(mneighbors, dataset_labels, queryset_labels, 10);

  interface::output::MH::writeOutput(files.output_file, mavg, status);

  /* free the datasets and return, as we have finished */
  interface::freeDataset(dataset);
  interface::freeDataset(queryset);
  interface::freeLabelset(dataset_labels);
  interface::freeLabelset(queryset_labels);

  return 0;
}
