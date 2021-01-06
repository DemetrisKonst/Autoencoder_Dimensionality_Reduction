#include <iostream>

#include "../include/Clustering/clustering.hpp"



int main(int argc, char const *argv[])
{
  /* define useful variables */
  int success = 0;
  double duration = 0.0;
  interface::ExitCode status;
  interface::Dataset<uint8_t> original_dataset;
  interface::Dataset<uint16_t> reduced_dataset;
  interface::Clusterset clusterset;
  interface::IOCFiles files;
  interface::input::clustering::ClusteringConfig cluster_config;

  /* parse clustering input */
  success = interface::input::clustering::ClusteringParseInput(argc, argv, files, status);
  if (success != 1) interface::output::PrintErrorMessageAndExit(status);

  /* parse configuration file */
  success = interface::input::clustering::ClusteringParseConfigFile(files.configuration_file, cluster_config, status);
  if (success != 1) interface::output::PrintErrorMessageAndExit(status);

  /* parse original dataset and create a Data object */
  success = interface::ParseDataset(files.input_file, original_dataset, status);
  if (success != 1) interface::output::PrintErrorMessageAndExit(status);
  interface::Data<uint8_t> original_data(original_dataset);

  /* parse reduced dataset and create a Data object */
  success = interface::ParseDataset(files.reduced_input_file, reduced_dataset, status);
  if (success != 1) interface::output::PrintErrorMessageAndExit(status);
  interface::Data<uint16_t> reduced_data(reduced_dataset);

  /* parse clusterset */
  success = interface::ParseClusterset(files.labels_file, clusterset, cluster_config.clusters_K, status);
  if (success != 1) interface::output::PrintErrorMessageAndExit(status);


  // /* create a Clustering object in order to perform the clustering */
  // clustering::Clustering<uint8_t> cluster(cluster_config, data);
  // /* perform the clustering */
  // cluster.perform_clustering(data, cluster_input.algorithm, &duration);
  //
  // /* get the silhouette and print it */
  // double average_silhouette = cluster.compute_average_silhouette(data);
  //
  // /* create an Output object, build it and use it to log the results to the outfile */
  // interface::output::clustering::ClusteringOutput output;
  // cluster.build_output(output, data, cluster_input, duration);
  // interface::output::clustering::writeOutput(files.output_file, output, status);
  // cluster.free_output_object_memory(output);


  /* free up the allocated space and return */
  interface::freeDataset(original_dataset);
  interface::freeDataset(reduced_dataset);
  return 0;
}
