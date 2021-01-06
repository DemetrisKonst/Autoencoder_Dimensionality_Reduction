#include <iostream>

#include "../include/Clustering/clustering.hpp"



int main(int argc, char const *argv[])
{
  /* define useful variables */
  int success = 0;
  double duration1 = 0.0;
  double duration2 = 0.0;
  interface::ExitCode status;
  interface::Dataset<uint8_t> original_dataset;
  interface::Dataset<uint16_t> reduced_dataset;
  interface::Clusterset clusterset_from_reduced_data;
  interface::Clusterset clusterset_from_predicted_data;
  interface::IOCFiles files;
  interface::input::clustering::ClusteringConfig cluster_config;
  interface::output::clustering::ClusteringOutput<uint8_t> original_data_output;
  interface::output::clustering::ClusteringOutput<uint8_t> reduced_to_original_data_output;
  interface::output::clustering::ClusteringOutput<uint8_t> predicted_data_output;


  /* parse clustering command line input */
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
  success = interface::ParseClusterset(files.labels_file, clusterset_from_predicted_data, cluster_config.clusters_K, status);
  if (success != 1) interface::output::PrintErrorMessageAndExit(status);


  /* create clustering objects that will be used to perform the clusterings */
  clustering::Clustering<uint8_t> original_data_cluster(cluster_config, original_data);
  clustering::Clustering<uint16_t> reduced_data_cluster(cluster_config, reduced_data);

  /* perform the clusterings */
  original_data_cluster.perform_clustering(original_data, &duration1);
  reduced_data_cluster.perform_clustering(reduced_data, &duration2);

  /* build the clusterset that gets generated from the clustering in the reduced data space */
  reduced_data_cluster.build_clusterset(clusterset_from_reduced_data);

  /* now build the clusterings that get generated from the clustersets */
  clustering::Clustering<uint8_t> reduced_to_original_data_cluster(cluster_config, original_data, clusterset_from_reduced_data);
  clustering::Clustering<uint8_t> predicted_data_cluster(cluster_config, original_data, clusterset_from_predicted_data);

  /* compute the silhouettes */
  original_data_cluster.compute_silhouettes(original_data);
  reduced_to_original_data_cluster.compute_silhouettes(original_data);
  predicted_data_cluster.compute_silhouettes(original_data);

  /* build the output objects that will be used to write the outputs in the output file */
  original_data_cluster.build_output(original_data_output, original_data, "ORIGINAL SPACE", duration1, false);
  reduced_to_original_data_cluster.build_output(reduced_to_original_data_output, original_data, "NEW SPACE", duration2, false);
  predicted_data_cluster.build_output(predicted_data_output, original_data, "CLASSES AS CLUSTERS", 0.0, true);

  /* write to the outfile the data */
  interface::output::clustering::writeOutput(files.output_file, reduced_to_original_data_output, status);
  interface::output::clustering::writeOutput(files.output_file, original_data_output, status);
  interface::output::clustering::writeOutput(files.output_file, predicted_data_output, status);

  /* free up the space that the output objects allocated */
  original_data_cluster.free_output_object_memory(original_data_output);
  reduced_to_original_data_cluster.free_output_object_memory(reduced_to_original_data_output);
  predicted_data_cluster.free_output_object_memory(predicted_data_output);


  /* free up the allocated space and return */
  interface::freeDataset(original_dataset);
  interface::freeDataset(reduced_dataset);
  return 0;
}
