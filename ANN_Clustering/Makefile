CC=g++
CFLAGS=-O3

# used to compile all programs
# ALL: CLUSTER LSH
ALL: LSH

# compile clustering
CLUSTER: ClusterMain clusteringInterface.o interface_utils.o
	$(CC) -o bin/cluster object/ClusterMain.o object/clustering_interface.o object/interface_utils.o $(CFLAGS)

ClusterMain:
	$(CC) -c -o object/ClusterMain.o Clustering/ClusterMain.cpp $(CFLAGS)

clusteringInterface.o:
	$(CC) -c -o object/clustering_interface.o src/interfaces/clustering_interface.cpp $(CFLAGS)

# compile LSH
LSH: LSHMain interface_utils.o
	$(CC) -o bin/lsh object/LSHMain.o object/interface_utils.o $(CFLAGS)

LSHMain:
	$(CC) -c -o object/LSHMain.o ANN/LSHMain.cpp $(CFLAGS)

interface_utils.o:
	$(CC) -c -o object/interface_utils.o src/interfaces/interface_utils.cpp $(CFLAGS)

clean:
	rm bin/* object/*.o
