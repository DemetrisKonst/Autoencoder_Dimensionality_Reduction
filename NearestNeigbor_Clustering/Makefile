CC=g++
CFLAGS=-O3

# used to compile all programs
ALL: CLUSTER SEARCH

# compile clustering
CLUSTER: ClusterMain interface_utils.o
	$(CC) -o bin/cluster object/ClusterMain.o object/interface_utils.o $(CFLAGS)

ClusterMain:
	$(CC) -c -o object/ClusterMain.o Clustering/ClusterMain.cpp $(CFLAGS)

# compile search
SEARCH: SearchMain interface_utils.o
	$(CC) -o bin/lsh object/search_main.o object/interface_utils.o $(CFLAGS)

SearchMain:
	$(CC) -c -o object/search_main.o NearestNeighbor/search.cpp $(CFLAGS)

interface_utils.o:
	$(CC) -c -o object/interface_utils.o src/interfaces/interface_utils.cpp $(CFLAGS)

clean:
	rm bin/* object/*.o
