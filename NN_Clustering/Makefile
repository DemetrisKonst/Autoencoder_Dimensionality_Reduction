CC=g++
CFLAGS=-O3

# used to compile all programs
ALL: CLUSTER SEARCH MANHATTAN

# clustering
CLUSTER: ClusterMain interface_utils.o
	$(CC) -o bin/cluster object/cluster_main.o object/interface_utils.o $(CFLAGS)

ClusterMain:
	$(CC) -c -o object/cluster_main.o ./cluster.cpp $(CFLAGS)

# search
SEARCH: SearchMain interface_utils.o
	$(CC) -o bin/search object/search_main.o object/interface_utils.o $(CFLAGS)

SearchMain:
	$(CC) -c -o object/search_main.o ./search.cpp $(CFLAGS)

# emd
MANHATTAN: MHMain interface_utils.o
	$(CC) -o bin/manhattan object/mh_main.o object/interface_utils.o $(CFLAGS)

MHMain:
	$(CC) -c -o object/mh_main.o ./manhattan.cpp $(CFLAGS)

# utilities
interface_utils.o:
	$(CC) -c -o object/interface_utils.o src/interface_utils.cpp $(CFLAGS)

# clean
clean:
	rm bin/* object/*.o
