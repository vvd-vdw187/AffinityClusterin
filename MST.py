from argparse import ArgumentParser
from DataLoader import DataLoader
from AffinityClustering import AffinityClustering

def main(args):
    loader = DataLoader()
    datasets = loader.load_data()

    #TODO make input or something
    # Parameters
    beta = 0.2  # 0 <= beta <= 1 (buckets)
    alpha = 0.1  # shift of buckets
    gamma = 0.05  # shift of edge weights
    #TODO make plotter
    for dataset, params in datasets:
        print(dataset)
        V = [item for item in dataset[0]]
        print(V)
        k = len(V) - 1 

        AC = AffinityClustering()
        for i in range(10):
            if args.buckets:
                #TODO needs to make a new spark session for every iteration/ bucket
                # For now test without buckets
                E, size, vertex_coordinates, W = AC.create_distance_matrix(V, full_dm=True)
                E, W = AC.shift_edge_weights(E, gamma)
                # E, buckets, counter = create_buckets(E, alpha, beta, W)
                adjacency_list = AC.get_nearest_neighbours(E, k, buckets=True)
            else:
                adjacency_list = AC.get_nearest_neighbours(V, k, buckets=False)
            runs, graph, yhats, contracted_leader, mst = AC.affinity_clustering(adjacency_list, vertex_coordinates=None, plot_intermediate=False)
            print(f"Run {i} finished")
            print(f"Graph length: {len(graph)}")
            #STILL need to quit out of pyspark as well.
        break



if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--buckets', help="Use buckets [default=False]", action="store_true")
    args = parser.parse_args()

    main(args)