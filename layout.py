#!/usr/bin/env python3

if __name__ == '__main__':
    #Driver script adapted from https://github.com/HanKruiger/tsNET
    from argparse import ArgumentParser
    parser = ArgumentParser(description='Read a graph, and produce a layout with SMDS (output into a json file to be loaded with the webapp).')

    # Input
    parser.add_argument('input_graph')
    parser.add_argument('--opt_scale', '-s', type=int, default=0, help='Whether to optimize scaling parameter.')
    parser.add_argument('--epsilon', '-e', type=float, default=1e-7, help='Threshold for convergence.')
    parser.add_argument('--max_iter', '-m', type=int, default=500, help='Maximum number of iterations.')
    parser.add_argument('--learning_rate', '-l', type=str, default="convergent", help='Learning rate (hyper)parameter for optimization.')
    parser.add_argument('--output', '-o', type=str, help='Save layout to the specified file.')

    args = parser.parse_args()

    #Import needed libraries
    import os
    import time
    import graph_tool.all as gt

    #Import modules
    from modules.graph_functions import apsp
    from modules.SGD_MDS_sphere import SMDS

    #Check for valid input
    assert(os.path.isfile(args.input_graph))
    graph_name = os.path.splitext(os.path.basename(args.input_graph))[0]

    #Global hyperparameters
    max_iter = args.max_iter
    eps = args.epsilon
    opt_scale = args.opt_scale
    lr = args.learning_rate

    #Load input graph
    print('Reading graph: {0}...'.format(args.input_graph), end=' ', flush=True)
    G = gt.load_graph_from_csv(args.input_graph,hashed=False)
    print('Done.')

    print('Input graph: {0}, (|V|, |E|) = ({1}, {2})'.format(graph_name, G.num_vertices(), G.num_edges()))

    #Start timer
    start = time.perf_counter()

    #Get all-pairs-shortest-path matrix
    print('Computing SPDM...'.format(graph_name), end=' ', flush=True)
    d = apsp(G)
    print("Done.")

    #Perform optimization from SGD_MDS_sphere module
    X = SMDS(d,scale_heuristic = not args.opt_scale).solve(
        num_iter = args.max_iter,
        schedule = args.learning_rate,
        epsilon = args.epsilon
    )

    end = time.perf_counter()
    comp_time = end - start
    print('SMDS took {0:.2f} s.'.format(comp_time))

    print("-----------------------")

    #Save output to json
    from modules.graph_io import write_to_json
    if args.output: write_to_json(G,X,fname=args.output)
    else: write_to_json(G,X)
