def parseBibtexEntry(file):
    adjacency_list = []
    # Open the file and read the lines
    with open(file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or ':' not in line:
                continue
            # Splitting line by ":" to separate node from its neighbors
            node, neighbors = line.split(":")
            # Removing any whitespace
            node = int(node.strip())-1
            neighbor_nodes = neighbors.strip().split()
            # Adding the connections to the adjacency list
            for conn in neighbor_nodes:
                conn = int(conn)-1
                if conn > node:  # Ensuring only higher connections are stored
                    adjacency_list.append([node, conn])

    # Write adjacency list to a text file
    with open('adjacency_list.txt', 'w') as f_out:
        for node, neighbor in adjacency_list:
            f_out.write(f"{node},{neighbor}\n")

    return adjacency_list
parseBibtexEntry("datasets/hamiltonian/list_1_graphs (2).lst")
