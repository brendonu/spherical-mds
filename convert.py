import numpy as np

def parseNpyEntry(file):
    adjacency_list = []
    
    data = np.load(file)

    for i in range(data.shape[0]):  
        for j in range(i + 1, data.shape[1]):  
            if data[i][j] == 1:  
                adjacency_list.append([i, j])

    # Write adjacency list to a text file
    with open('adjacency_list.txt', 'w') as f_out:
        f_out.truncate(0)
        for node, neighbor in adjacency_list:
            f_out.write(f"{node},{neighbor}\n")

    return adjacency_list

# Example usage
parseNpyEntry("new_data/hamiltonian_medium_mat/hamiltonian_5909.npy")

