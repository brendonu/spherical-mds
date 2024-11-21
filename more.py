import h5py
import numpy as np
import os
import json
import re
import time
from concurrent.futures import ProcessPoolExecutor
from convert import parseNpyEntry
from modules.graph_functions import count_intersection, apsp
from modules.SGD_MDS_sphere import SMDS
from modules.graph_io import write_to_json
import graph_tool.all as gt

# Define a base directory that can be adjusted as needed
BASE_DIR = os.path.abspath("/Users/brendonuzoigwe/spherical-mds")

def convert_js_to_json(js_file):
    """Convert a .js file with single quotes into valid JSON."""
    with open(js_file, 'r') as file:
        content = file.read()
        # Remove the variable declaration (if present) and replace single quotes with double quotes
        content = re.sub(r'^\s*G\s*=\s*', '', content, flags=re.MULTILINE)
        json_content = content.replace("'", '"')
    return json_content

def lat_long_to_cartesian(lat, long, radius=1.0):
    """Convert latitude and longitude to cartesian coordinates."""
    lat_rad = np.radians(lat)
    long_rad = np.radians(long)
    x = radius * np.cos(lat_rad) * np.cos(long_rad)
    y = radius * np.cos(lat_rad) * np.sin(long_rad)
    z = radius * np.sin(lat_rad)
    return [x, y, z]

def generate_layout(input_graph, opt_scale=0, epsilon=1e-7, max_iter=500, learning_rate="convergent", output=None):
    """Generate layout from an input graph using SMDS."""
    assert os.path.isfile(input_graph)
    graph_name = os.path.splitext(os.path.basename(input_graph))[0]

    # Load input graph
    print(f'Reading graph: {input_graph}...', end=' ', flush=True)
    G = gt.load_graph_from_csv(input_graph, hashed=False)
    print('Done.')

    print(f'Input graph: {graph_name}, (|V|, |E|) = ({G.num_vertices()}, {G.num_edges()})')

    # Start timer
    start = time.perf_counter()

    # Get all-pairs-shortest-path matrix
    print('Computing SPDM...', end=' ', flush=True)
    d = apsp(G)
    print("Done.")

    # Perform optimization
    X, _ = SMDS(d, scale_heuristic=not opt_scale).solve(
        num_iter=max_iter,
        schedule=learning_rate,
        epsilon=epsilon
    )

    print(f'SMDS took {time.perf_counter() - start:.2f} s.')

    # Save output to JSON
    if output:
        write_to_json(G, X, fname=output)
    else:
        write_to_json(G, X)

def process_single_file(file, data_dir, save_dir, label):
    """Process a single file, generating layout and saving to H5."""
    adjacency_list = parseNpyEntry(os.path.join(data_dir, file))
    
    # Run layout generation directly
    generate_layout('adjacency_list.txt', output=os.path.join(BASE_DIR, 'webapp', 'data.js'))

    # Read and convert JS file to JSON
    js_file = os.path.join(BASE_DIR, 'webapp', 'data.js')
    json_content = convert_js_to_json(js_file)
    G = json.loads(json_content)

    # Convert node positions
    positions = np.array([lat_long_to_cartesian(node['pos'][0], node['pos'][1]) for node in G['nodes']])

    # Collect all node positions initially
    graph_data = list(positions)
    num_nodes = len(G['nodes'])
    num_edges = len(G['edges'])

    # Ensure output has exactly 2048 points
    num_edge_points = 2048 - num_nodes
    points_per_edge = num_edge_points // num_edges if num_edges > 0 else 0
    extra_points = num_edge_points % num_edges if num_edges > 0 else 0

    # Create edge points
    first_edge = True
    for edge in G['edges']:
        node1 = int(edge['source'])
        node2 = int(edge['target'])
        if node1 < len(positions) and node2 < len(positions):
            point1 = positions[node1]
            point2 = positions[node2]

            edge_points = points_per_edge + (1 if first_edge and extra_points > 0 else 0)
            first_edge = False

            for t in np.linspace(0, 1, edge_points, endpoint=False):
                interpolated_point = point1 * (1 - t) + point2 * t
                graph_data.append(interpolated_point)

    # Ensure graph_data has exactly 2048 points
    while len(graph_data) < 2048:
        graph_data.append(graph_data[-1])  # Repeat last point to reach 2048
    graph_data = np.array(graph_data[:2048])

    # Save data to H5 file
    file_name = os.path.splitext(file)[0] + '.h5'
    file_path = os.path.join(save_dir, file_name)
    with h5py.File(file_path, 'w') as h5file:
        h5file.create_dataset('data', data=graph_data)
        h5file.create_dataset('labels', data=label)

def process_files_parallel(data_dir, save_dir, label):
    """Process all files in a directory in parallel."""
    files = os.listdir(data_dir)
    with ProcessPoolExecutor() as executor:
        for file in files:
            executor.submit(process_single_file, file, data_dir, save_dir, label)

# Define directories to process
data_dirs = [
    ('new_data/hamiltonian_small_mat', 'more_data/small/hamiltonian', 1),
    ('new_data/non_hamiltonian_small_mat', 'more_data/small/non_hamiltonian', 2),
    ('new_data/hamiltonian_medium_mat', 'more_data/medium/hamiltonian', 1),
    ('new_data/non_hamiltonian_medium_mat', 'more_data/medium/non_hamiltonian', 2),
]

# Process each directory in parallel
for data_dir, save_dir, label in data_dirs:
    process_files_parallel(os.path.join(BASE_DIR, data_dir), os.path.join(BASE_DIR, save_dir), label)
