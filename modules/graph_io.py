import os
import numpy as np
import graph_tool.all as gt

def write_to_json(G,X,fname='webapp/data.js',name_map = None,classes = None):

    name = lambda v: name_map[v] if name_map else v

    json_str = "G = { \n    \'nodes\': [\n"

    for v in G.iter_vertices():
        if classes:
            json_str += "       {{\'id\': \'{v_id}\', \'pos\': {v_pos}, \'class\': {c}}}".format(c= classes[v],v_id=name(v),v_pos=np.array2string(np.flip(X[v]),separator=','))
        else:
            json_str += "       {{\'id\': \'{v_id}\', \'pos\': {v_pos}}}".format(v_id=name(v),v_pos=np.array2string(np.flip(X[v]),separator=','))
        if v != G.num_vertices()-1:
            json_str += ",\n"
    json_str += "\n     ],\n    \'edges\': [\n"

    for i, (e1,e2) in enumerate(G.iter_edges()):
        json_str += "       {{\'source\': \'{s_id}\', \'target\': \'{t_id}\'}}".format(s_id=name(e1),t_id=name(e2))
        if i != G.num_edges()-1:
            json_str += ",\n"
    json_str += "\n     ]\n}"


    with open(fname, "w") as text_file:
        text_file.write(json_str)
    text_file.close()

def load_graph(in_file):
    extension = os.path.splitext(in_file)[1]
    if extension == '.vna':
        g = load_vna(in_file)
    ##Jacob Additon
    elif extension == '.dot':
        g = gt.load_graph(in_file)
        g.set_directed(False)
    ##End Jacob addition
    return g


def save_graph(out_file, g):
    extension = os.path.splitext(out_file)[1]
    if extension == '.vna':
        save_vna(out_file, g)
    ##Jacob addition
    elif extension == '.dot':
        g.save(out_file)


def load_vna(in_file):
    with open(in_file) as f:
        all_lines = f.read().splitlines()

        it = iter(all_lines)

        # Ignore preamble
        line = next(it)
        while not (line.lower().startswith('*node properties') or line.lower().startswith('*node data')):
            line = next(it)

        node_properties = next(it).split(' ')
        node_properties = [word.lower() for word in node_properties]
        assert('id' in node_properties)

        vertices = dict()
        line = next(it)
        gt_idx = 0 # Index for gt
        while not line.startswith('*'):
            entries = line.split(' ')
            vna_id = entries[0]
            vertex = dict()
            for i, prop in enumerate(node_properties):
                vertex[prop] = entries[i]
            vertex['id'] = gt_idx # Replace VNA ID by numerical gt index
            vertices[vna_id] = vertex # Retain VNA ID as key of the vertices dict

            gt_idx += 1
            line = next(it)

        # Skip node properties, if any
        while not (line.lower().startswith('*tie data')):
            line = next(it)

        edge_properties = next(it).split(' ')
        assert(edge_properties[0] == 'from' and edge_properties[1] == 'to')

        edges = []
        try:
            while True:
                line = next(it)
                entries = line.split(' ')
                v_i = vertices[entries[0]]['id']
                v_j = vertices[entries[1]]['id']
                edges.append((v_i, v_j))
        except StopIteration:
            pass

        g = gt.Graph(directed=False)
        g.add_vertex(len(vertices))
        for v_i, v_j in edges:
            g.add_edge(v_i, v_j)

        gt.remove_parallel_edges(g)

        return g
    return None


def save_vna(out_file, g):
    with open(out_file, 'w') as f:
        f.write('*Node data\n')
        f.write('ID\n')
        for v in g.vertices():
            f.write('{0}\n'.format(int(v)))
        f.write('*Tie data\n')
        f.write('from to strength\n')
        for v1, v2 in g.edges():
            f.write('{0} {1} 1\n'.format(int(v1), int(v2)))
        f.close()
