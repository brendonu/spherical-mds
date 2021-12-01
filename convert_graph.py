import networkx as nx

G = nx.icosahedral_graph()
nx.write_graphml(G,'graphs/isocahedron.xml')

H = nx.dodecahedral_graph()
nx.write_graphml(H,'graphs/dodecahedron.xml')
