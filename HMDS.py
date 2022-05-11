import numpy as np
#import tensorflow as tf
import math
import random
import itertools


from numba import jit

import pygraphviz
import graph_tool.all as gt
import io

acosh, cosh, sinh,sqrt = np.arccosh, np.cosh, np.sinh,np.sqrt

@jit(nopython=True)
def geodesic(u,v):
    return acosh( cosh(u[1])*cosh(v[0]-u[0])*cosh(v[1]) - sinh(u[1])*sinh(v[1]) )

@jit(nopython=True)
def grad(u,v):
    a,b = u
    c,d = v
    bottom = sqrt( ( sinh(b)*sinh(d) - cosh(b)*cosh(d)*cosh(a-c)) ** 2 - 1 )

    da = sinh(a-c) * cosh(b) * cosh(d)
    db = -( -sinh(b) * cosh(d) * cosh(a-c) + sinh(d) * cosh(b) )
    dc = -( sinh(a-c) * cosh(b) * cosh(d) )
    dd = -sinh(b) * cosh(d) + sinh(d) * cosh(b) * cosh(a-c)

    grad = np.array([   [da,db],
                        [dc,dd] ]) / bottom
    return np.minimum(grad,np.ones(grad.shape)*10)

@jit(nopython=True)
def satisfy(u,v,d,w,step):
    """
    u,v: hyperbolic vectors
    d: ideal distance between u and v from shortest path matrix
    w: associated weight of the pair u,v
    step: Fraction of distance u and v should be moved along gradient
    Returns: updated hyperbolic vectors of u and v

    Code modified from https://github.com/jxz12/s_gd2 and associated paper.
    """

    wc = step / d**2
    wc = 0.1 if wc > 0.1 else wc

    g = 2*(geodesic(u,v)-d) * grad(u,v)
    m = wc*g

    return u-m[0], v-m[1]

@jit(nopython=True)
def step_func1(count):
    return 1/(5+count)

@jit(nopython=True)
def calc_stress(X,d,w):
    """
    Calculates the standard measure of stress: \sum_{i,j} w_{i,j}(dist(Xi,Xj)-D_{i,j})^2
    Or, in English, the square of the difference of the realized distance and the theoretical distance,
    weighted by the table w, and summed over all pairs.
    """
    stress = 0
    for i in range(len(X)):
        for j in range(i):
            stress += (1/pow(d[i][j],2))*pow(geodesic(X[i],X[j])-d[i][j],2)
    return pow(stress,0.5)

@jit(nopython=True)
def choose1(n,k):
    product = 1
    for i in range(1,k+1):
        product *= (n-(k-1))/i
    return product

@jit(nopython=True)
def calc_distortion(X,d):
    distortion = 0
    for i in range(len(X)):
        for j in range(i):
            distortion += abs((geodesic(X[i],X[j])-d[i][j]))/d[i][j]
    return (1/choose1(len(X),2))*distortion

@jit(nopython=True)
def stoch_solver(X,d,w,indices,schedule,num_iter=15,epsilon=1e-3):
    step = schedule[0]
    shuffle = random.shuffle

    for count in range(num_iter):

        for i,j in indices: # Random pair
            X[i],X[j] = satisfy(X[i],X[j],d[i][j],w[i][j],step) #Gradient w.r.t. pair i and j

        step = schedule[count] if count < len(schedule) else schedule[-1] #Get next step size
        #step = 1/(count+1)
        shuffle(indices) #Shuffle pair order
        #print(calc_stress(X,d,w))
        #print(calc_stress(X,d,w))



    return X

@jit(nopython=True)
def stoch_solver_debug(X,d,w,indices,schedule,num_iter=15,epsilon=1e-3):
    step = 1
    shuffle = random.shuffle
    print(schedule)
    yield X.copy()
    for count in range(num_iter):
        for i,j in indices: # Random pair
            X[i],X[j] = satisfy(X[i],X[j],d[i][j],w[i][j],step) #Gradient w.r.t. pair i and j

        step = schedule[count] if count <= len(schedule) else schedule[-1]/10 #Get next step size

        shuffle(indices) #Shuffle pair order
        print(calc_stress(X,d,w))
        yield X.copy()

    return X

@jit(nopython=True)
def set_step(w_max,eta_max,eta_min):
    # a = 1/w_max
    # b = np.log(eta_min/eta_max)/(15-1)
    # step = lambda count: eta_max*np.exp(b*count)
    # step = lambda count: a/pow(b+count,0.5)

    lamb = np.log(eta_min/eta_max)/(15-1)
    step = lambda count: eta_max*np.exp(lamb*count)

    return np.array([step(count) for count in range(15)])

def schedule_convergent(d,t_max,eps,t_maxmax):
    w = np.divide(np.ones(d.shape),d**2,out=np.zeros_like(d), where=d!=0)
    w_min,w_max = np.amin(w,initial=10000,where=w > 0), np.max(w)

    eta_max = 1.0 / w_min
    eta_min = eps / w_max

    lamb = np.log(eta_max/eta_min) / (t_max-1)

    # initialize step sizes
    etas = np.zeros(t_maxmax)
    eta_switch = 1.0 / w_max
    for t in range(t_maxmax):
        eta = eta_max * np.exp(-lamb * t)
        if (eta < eta_switch): break

        etas[t] = eta

    tau = t
    for t in range(t,t_maxmax):
        eta = eta_switch / (1 + lamb*(t-tau))
        etas[t] = eta
        #etas[t] = 1e-7

    #etas = [eps for t in range(t_maxmax)]
    #print(etas)
    return np.array(etas)

def preprocess(graph,input_format='dot'):
    graph_file = io.StringIO(pygraphviz.AGraph(graph).to_string())
    G = gt.load_graph(graph_file,fmt='dot')
    d = distance_matrix(G)
    return G,d

def postprocess(G,embedding):
    pos = G.new_vp('vector<float>')
    pos.set_2d_array(embedding.T)
    G.vertex_properties['pos'] = pos

    import tempfile
    with tempfile.TemporaryFile() as file:
        G.save(file,fmt='dot')
        file.seek(0)
        dot_rep = file.read()
    return gt_to_json(G,embedding), dot_rep

def gt_to_json(G,embedding):
    nodes, edges = G.iter_vertices(),G.iter_edges()

    out = {"nodes": [None for i in range(G.num_vertices())],
            "edges": [None for i in range(G.num_edges())]
            }

    sinh, cosh = np.sinh, np.cosh
    for v in nodes:
        x,y = embedding[int(v)]
        Rh = np.arccosh(cosh(x)*cosh(y))
        Re = (np.exp(Rh)-1)/(np.exp(Rh)+1)
        theta = 2*np.arctan( sinh(y) / ( sinh(x)*cosh(y) + np.sqrt( pow(cosh(x),2) * pow(cosh(y),2) - 1 ) ) )
        pos = [Re * np.cos(theta), Re * np.sin(theta)]
        if pow(pos[0],2) + pow(pos[1],2) >= 1:
            print("yell")

        out["nodes"][int(v)] = {
            "id": int(v),
            "pos": pos
        }
    count = 0
    for u,v in edges:
        ##Implement map or zip or something
        out["edges"][count] = {
            "s": int(u),
            "t": int(v)
        }
        count += 1

    return out


def optimize_scale(X,d,w,num_iter,until_conv,indices,schedule):
    from scipy.optimize import minimize_scalar
    init_X = X.copy()
    def stress_at(a):
        new_X = stoch_solver(X.copy(),d*a,w,indices,schedule,num_iter=num_iter)
        return calc_distortion(new_X,d*a)

    opt_alpha = minimize_scalar(stress_at,bounds=(0.1,10),method='bounded')
    print(opt_alpha.x)
    return stoch_solver(X.copy(),d*opt_alpha.x,w,indices,schedule,num_iter=num_iter)



class HMDS:
    def __init__(self,dissimilarities,
                 opt_scale=False,
                 scaling_factor=0,
                 init_pos=None
                 ):
        self.d = dissimilarities
        self.opt_scale = opt_scale

        self.d_max = np.max(dissimilarities)

        # self.d = self.d*(2*math.pi/self.d_max)
        self.d_min = 1
        self.n = len(self.d)
        # if self.n > 20:
        #     self.d = self.d*(10/self.d_max)
        if init_pos: #If an initial configuration is desired, assign it
            self.X = init_pos
            if self.X.shape[0] != self.n:
                raise Exception("Number of elements in starting configuration must be equal to the number of elements in the dissimilarity matrix.")
        else: #Random point in the chosen geometry
            self.X = [[0,0] for i in range(self.n)]
            for i in range(len(self.d)):
                self.X[i] = self.init_point()
            self.X = np.asarray(self.X,dtype="float64")

        #Weight is inversely proportional to the square of the theoretic distance
        self.w = np.array([[1/pow(self.d[i][j],2) if self.d[i][j] > 0 else 0 for i in range(self.n)]
                            for j in range(self.n)])

        #Values for step size calculation
        w_min = 1/pow(self.d_max,2)
        self.w_max = 1/pow(self.d_min,2)

        self.eta_max = 1/w_min
        self.eta_min = 0.1/self.w_max

        self.indices = np.array(list(itertools.combinations(range(self.n), 2)))

        self.steps = schedule_convergent(self.d, 30, 0.01, 200)


    def solve(self,num_iter=20,debug=False,until_conv=False):
        X = self.X
        d = self.d
        w = self.w
        if self.opt_scale:
            return optimize_scale(X,d,w,num_iter,until_conv,self.indices,self.steps)
        if debug:
            solve_step = stoch_solver_debug(X,d,w,self.indices,self.steps,num_iter=num_iter)
            #print(next(solve_step))
            Xs = [x for x in solve_step]
            self.stress_hist = [calc_stress(x,d,w) for x in Xs]
            self.X =  Xs[-1]
            return

        X = stoch_solver(self.X,self.d,self.w,self.indices,self.steps,num_iter=num_iter)
        self.X = X
        return X

    def calc_stress3(self):
        """
        Calculates the standard measure of stress: \sum_{i,j} w_{i,j}(dist(Xi,Xj)-D_{i,j})^2
        Or, in English, the square of the difference of the realized distance and the theoretical distance,
        weighted by the table w, and summed over all pairs.
        """
        stress = 0
        for i in range(self.n):
            for j in range(i):
                stress += self.w[i][j]*pow(geodesic(self.X[i],self.X[j])-self.d[i][j],2)
        return pow(stress,0.5)

    def calc_stress2(self):
        """
        Calculates the standard measure of stress: \sum_{i,j} w_{i,j}(dist(Xi,Xj)-D_{i,j})^2
        Or, in English, the square of the difference of the realized distance and the theoretical distance,
        weighted by the table w, and summed over all pairs.
        """
        stress = 0
        for i in range(self.n):
            for j in range(i):
                stress += self.w[i][j]*pow(geodesic(self.X[i],self.X[j])-self.d[i][j],2)
        bottom = 0
        for i in range(self.n):
            for j in range(i):
                bottom += self.d[i][j] ** 2
        return stress/bottom

    def calc_distortion1(self):
        """
        A normalized goodness of fit measure.
        """
        distortion = 0
        for i in range(self.n):
            for j in range(i):
                distortion += abs((geodesic(self.X[i],self.X[j])-self.d[i][j]))/self.d[i][j]
        return (1/choose(self.n,2))*distortion


    def compute_step_size_old(self,count,num_iter):
        lamb = math.log(self.eta_min/self.eta_max)/(num_iter-1)
        return self.eta_max*math.exp(lamb*count)

    def compute_step_size(self,count,num_iter):
        a = 1/self.w_max
        b = -math.log(self.eta_min/self.eta_max)/(num_iter-1)
        return a/(pow(1+b*count,0.5))

    def init_point(self):
            r = pow(random.uniform(0,1),0.5)
            theta = random.uniform(0,2*math.pi)
            x = math.atanh(math.tanh(r)*math.cos(theta))
            y = math.asinh(math.sinh(r)*math.sin(theta))
            return [x,y]


def normalize(v):
    mag = pow(sum([val*val for val in v]), 0.5)
    return np.array([val/mag for val in v])



def lob_dist(xi,xj):
    x1,y1 = xi
    x2,y2 = xj
    dist = np.arccosh(np.cosh(y1)*np.cosh(x2-x1)*np.cosh(y2)-np.sinh(y1)*np.sinh(y2))
    return dist

def scale_matrix(d,new_max):
    d_new = np.zeros(d.shape)

    t_min,r_min,r_max,t_max = 0,0,np.max(d),new_max

    for i in range(d.shape[0]):
        for j in range(i):
            m = ((d[i][j]-r_min)/(r_max-r_min))*(t_max-t_min)+t_min
            d_new[i][j] = m
            d_new[j][i] = m
    return d_new



#Code taken from **link t-SNET Github
def get_shortest_path_distance_matrix(g, k=10, weights=None):
    # Used to find which vertices are not connected. This has to be this weird,
    # since graph_tool uses maxint for the shortest path distance between
    # unconnected vertices.
    def get_unconnected_distance():
        g_mock = gt.Graph()
        g_mock.add_vertex(2)
        shortest_distances_mock = gt.shortest_distance(g_mock)
        unconnected_dist = shortest_distances_mock[0][1]
        return unconnected_dist

    # Get the value (usually maxint) that graph_tool uses for distances between
    # unconnected vertices.
    unconnected_dist = get_unconnected_distance()

    # Get shortest distances for all pairs of vertices in a NumPy array.
    X = gt.shortest_distance(g, weights=weights).get_2d_array(range(g.num_vertices()))

    if len(X[X == unconnected_dist]) > 0:
        print('[distance_matrix] There were disconnected components!')

    # Get maximum shortest-path distance (ignoring maxint)
    X_max = X[X != unconnected_dist].max()

    # Set the unconnected distances to k times the maximum of the other
    # distances.
    X[X == unconnected_dist] = k * X_max

    return X


# Return the distance matrix of g, with the specified metric.
def distance_matrix(g, distance_metric='shortest_path', normalize=False, k=10.0, verbose=True, weights=None):

    if verbose:
        print('[distance_matrix] Computing distance matrix (metric: {0})'.format(distance_metric))

    if distance_metric == 'shortest_path' or distance_metric == 'spdm':
        X = get_shortest_path_distance_matrix(g, weights=weights)
    elif distance_metric == 'modified_adjacency' or distance_metric == 'mam':
        X = get_modified_adjacency_matrix(g, k)
    else:
        raise Exception('Unknown distance metric.')

    # Just to make sure, symmetrize the matrix.
    X = (X + X.T) / 2

    # Force diagonal to zero
    X[range(X.shape[0]), range(X.shape[1])] = 0

    # Normalize matrix s.t. max is 1.
    if normalize:
        X /= np.max(X)
    if verbose:
        print('[distance_matrix] Done!')

    return X
