import osqp
import numpy
import scipy.sparse as sparse

def QP(feature_size,sample_size,w_bar,mu_c,g1,h1):
    p1 = sparse.identity(feature_size+1)
    p2 = sparse.csc_matrix((feature_size+1,sample_size))
    p3 = sparse.csc_matrix((sample_size,feature_size+1))
    p4 = sparse.csc_matrix((sample_size,sample_size))
    P = sparse.bmat([[p1,p2],[p3,p4]])
    P = sparse.csc_matrix(P)
    
    q = numpy.vstack((-w_bar,mu_c))
    # q = sparse.vstack((-w_bar,mu_c))

    g2 = -sparse.identity(sample_size)
    G = sparse.bmat([ [g1,g2] ])
    G = sparse.csc_matrix(G)
   
    u = h1
    
    l = -numpy.inf*numpy.ones((sample_size,1), numpy.float32)
 
    prob = osqp.OSQP()
    prob.setup(P=P, q=q, A=G, l=l, u=u, verbose=False)
    a = prob.solve()
    return a.x
