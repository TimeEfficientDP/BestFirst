from pqdict import PQDict
import numpy as np 
from collections import defaultdict 
import logging



def mint(G, start, T):
    

    '''
    Mint algorithm
    Parameters: 
        G: graph  (graph)
        start: start node (int, str) 
        y: observation sequence (array) 
    
    Returns: 
        d: optimal costs (dict)
        P[k] + [k]: path (list)
    
    '''
        
    D = defaultdict(float) 
    for state in start:
        D[(state, 0)] = 0  # mapping of nodes to their dist from start
    Q = PQDict(D)           # priority queue for tracking min shortest path
    P = defaultdict(list)                # mapping of nodes to their direct predecessors
    visited = set() 
    success=False
        
    while Q:             
        (k, d) = Q.popitem()                # pop node w min dist d on frontier in constant time 
        D[k]=d                              # est dijkstra greedy score
        visited.add(k)                      # remove from unexplored
        if k[1] == T-1: 
            success=True 
            break                 
        # now consider the edges from v with an unexplored head -
        # we may need to update the dist of unexplored successors 
        for w in G.adj[k[0]]:                          # successors to vù
            if ( w , k[1]+1 ) not in visited:        
                # then w is a frontier node
                new_d = D[k] - G.adj[k[0]][w]  - G.emission_probabilities[ k[1]+1 ][ (k[0], w) ]      # dgs: dist of start -> v -> w
                if new_d < Q.get((w,k[1]+1), float("inf")):
                    Q[(w, k[1]+1)] = new_d  
                    P[(w, k[1]+1)] = P[k] + [k]  
    if not success:
        logging.warning('Algorithm terminated before last frame was reached.')

    #r =  len(visited) / (len(G.nodes) * len(y))
    return D[k], P[k] + [k] #, r




def mint_bound(G, start, T):
    
    '''
    Mint-bound algorithm
    Parameters: 
        G: graph  (graph)
        start: start node (int, str) 
        y: observation sequence (array) 
    
    Returns: 
        d: optimal costs (dict)
        P[k] + [k]: path (list)    
    
    '''
    
    D = defaultdict(dict) 
    for state in start: 
        D[(state, 0)] = -G.upper_bound * T         # mapping of nodes to their dist from start
    Q = PQDict(D)           # priority queue for tracking min shortest path
    P = defaultdict(list)                # mapping of nodes to their direct predecessors
    visited = set() 
    success=False
    
    
    while Q:                                 # nodes yet to explore
        (k, d) = Q.popitem()                 # pop node w min dist d on frontier in constant time 
        D[k]= d                              # est dijkstra greedy score
        visited.add(k)  

        # remove from unexplored
        if k[1] == T-1: 
            success=True 
            break                

        # now consider the edges from v with an unexplored head -
        # we may need to update the dist of unexplored successors 
        for w in G.adj[k[0]]:                          # suP_Viterbiccessors to v
            if ( w, k[1]+1 ) not in visited:                         # then w is a frontier node
                new_d = D[k] + G.upper_bound - G.adj[k[0]][w]  - G.emission_probabilities[ k[1]+1 ][(k[0], w)]        # dgs: dist of start -> v -> w

                if new_d < Q.get((w,k[1]+1), float("inf")):
                    Q[(w, k[1]+1)] = new_d              # set/update dgs
                    P[(w, k[1]+1)] = P[k] + [k]   
                 
    if not success:
        logging.warning('Algorithm terminated before last frame was reached.')
   
    
    return D[k],  P[k] + [k]





def viterbi_tp(G, start, T):
    """
    Viterbi algorithm - push (token-passing) implementation
    Parameters:
        G: graph  (graph)
        start: start node (int, str) 
        y: observation sequence (array) 
    
    Returns: 
        [G.nodes[i] for i in x]: optimal paths (dict)
        T1; path probability table (array) 
    """
    
    K = len(G.nodes)
    T1 = np.zeros((K, T))
    T2 = np.zeros((K, T))

    # Initilaize the tracking tables from first observation
    #print("start " + str(start)) 
    Pi = np.array([float("-inf") if it not in start else 0 for it in G.nodes])
    
    #print(max(Pi))
    for idx in range(len(Pi)): 
        T1[idx, 0] = Pi[idx] 
        T2[idx, 0] = 0 
    # Iterate through the observations updating the tracking tables
    for j in range(1, T): 
        
        this_j_T1 = [float("-inf")  for _ in range(K)] 
        this_j_T2 = [float("-inf")  for _ in range(K)] 
        
        for i in range(K): 
            node_i = G.nodes[i]  # this seems to be the only difference ??!!! 
            for neig in G.adj[node_i]: 
                this_likelihood = T1[i,j-1]  + G.adj[node_i][neig] + G.emission_probabilities[ j ][ (node_i, neig) ]   
                
                
                if this_likelihood > this_j_T1[G.node2idx[neig]]:      
                    this_j_T1[G.node2idx[neig]] = this_likelihood
                    this_j_T2[G.node2idx[neig]] = i  
            
        T1[:,j] = this_j_T1
        T2[:,j] = this_j_T2
        
    # Build the output, optimal model trajectory by Backtracking 
    x = np.zeros(T, dtype=int)
    
    x[-1] = int(np.argmax(T1[:, T - 1]))
    
    for i in reversed(range(1, T)):
      
        x[i - 1] = T2[x[i], i]
    
    
    return [G.nodes[i] for i in x], T1




def bidirectional_mint(G, start, T, last_nodes):
    
    
    '''
    bidirectional-mint algorithm
    Parameters: 
        G: graph  (graph)
        start: start node (int, str) 
        y: observation sequence (array) 
        last_nodes: possible last states (e.g., set of nodes reachable in T steps) 
    
    Returns: 
        best path: path (list)
        mu: best path cost (float)
    '''
    
    
    D_f = defaultdict(float) 
    D_b = defaultdict(float) 
    for state in start: 
        D_f[(state, 0)] = 0 
    for node in last_nodes:
        D_b[(node, T-1)] = 0 
    
    Q_f = PQDict(D_f)           # priority queue for tracking min shortest path
    Q_b = PQDict(D_b)
    P_f = defaultdict(list)                # mapping of nodes to their direct predecessors
    P_b = defaultdict(list)  
    visited_f = set()       # unexplored node
    visited_b = set() 
    
    mu = float("inf") 
    while len(Q_f)>0 and len(Q_b)>0:                             # nodes yet to explore
        (k_f, d_f) = Q_f.popitem() 
        (k_b, d_b) = Q_b.popitem()                    # pop node w min dist d on frontier in constant time 
        D_f[k_f]=  d_f 
        D_b[k_b] = d_b      
        # update explored 
        visited_f.add(k_f)
        visited_b.add(k_b)
        
        if k_f[1] < T-1: 
            for w in G.adj[k_f[0]]:                          # successors to v
                if ( w , k_f[1]+1 ) not in visited_f:        
                    d = D_f[k_f] - G.adj[k_f[0]][w]  - G.emission_probabilities[ k_f[1] + 1 ][(k_f[0], w)]       # dgs: dist of start -> v -> w
                    if d < Q_f.get((w,k_f[1]+1), float("inf")):
                        Q_f[(w, k_f[1]+1)] = d     
                        P_f[(w, k_f[1]+1)] = P_f[k_f] + [k_f] 
                                       
                
                
                candidate_mu = D_f[k_f] - G.adj[k_f[0]][w]  + D_b[( w , k_f[1]+1 )] - G.emission_probabilities[ k_f[1]+1 ][ (k_f[0], w) ] 
                if (w , k_f[1]+1) in visited_b and  candidate_mu < mu: #and len(P_f[k_f]) + len(P_b[( w , k_f[1]+1 )]) == T-2:
                    mu = candidate_mu
                    best_path = P_f[k_f] + [(k_f[0], k_f[1]), (w, k_f[1]+1)] + P_b[( w , k_f[1]+1 )][::-1]
                    
            
        if k_b[1] > 0: 
            for w in G.adj_inv[k_b[0]]:  # successors to v
                if ( w , k_b[1] - 1 ) not in visited_b: 
                    
                    d = D_b[k_b] - G.adj_inv[k_b[0]][w]  - G.emission_probabilities[ k_b[1] - 1 ][(w, k_b[0])]          # dgs: dist of start -> v -> w
                    if d < Q_b.get((w,k_b[1]-1), float("inf")):
                        Q_b[(w, k_b[1]-1)] = d                             
                        P_b[(w, k_b[1]-1)] = P_b[k_b] + [k_b]  
               
                candidate_mu = D_b[k_b] - G.emission_probabilities[ k_b[1] - 1 ][ (w, k_b[0]) ] - G.adj_inv[k_b[0]][w]  + D_f[(w, k_b[1]-1)] 
                if ( w, k_b[1]-1 ) in visited_f and candidate_mu < mu: #and len(P_f[(w, k_b[1]-1)]) + len(P_b[k_b]) == T - 2:                    
                    mu = candidate_mu #D_b[k_b]  - G.adj_inv[k_b[0]][w]  - G.emission_probabilities[ k_b[1] ][ (w, k_b[0]) ]  + D_f[(w, k_b[1]-1)]
                    best_path = P_f[(w, k_b[1]-1)] +  [(w, k_b[1]-1), (k_b[0], k_b[1])] +  P_b[k_b][::-1] #P_f[(w, t_forw)] + [ (w,k_b[1]-1) , (k_b[0], k_b[1]) ] + P_b[k_b][::-1]
       

        # check condition 
        if D_f[k_f] + D_b[k_b] >= mu:
            break 
        
  
    return best_path , mu 


    

def bidirectional_mint_bound(G, start, T, last_nodes):
    '''
   bidirectional-mint-bound
   Parameters: 
       G: graph  (graph)
       start: start node (int, str) 
       y: observation sequence (array) 
       last_nodes: possible last states (e.g., set of nodes reachable in T steps) 
    
    Returns: 
        best path: path (list)
        mu: best path cost (float)
    '''
    
    D_f = defaultdict(float) 
    D_b = defaultdict(float) 
    
    for state in start: 
        D_f[(state, 0)] = -G.upper_bound * T  # mapping of nodes to their dist from start
    for node in last_nodes:
        D_b[(node, T-1)] =  -G.upper_bound_reversed * T 
    
   
    Q_f = PQDict(D_f)           # priority queue for tracking min shortest path
    Q_b = PQDict(D_b)
    P_f = defaultdict(list)     # mapping of nodes to their direct predecessors
    P_b = defaultdict(list)  
    visited_f = set()       # unexplored node
    visited_b = set() 
    mu = float("inf") 
    while len(Q_f)>0 and len(Q_b)>0:     
        
        (k_f, d_f) = Q_f.popitem() 
        (k_b, d_b) = Q_b.popitem()                    # pop node w min dist d on frontier in constant time 
        D_f[k_f]=  d_f 
        D_b[k_b] = d_b      
        # update explored 
        visited_f.add(k_f)
        visited_b.add(k_b)   
                  
        if k_f[1] < T-1: 
            for w in G.adj[k_f[0]]:                          # successors to v
                if ( w , k_f[1]+1 ) not in visited_f:        
                    d = D_f[k_f] + G.upper_bound - G.adj[k_f[0]][w]  - G.emission_probabilities[ k_f[1] + 1 ][(k_f[0], w)]      # dgs: dist of start -> v -> w
                    if d < Q_f.get((w,k_f[1]+1), float("inf")):
                        Q_f[(w, k_f[1]+1)] = d     
                        P_f[(w, k_f[1]+1)] = P_f[k_f] + [k_f]  
                                        
                
                actual_cost_forward = D_f[k_f] + (T - 1 - k_f[1]) * G.upper_bound
                actual_cost_backward = D_b[( w , k_f[1]+1 )] + (T - 1 -  k_f[1] - 1) * G.upper_bound
                candidate_mu = actual_cost_forward - G.adj[k_f[0]][w]  + actual_cost_backward - G.emission_probabilities[ k_f[1]+1 ][ (k_f[0], w) ]
                if ( w , k_f[1]+1 ) in visited_b and candidate_mu < mu and len(P_f[k_f]) + len(P_b[( w , k_f[1]+1 )]) == T-2:
                    mu = candidate_mu
                    best_path = P_f[k_f] + [(k_f[0], k_f[1]), (w, k_f[1]+1)] + P_b[( w , k_f[1]+1 )][::-1]
                    
            
        if k_b[1] > 0: 
            for w in G.adj_inv[k_b[0]]:  # successors to v
                if ( w , k_b[1] - 1 ) not in visited_b: 
                    
                    d = D_b[k_b] + G.upper_bound - G.adj_inv[k_b[0]][w] - G.emission_probabilities[ k_b[1] ][ ( w , k_b[0] ) ]      # here we add the emission probability of the source and not of the destination 
                    if d < Q_b.get((w,k_b[1]-1), float("inf")):
                        Q_b[(w, k_b[1]-1)] = d                             
                        P_b[(w, k_b[1]-1)] = P_b[k_b] + [k_b]   
                                     
                actual_cost_forward = D_f[(w, k_b[1]-1)] + (T - 1 - k_b[1] + 1) * G.upper_bound
                actual_cost_backward = D_b[k_b] + (T - 1 - k_b[1]) * G.upper_bound
                candidate_mu = actual_cost_backward - G.emission_probabilities[ k_b[1] ][ (w, k_b[0] )] - G.adj_inv[k_b[0]][w]  + actual_cost_forward
                if ( w, k_b[1]-1 ) in visited_f and candidate_mu < mu and len(P_f[(w, k_b[1]-1)]) + len(P_b[k_b]) == T - 2:                                        
                    mu = candidate_mu
                    best_path = P_f[(w, k_b[1]-1)] +  [(w, k_b[1]-1), (k_b[0], k_b[1])] +  P_b[k_b][::-1] #P_f[(w, t_forw)] + [ (w,k_b[1]-1) , (k_b[0], k_b[1]) ] + P_b[k_b][::-1]

        # check condition 
        if D_f[k_f] + D_b[k_b] >= mu : 
            print("Breaking!")
            break 
                 
    
    return  best_path , mu




