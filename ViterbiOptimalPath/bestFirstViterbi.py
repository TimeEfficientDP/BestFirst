from pqdict import PQDict
import numpy as np 
from collections import defaultdict 
import logging


def mint(G, start, y):
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
    D[(start, 0)] = - G.emission_probabilities[ start ][ y[0] ]          # mapping of nodes to their dist from start
    Q = PQDict(D)          
    P = defaultdict(list)               
    visited = set() 
    T = len(y) 
    success = False # debugging only 
        
    while Q:             
                                          
        (k, d) = Q.popitem()     # pop node w min dist d on frontier in constant time            
        if k[1] == T-1: 
            success=True # debugging only 
            break                 
        
        visited.add(k) 
        for w in G.adj[k[0]]:                       
            if ( w , k[1]+1 ) not in visited:        
                # then w is a frontier node
                new_d = d - G.adj[k[0]][w]  - G.emission_probabilities[ w ][ y[k[1]+1] ]   
                if new_d < Q.get((w,k[1]+1), float("inf")):
                    Q[(w, k[1]+1)] = new_d   
                    P[(w, k[1]+1)] = P[k] + [k] 
                    
        if not success: 
            logging.warning('Algorithm terminated before last frame was reached.')  # debugging only 
        
    return d , P[k] + [k]



def mint_bound(G, start, y):
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
    T = len(y) 
    D[(start, 0)] = -G.upper_bound * T        
    Q = PQDict(D)
    P = defaultdict(list)               
    visited = set() 
    
    while Q:                                
        (k, d) = Q.popitem()               
        visited.add(k)  

        if k[1] == T-1: 
            break                 

        for w in G.adj[k[0]]:                         
            if ( w, k[1]+1 ) not in visited:                        
                # then w is a frontier node
                new_d = d + G.upper_bound - G.adj[k[0]][w]  - G.emission_probabilities[ w ][ y[k[1]+1] ]         # dgs: dist of start -> v -> w

                if new_d < Q.get((w,k[1]+1), float("inf")):
                    Q[(w, k[1]+1)] = new_d              
                    P[(w, k[1]+1)] = P[k] + [k]  

    return d,  P[k] + [k] 




def viterbi(G, start, y):
    """
    Viterbi algorithm - pull implementation
    Parameters:
        G: graph  (graph)
        start: start node (int, str) 
        y: observation sequence (array) 
    Returns: 
        x: optimal paths (dict)
        T1; path probability table (array) 
        
    """
    K = len(G.nodes)
    T = len(y)
    T1 = np.zeros((K, T))
    T2 = np.zeros((K, T))

    # Initilaize the tracking tables from first observation
    Pi = np.array([float("-inf") if it!=start else 0 for it in G.nodes])
    
    T1[:, 0] = Pi +  G.emission_probabilities[:, y[0]] 
    T2[:, 0] = 0
    
    # Iterate throught the observations updating the tracking tables
    for j in range(1, T):                 
        for i in range(K): 
            i_max = float("-inf") 
            best_neig = float("-inf")  
            for neig in G.adj_inv[i]: 
                this_likelihood = T1[neig , j - 1]  + G.adj_inv[i][neig] + G.emission_probabilities[i][y[j]] 
                if this_likelihood > i_max: 
                    i_max = this_likelihood
                    best_neig = neig 
                
            T1[i,j] = i_max
            T2[i,j] = best_neig
                          
    # Build the output, optimal model trajectory by Backtracking 
    x = np.zeros(T, dtype=int)
    x[-1] = int(np.argmax(T1[:, T - 1]))
    for i in reversed(range(1, T)):
        x[i - 1] = T2[x[i], i]
    return x, T1



def viterbi_tp(G, start, y):
    """
    Viterbi algorithm - push (token-passing) implementation
    Parameters:
        G: graph  (graph)
        start: start node (int, str) 
        y: observation sequence (array) 
    
    Returns: 
        x: optimal paths (dict)
        T1; path probability table (array) 
    """
    K = len(G.nodes)
    T = len(y)
    T1 = np.zeros((K, T))
    T2 = np.zeros((K, T))


    # Initilaize the tracking tables from first observation
    Pi = np.array([float("-inf") if it!=start else 0 for it in G.nodes])
    
    T1[:, 0] = Pi +  G.emission_probabilities[:, y[0]] 
    T2[:, 0] = 0
    
    
    # Iterate through the observations updating the tracking tables
    for j in range(1, T): 
        this_j_T1 = [float("-inf")  for _ in range(K)] 
        this_j_T2 = [float("-inf")  for _ in range(K)] 
        for i in range(K):             
            for neig in G.adj[i]: 
                this_likelihood = T1[i , j - 1]  + G.adj[i][neig] + G.emission_probabilities[neig][y[j]] 
                if this_likelihood > this_j_T1[neig]: 
                    this_j_T1[neig] = this_likelihood
                    this_j_T2[neig] = i  
            
            
        T1[:,j] = this_j_T1
        T2[:,j] = this_j_T2
                      
    # Build the output, optimal model trajectory by Backtracking 
    x = np.zeros(T, dtype=int)
    x[-1] = int(np.argmax(T1[:, T - 1]))
    for i in reversed(range(1, T)):
        x[i - 1] = T2[x[i], i]
    
    return x, T1
    
    
    
    
    

def bidirectional_mint(G, start, y, last_nodes):
    '''
    bidirectional-mint algorithm
    Parameters: 
        G: graph  (graph)
        start: start node (int, str) 
        y: observation sequence (array) 
        last_nodes: possible last states (e.g., set of nodes reachable in T steps) 
    
    Returns: 
        best path: path (list)
    '''
    
    D_f = defaultdict(float) 
    D_b = defaultdict(float) 
    T = len(y) 
    D_f[(start, 0)] = - G.emission_probabilities[ start ][ y[0] ] # mapping of nodes to their dist from start
    for node in last_nodes:
        D_b[(node, T-1)] = 0 
    Q_f = PQDict(D_f)
    Q_b = PQDict(D_b)
    P_f = defaultdict(list)             
    P_b = defaultdict(list)  
    visited_f = set()       # unexplored node
    visited_b = set() 
    mu = float("inf") 
    while len(Q_f)>0 and len(Q_b)>0:     
       
        (k_f, d_f) = Q_f.popitem() 
        (k_b, d_b) = Q_b.popitem()                   
        D_f[k_f]=  d_f 
        D_b[k_b] = d_b      
        # update explored 
        visited_f.add(k_f)
        visited_b.add(k_b)   
        if k_f[1] < T-1: 
            for w in G.adj[k_f[0]]:                         
                if ( w , k_f[1]+1 ) not in visited_f:        
                    d = D_f[k_f] - G.adj[k_f[0]][w]  - G.emission_probabilities[ w ][ y[k_f[1]+1] ]     
                    if d < Q_f.get((w,k_f[1]+1), float("inf")):
                        Q_f[(w, k_f[1]+1)] = d     
                        P_f[(w, k_f[1]+1)] = P_f[k_f] + [k_f] 
                                        
                
                if ( w , k_f[1]+1 ) in visited_b and D_f[k_f] - G.adj[k_f[0]][w]  + D_b[( w , k_f[1]+1 )] - G.emission_probabilities[ w ][ y[k_f[1]+1] ] < mu and len(P_f[k_f]) + len(P_b[( w , k_f[1]+1 )]) == T-2:
                    mu = D_f[k_f] - G.adj[k_f[0]][w]  + D_b[( w , k_f[1]+1 )]  - G.emission_probabilities[ w ][ y[k_f[1]+1] ]
                    best_path = P_f[k_f] + [(k_f[0], k_f[1]), (w, k_f[1]+1)] + P_b[( w , k_f[1]+1 )][::-1]
                    
            
        if k_b[1] > 0: 
            for w in G.adj_inv[k_b[0]]:  # successors to v
                if ( w , k_b[1] - 1 ) not in visited_b: 
                    
                    d = D_b[k_b] - G.adj_inv[k_b[0]][w]  - G.emission_probabilities[ k_b[0] ][ y[k_b[1]] ]      # here we add the emission probability of the source and not of the destination 
                    if d < Q_b.get((w,k_b[1]-1), float("inf")):
                        Q_b[(w, k_b[1]-1)] = d                             
                        P_b[(w, k_b[1]-1)] = P_b[k_b] + [k_b]   
                                       
                if ( w, k_b[1]-1 ) in visited_f and D_b[k_b] - G.emission_probabilities[ k_b[0] ][ y[k_b[1]] ] - G.adj_inv[k_b[0]][w]  + D_f[(w, k_b[1]-1)] < mu and len(P_f[(w, k_b[1]-1)]) + len(P_b[k_b]) == T - 2:                                        
                    mu = D_b[k_b]  - G.adj_inv[k_b[0]][w]  - G.emission_probabilities[ k_b[0] ][ y[k_b[1]] ] + D_f[(w, k_b[1]-1)]
                    best_path = P_f[(w, k_b[1]-1)] +  [(w, k_b[1]-1), (k_b[0], k_b[1])] +  P_b[k_b][::-1] #P_f[(w, t_forw)] + [ (w,k_b[1]-1) , (k_b[0], k_b[1]) ] + P_b[k_b][::-1]
                   
        # check condition 
        if D_f[k_f] + D_b[k_b] >= mu : 
            print("Breaking!")
            break 
                     
    return  best_path 










def bidirectional_dijkstra_bound(G, start, y, last_nodes):
   '''
   bidirectional-mint-bound
   Parameters: 
       G: graph  (graph)
       start: start node (int, str) 
       y: observation sequence (array) 
       last_nodes: possible last states (e.g., set of nodes reachable in T steps) 
    
    Returns: 
        best path: path (list)
    '''
    
   D_f = defaultdict(float) 
   D_b = defaultdict(float) 
   T = len(y) 
   D_f[(start, 0)] =  -G.upper_bound * T # 
   for node in last_nodes:
       D_b[(node, T-1)] = -G.upper_bound_reversed * T 
   Q_f = PQDict(D_f)          
   Q_b = PQDict(D_b)
   P_f = defaultdict(list)     
   P_b = defaultdict(list)  
   visited_f = set()       # unexplored node
   visited_b = set() 
   mu = float("inf") 
   while len(Q_f)>0 and len(Q_b)>0:     
       (k_f, d_f) = Q_f.popitem() 
       (k_b, d_b) = Q_b.popitem()      
       D_f[k_f]=  d_f 
       D_b[k_b] = d_b      
       visited_f.add(k_f)
       visited_b.add(k_b)   
       if k_f[1] < T-1: 
           for w in G.adj[k_f[0]]:                        
               if ( w , k_f[1]+1 ) not in visited_f:        
                   d = D_f[k_f] + G.upper_bound - G.adj[k_f[0]][w]  - G.emission_probabilities[ w ][ y[k_f[1]+1] ]      # dgs: dist of start -> v -> w
                   if d < Q_f.get((w,k_f[1]+1), float("inf")):
                       Q_f[(w, k_f[1]+1)] = d     
                       P_f[(w, k_f[1]+1)] = P_f[k_f] + [k_f]  
                                       
               
               actual_cost_forward = D_f[k_f] + (T - 1 - k_f[1]) * G.upper_bound
               actual_cost_backward = D_b[( w , k_f[1]+1 )] + (T - 1 -  k_f[1] - 1) * G.upper_bound
               if ( w , k_f[1]+1 ) in visited_b and actual_cost_forward - G.adj[k_f[0]][w]  + actual_cost_backward - G.emission_probabilities[ w ][ y[k_f[1]+1] ] < mu and len(P_f[k_f]) + len(P_b[( w , k_f[1]+1 )]) == T-2:
                   mu = actual_cost_forward - G.adj[k_f[0]][w]  + actual_cost_backward  - G.emission_probabilities[ w ][ y[k_f[1]+1] ]
                   best_path = P_f[k_f] + [(k_f[0], k_f[1]), (w, k_f[1]+1)] + P_b[( w , k_f[1]+1 )][::-1]
                   
           
       if k_b[1] > 0: 
           for w in G.adj_inv[k_b[0]]:  
               if ( w , k_b[1] - 1 ) not in visited_b: 
                   
                   d = D_b[k_b] + G.upper_bound - G.adj_inv[k_b[0]][w] - G.emission_probabilities[ k_b[0] ][ y[k_b[1]] ]      # here we add the emission probability of the source and not of the destination 
                   if d < Q_b.get((w,k_b[1]-1), float("inf")):
                       Q_b[(w, k_b[1]-1)] = d                             
                       P_b[(w, k_b[1]-1)] = P_b[k_b] + [k_b]   
                                    
               actual_cost_forward = D_f[(w, k_b[1]-1)] + (T - 1 - k_b[1] + 1) * G.upper_bound
               actual_cost_backward = D_b[k_b] + (T - 1 - k_b[1]) * G.upper_bound
               if ( w, k_b[1]-1 ) in visited_f and actual_cost_backward - G.emission_probabilities[ k_b[0] ][ y[k_b[1]] ] - G.adj_inv[k_b[0]][w]  + actual_cost_forward < mu and len(P_f[(w, k_b[1]-1)]) + len(P_b[k_b]) == T - 2:                                        
                   mu = actual_cost_backward - G.adj_inv[k_b[0]][w]  - G.emission_probabilities[ k_b[0] ][ y[k_b[1]] ] + actual_cost_forward
                   best_path = P_f[(w, k_b[1]-1)] +  [(w, k_b[1]-1), (k_b[0], k_b[1])] +  P_b[k_b][::-1] #P_f[(w, t_forw)] + [ (w,k_b[1]-1) , (k_b[0], k_b[1]) ] + P_b[k_b][::-1]

                   
       # check condition 
       if D_f[k_f] + D_b[k_b] >= mu :
           print("Breaking!")
           break 
                
   return  best_path 