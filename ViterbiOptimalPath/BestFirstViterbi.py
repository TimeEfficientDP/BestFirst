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
        mu: best path cost (float)
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
                     
    return  best_path , mu 



def bidirectional_mint_bound(G, start, y, last_nodes):
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
                
   return  best_path , mu



def mint_dfs(G, all_nodes, start, frames, space_budget_global, y,
                                   final_state = None, start_frame = 0, last_frame = None,
                                   n_frames = None,  middle_pairs = []):
    
    
    '''
    mint-dfs 
    Parameters: 
        G: transition graph (graph object) 
        all_nodes: graph nodes (list) 
        start: start state (int)
        space_budget_global: space budget (float)
        y: vector of observations (list) 
        final_state: if available, last state (int) 
        start_frame: start frame (int) 
        last_frame: last frame (int) 
        middle_pairs: current middle pairs (list) 
        
            
    Returns 
    middle_pairs: final middle pairs (list)
    '''
   
    # init 
    T = len(frames) 
    th = ceil((frames[0] + frames[-1])/2)   
    D = defaultdict(float) 
    D[(start, start_frame) ] = ( -G.emission_probabilities[ start ][ y[ start_frame ] ] , -1, (-1,-1) )         # mapping of nodes to their dist from start
    Q = PQDict(D)          
    stored_tokens = set() 
    stored_tokens.add( (start, start_frame) ) 
    visited = set() 
        
    if last_frame == None:
        last_frame = T-1 
                
    if n_frames == None:
        n_frames = T    
   

    while Q:             
                       
        (k, val) = Q.popitem()   
        d, pred , med_pair = val #unpack distance and optimal predecessor 
        stored_tokens.remove(k)        
        
        D[k]=d                           # est dijkstra greedy score
        visited.add(k)                   # remove from unexplored

        if k[1]==th: 
            if med_pair == (-1,-1): 
                med_pair = (pred, k[0]) 
                
        if (k[1] == last_frame and final_state is None) or (k[0] == final_state and k[1] == last_frame): 
         
            x_a, x_b =  med_pair 
                      
            N_left = floor(len(frames)/2)      
            
            if N_left >1 : 
                
                left_frames = frames[:N_left]
                start_frame = left_frames[0]
                last_frame = left_frames[-1]
                ancestors = BFS_ancestors(G, x_a,  N_left)
                mint_dfs(G, ancestors.union({x_a}), start, left_frames, space_budget_global, y, final_state = x_a, start_frame = start_frame, last_frame = last_frame, n_frames = N_left, middle_pairs = middle_pairs)
                        
            middle_pairs.append((x_a, x_b)) 
            N_right = len(frames) - N_left
            
            if N_right >1: 
                
                right_frames = frames[-N_right:]
                start_frame_right = right_frames[0]
                last_frame_right = right_frames[-1]
                descendants = BFS_descendants(G, x_b,  N_right)
                mint_dfs(G, descendants.union({x_b}), x_b, right_frames, space_budget_global, y, final_state = final_state, start_frame =   start_frame_right , last_frame = last_frame_right , n_frames = N_right, middle_pairs = middle_pairs)
            break
            
        
        for w in G.adj[k[0]]:                        
            if k[1]+1 > last_frame: 
                break 
            
            if ( w , k[1]+1 ) not in visited and w in all_nodes:        
                
                d = D[k] - G.adj[k[0]][w]  - G.emission_probabilities[ w ][ y[k[1]+1] ]      # dgs: dist of start -> v -> w
                tmp = Q.get( (w,k[1]+1), (float("inf") , -1, (-1,1))   )[0]
                    
                if d < tmp:
                    if  len(stored_tokens) > space_budget_global and (w, k[1]+1) not in stored_tokens: 
                        # running dfs 
                        Q, stored_tokens = dfs(G, all_nodes, k[0], w, k[1]+1, d, med_pair, Q, stored_tokens, y, set(), space_budget_global, n_frames, len(stored_tokens), th)                          
                        
                    else:
                        # update queue ´
                        Q[(w, k[1]+1)] = (d, k[0], med_pair)   
                        stored_tokens.add( (w, k[1]+1) )
    
    return middle_pairs 


def dfs( G, all_nodes, pred, state, frame, cost, med_pair, Q, stored_tokens,  y, visited_dfs, space_budget, n_frames, tot_tokens, th): 
                    
    ''' function invoked by mint_dfs to perform dfs '''
        
    if frame==th: 
        med_pair = (pred, state) 
    
    if frame < n_frames-1: 
        for neighbour in G.adj[state]:
            if neighbour in all_nodes:
                this_cost = cost - G.adj[state][neighbour]  - G.emission_probabilities[ neighbour ][ y[frame+1] ]                
    
                if (neighbour,frame+1) in stored_tokens: 
                        tmp = Q.get((neighbour,frame+1), float("inf"))
                        if isinstance(tmp, tuple): 
                            tmp = tmp[0] 
                            
                        if this_cost < tmp:
                            Q[(neighbour, frame+1)] = (this_cost , state, med_pair)

                else:
                    Q, stored_tokens  = dfs(G=G, all_nodes=all_nodes, pred=state, state=neighbour, frame= frame+1, cost= this_cost ,
                                                                      med_pair = med_pair , Q = Q , stored_tokens = stored_tokens , y = y, visited_dfs=visited_dfs, 
                                                                      space_budget=space_budget, n_frames=n_frames, tot_tokens=tot_tokens, th=th)
    else: 
        tmp =  Q.get((state,frame), (float("inf"), -1, (-1,-1)))[0]
        if cost < tmp:                 
            Q[(state, frame)] = (cost, pred, med_pair)
            stored_tokens.add(  (state,frame)    ) 
            
    return Q, stored_tokens



def BFS_ancestors(G, source, b):
 
        ''''
        function invoked by mint_dfs to perform single-source BFS traversal of ancestors up to b hops 
        
        '''
        
        # Mark all the vertices as not visited
        visited = set() 

        # Create a queue for BFS
        queue = []
 
        # Mark the source node as 
        # visited and enqueue it
        queue.append(source)
        queue.append("null") # for level 
      
        level = 0 
        
        while queue and level < b:
            # Dequeue a vertex from 
            # queue and print it
            s = queue.pop(0)
                        
            if s == "null": 
                level += 1 # you increase one level everytime you encounter a null 
                queue.append("null") 
                
            else: 
                # Get all adjacent vertices of the
                # dequeued vertex s. If a adjacent
                # has not been visited, then mark it
                # visited and enqueue it
                for state_idx in  G.adj_inv[s]:                     
                    if state_idx not in visited: 
                        queue.append(state_idx)
                        visited.add(state_idx)
                
        return visited



def BFS_descendants(G, source,  b):
 
        ''''
        function invoked by mint_dfs to perform traversal of descendants up to b hops 
        
        '''
        
        # Mark all the vertices as not visited
        visited = set() 
        # Create a queue for BFS
        queue = []
        # Mark the source node as 
        # visited and enqueue it
        queue.append(source)
        queue.append("null") # for level 
        #visited.add(source)
        level = 0 
      
        while queue and level < b:
            # Dequeue a vertex from 
            # queue and print it
            s = queue.pop(0)
            if s == "null": 
                level += 1 # you increase one level everytime you encounter a null 
                queue.append("null") 
                
            else: 
                # Get all adjacent vertices of the
                # dequeued vertex s. If a adjacent
                # has not been visited, then mark it
                # visited and enqueue it
                for state_idx in G.adj[s]:
                    if state_idx not in visited:
                        queue.append(state_idx)
                        visited.add(state_idx)
                      
        return visited 
