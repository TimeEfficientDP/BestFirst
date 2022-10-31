from pqdict import PQDict
import numpy as np 
from collections import defaultdict 
<<<<<<< HEAD
import logging



def mint_bound(G, start, y):
    '''
   
    Mint-bound algorithm
    Input: 
    G: graph  (graph)
    start: start node (int, str) 
    y: observation sequence (array) 
    
    Returns: 
    d: optimal costs (dict)
    P[k] + [k]: path (list)
=======
import copy 
import logging
import time 
import heapq
from math import ceil , floor 
from graph import Graph
### TODO : implement prruning of the priority queue 


def dijkstra_bound(G, start, y):
    
    
    '''
    dijkstra's algorithm determines the length from `start` to every other 
    vertex in the graph.
    The graph argument `G` should be a dict indexed by nodes.  The value 
    of each item `G[v]` should also a dict indexed by successor nodes.
    In other words, for any node `v`, `G[v]` is itself a dict, indexed 
    by the successors of `v`.  For any directed edge `v -> w`, `G[v][w]` 
    is the length of the edge from `v` to `w`.
        graph = {'a': {'b': 1}, 
                 'b': {'c': 2, 'b': 5}, 
                 'c': {'d': 1},
                 'd': {}}
    Returns two dicts, `dist` and `pred`:
        dist, pred = dijkstra(graph, start='a') 
    
    `dist` is a dict mapping each node to its shortest distance from the
    specified starting node:
        assert dist == {'a': 0, 'c': 3, 'b': 1, 'd': 4}
    `pred` is a dict mapping each node to its predecessor node on the
    shortest path from the specified starting node:
        assert pred == {'b': 'a', 'c': 'b', 'd': 'c'}
>>>>>>> 153cb145f02d8e319d061cc3839923d57d7f34ce
    
    '''
    
    D = defaultdict(dict) 
    T = len(y) 
    D[(start, 0)] = -G.upper_bound * T         # mapping of nodes to their dist from start
    Q = PQDict(D)           # priority queue for tracking min shortest path
    P = defaultdict(list)                # mapping of nodes to their direct predecessors
    visited = set() 
    success=False
    
<<<<<<< HEAD
    while Q:                                 
        (k, d) = Q.popitem()                 # pop node w min dist d on frontier in constant time 
=======
    
    while Q:                                 # nodes yet to explore
        (k, d) = Q.popitem()                 # pop node w min dist d on frontier in constant time 
        D[k]= d                              # est dijkstra greedy score
>>>>>>> 153cb145f02d8e319d061cc3839923d57d7f34ce
        visited.add(k)  

        # remove from unexplored
        if k[1] == T-1: 
            success=True 
<<<<<<< HEAD
            break                 
     
        for w in G.adj[k[0]]:                          
            if ( w, k[1]+1 ) not in visited:                       
                d = D[k] + G.upper_bound - G.adj[k[0]][w]  - G.emission_probabilities[ w ][ y[k[1]+1] ]       
                if d < Q.get((w,k[1]+1), float("inf")):
                    Q[(w, k[1]+1)] = d               
=======
            break                  # if the frame k[1] is T we arrived to the end 

        # now consider the edges from v with an unexplored head -
        # we may need to update the dist of unexplored successors 
        for w in G.adj[k[0]]:                          # suP_Viterbiccessors to v
            if ( w, k[1]+1 ) not in visited:                         # then w is a frontier node
                d = D[k] + G.upper_bound - G.adj[k[0]][w]  - G.emission_probabilities[ w ][ y[k[1]+1] ]         # dgs: dist of start -> v -> w

                if d < Q.get((w,k[1]+1), float("inf")):
                    Q[(w, k[1]+1)] = d               # set/update dgs
                   # history = copy.deepcopy(P[k])
                   # history.append(k)
>>>>>>> 153cb145f02d8e319d061cc3839923d57d7f34ce
                    P[(w, k[1]+1)] = P[k] + [k]  
                 
    if not success:
        logging.warning('Algorithm terminated before last frame was reached.')
    
<<<<<<< HEAD
    return D, d
=======
    r =  len(visited) /  (len(G.nodes) * len(y))
    
    #    history = P[k]
    #    history.append(k)
    
    return D,  P[k] + [k] , r
>>>>>>> 153cb145f02d8e319d061cc3839923d57d7f34ce




<<<<<<< HEAD
def mint(G, start, y):
    '''
    Mint algorithm
    Input: 
    G: graph  (graph)
    start: start node (int, str) 
    y: observation sequence (array) 
    
    Returns: 
    d: optimal costs (dict)
    P[k] + [k]: path (list)
    
    '''
    D = defaultdict(float) 
    D[(start, 0)] = - G.emission_probabilities[ start ][ y[0] ]         
    Q = PQDict(D)           
    P = defaultdict(list)              
=======
def dijkstra(G, start, y):
    
    
    '''
    dijkstra's algorithm determines the length from `start` to every other 
    vertex in the graph.
    The graph argument `G` should be a dict indexed by nodes.  The value 
    of each item `G[v]` should also a dict indexed by successor nodes.
    In other words, for any node `v`, `G[v]` is itself a dict, indexed 
    by the successors of `v`.  For any directed edge `v -> w`, `G[v][w]` 
    is the length of the edge from `v` to `w`.
        graph = {'a': {'b': 1}, 
                 'b': {'c': 2, 'b': 5}, 
                 'c': {'d': 1},
                 'd': {}}
    Returns two dicts, `dist` and `pred`:
        dist, pred = dijkstra(graph, start='a') 
    
    `dist` is a dict mapping each node to its shortest distance from the
    specified starting node:
        assert dist == {'a': 0, 'c': 3, 'b': 1, 'd': 4}
    `pred` is a dict mapping each node to its predecessor node on the
    shortest path from the specified starting node:
        assert pred == {'b': 'a', 'c': 'b', 'd': 'c'}
    
    '''
    D = defaultdict(float) 
    D[(start, 0)] = - G.emission_probabilities[ start ][ y[0] ]          # mapping of nodes to their dist from start
    Q = PQDict(D)           # priority queue for tracking min shortest path
    P = defaultdict(list)                # mapping of nodes to their direct predecessors
    #to_visit = set(  zip( G.adj.keys() , [0 for _ in range(len(G.adj.keys()))]  ) )       # unexplored node
>>>>>>> 153cb145f02d8e319d061cc3839923d57d7f34ce
    visited = set() 
    success=False
    T = len(y) 
        
<<<<<<< HEAD
    
    while Q:             
                                          
        (k, d) = Q.popitem()
    
        if k[1] == T-1: 
            success=True 
            break               

        visited.add(k) 

        for w in G.adj[k[0]]:                         
=======
   # dict_paths = defaultdict(list) 
    
    while Q:             
                                          # nodes yet to explore
        (k, d) = Q.popitem()                 # pop node w min dist d on frontier in constant time 
        D[k]=d                              # est dijkstra greedy score
        visited.add(k)                   # remove from unexplored
        
        #print("visiting " + str(k)) 
        #print("len Q " + str(len(Q)))
        
        if k[1] == T-1: 
            success=True 
            break                  # if the frame k[1] is T we arrived to the end 

        # now consider the edges from v with an unexplored head -
        # we may need to update the dist of unexplored successors 
        for w in G.adj[k[0]]:                          # successors to v
>>>>>>> 153cb145f02d8e319d061cc3839923d57d7f34ce
        
            if ( w , k[1]+1 ) not in visited:        

                # then w is a frontier node
<<<<<<< HEAD
                new_d = d - G.adj[k[0]][w]  - G.emission_probabilities[ w ][ y[k[1]+1] ]   
                if new_d < Q.get((w,k[1]+1), float("inf")):
                    Q[(w, k[1]+1)] = new_d   
=======
                d = D[k] - G.adj[k[0]][w]  - G.emission_probabilities[ w ][ y[k[1]+1] ]      # dgs: dist of start -> v -> w
                if d < Q.get((w,k[1]+1), float("inf")):
                    Q[(w, k[1]+1)] = d     
>>>>>>> 153cb145f02d8e319d061cc3839923d57d7f34ce
                    P[(w, k[1]+1)] = P[k] + [k] 
        
    if not success:
        logging.warning('Algorithm terminated before last frame was reached.')
<<<<<<< HEAD

    return d , P[k] + [k]
=======
    
    r =  len(visited) / (len(G.nodes) * len(y))

    return D, P[k] + [k] , r




def dijkstra_dfs(G, start, y):
    
    
    '''
    dijkstra's algorithm determines the length from `start` to every other 
    vertex in the graph.
    The graph argument `G` should be a dict indexed by nodes.  The value 
    of each item `G[v]` should also a dict indexed by successor nodes.
    In other words, for any node `v`, `G[v]` is itself a dict, indexed 
    by the successors of `v`.  For any directed edge `v -> w`, `G[v][w]` 
    is the length of the edge from `v` to `w`.
        graph = {'a': {'b': 1}, 
                 'b': {'c': 2, 'b': 5}, 
                 'c': {'d': 1},
                 'd': {}}
        
    start: state state 
    y: vector of observation
    Returns two dicts, `dist` and `pred`:
        dist, pred = dijkstra(graph, start='a') 
    
   Returns 
   history: best path 
   '''
    
    D = defaultdict(float) 
    D[(start, 0)] = - G.emission_probabilities[ start ][ y[0] ]          # mapping of nodes to their dist from start
    Q = PQDict(D)           # priority queue for tracking min shortest path
    P = defaultdict(list)                # mapping of nodes to their direct predecessors
    #to_visit = set(  zip( G.adj.keys() , [0 for _ in range(len(G.adj.keys()))]  ) )       # unexplored node
    
    Q_by_state = [[] for _ in range(len(G.nodes))]  # we need to keep for each state an ordered list of (t, value) so that we can check the length and remove the best 
                                                    # each state will be dealt with by heapq
    
    map_state2t = defaultdict(set)
    heapq.heappush( Q_by_state[start] , (- G.emission_probabilities[ start ][ y[0] ]   , 0) )
    map_state2t[start].add(0) 
    
    visited = set() 
    success=False
    T = len(y) 
    last_frame = T-1 
        
    while Q:             
        
                                         # nodes yet to explore
        (k, d) = Q.popitem()             # pop nodap_state2t[k[0]]e w min dist d on frontier in constant time 
        
        print("k " + str(k)) 

        
        map_state2t[k[0]].remove(k[1])
        D[k]=d                           # est dijkstra greedy score
        visited.add(k)                   # remove from unexplored
       
        _ , _ = heapq.heappop(Q_by_state[k[0]]) # the gloabal best is also the best in that state 
        
        
        #print("visiting " + str(k)) 
        #print("len Q " + str(len(Q)))
        if k[1] == T-1: 
            success=True 
            break                  # if the frame k[1] is T we arrived to the end 

        # now consider the edges from v with an unexplored head -
        # we may need to update the dist of unexplored successors 
        for w in G.adj[k[0]]:                          # successors to v
        
            if ( w , k[1]+1 ) not in visited:        

                # then w is a frontier node
                d = D[k] - G.adj[k[0]][w]  - G.emission_probabilities[ w ][ y[k[1]+1] ]      # dgs: dist of start -> v -> w
                if d < Q.get((w,k[1]+1), float("inf")):
                    
                    
                    
                    
                    if len(map_state2t[w]) > 2 and k[1]+1 not in map_state2t[w]:
                        
                        print("running dfs!!")
                        Q, Q_by_state, map_state2t = dfs(G, w, k[1]+1, d, Q_by_state, Q, map_state2t, y)  

                                                
                    else:
                        # we already know that it is better 
                        Q[(w, k[1]+1)] = d    
                        heapq.heappush( Q_by_state[w] , (d, k[1]+1) ) # but if there is already an entry for k[1]+1 this would be double counted 
                        map_state2t[w].add(k[1]+1)
                        
                        
                    #if len(Q_by_state[w]) > 1: # after adding w it has become 3 
                        
                    #    print("running dfss") 
                        
                        # run dfs 
                      #  this_d, this_t = heapq.heappop(Q_by_state[w])
                        
    
    r =  len(visited) / (len(G.nodes) * len(y))
 
    print("final node " + str(k[0]))
    
    return D, r




def BFS_descendants(G, source,  b):
 
        ''''
        Perform single-source BFS traversal of ancestors up to b hops 
        for SIEVE-Middlepath
        
        Parameters: 
            source: starting node 
            indices: array (K) sequence of states 
            b: number of hops ¨
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
        
        adj = defaultdict(dict)   
        adj_inv = defaultdict(dict) 
        
        current_lower_bound = 0
        current_upper_bound = float("-inf") 
        
        current_lower_bound_reversed = 0
        current_upper_bound_reversed = float("-inf")     
        
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
                        
                        adj[s][state_idx] = G.adj[s][state_idx] 
                        adj_inv[state_idx][s] = G.adj_inv[state_idx][s] 
                        
                        adj[s][state_idx] = G.adj[s][state_idx] 
                        adj_inv[state_idx][s] = G.adj_inv[state_idx][s] 
                        
                        this_edge_cost_min = G.evaluate_edge_min(s, state_idx) 
                                        
                        if this_edge_cost_min < current_lower_bound: 
                            current_lower_bound=this_edge_cost_min
                            
                        this_edge_cost_max = G.evaluate_edge_max(s, state_idx) 
                                        
                        if this_edge_cost_max > current_upper_bound: 
                            current_upper_bound = this_edge_cost_max
                            
                            
                        this_edge_cost_min = G.evaluate_edge_min_reversed(s, state_idx) 
                        if this_edge_cost_min < current_lower_bound: 
                            current_lower_bound_reversed = this_edge_cost_min
                          
                        this_edge_cost_max = G.evaluate_edge_max_reversed(s, state_idx) 
                        if this_edge_cost_max > current_upper_bound: 
                            current_upper_bound_reversed = this_edge_cost_max    
                    
        return visited , adj, adj_inv, current_lower_bound, current_upper_bound , current_lower_bound_reversed , current_upper_bound_reversed




def BFS_ancestors(G, source, b):
 
        ''''
        Perform single-source BFS traversal of ancestors up to b hops 
        for SIEVE-Middlepath
        
        Parameters: 
            source: starting node 
            indices: array (K) sequence of states 
            b: number of hops ¨
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
        
        current_lower_bound = 0
        current_upper_bound = float("-inf") 
        
        current_lower_bound_reversed = 0
        current_upper_bound_reversed = float("-inf")     
        
        level = 0 
        
        adj = defaultdict(dict)   
        adj_inv = defaultdict(dict)   
        
        
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
                        
                        adj[s][state_idx] = G.adj[s][state_idx] 
                        adj_inv[state_idx][s] = G.adj_inv[state_idx][s] 
                        
                        this_edge_cost_min = G.evaluate_edge_min(s, state_idx) 
                                        
                        if this_edge_cost_min < current_lower_bound: 
                            current_lower_bound=this_edge_cost_min
                            
                        this_edge_cost_max = G.evaluate_edge_max(s, state_idx) 
                                        
                        if this_edge_cost_max > current_upper_bound: 
                            current_upper_bound = this_edge_cost_max
                            
                            
                        this_edge_cost_min = G.evaluate_edge_min_reversed(s, state_idx) 
                        if this_edge_cost_min < current_lower_bound: 
                            current_lower_bound_reversed = this_edge_cost_min
                          
                        this_edge_cost_max = G.evaluate_edge_max_reversed(s, state_idx) 
                        if this_edge_cost_max > current_upper_bound: 
                            current_upper_bound_reversed = this_edge_cost_max    
                    
                
        
        return visited , adj , adj_inv, current_lower_bound, current_upper_bound , current_lower_bound_reversed , current_upper_bound_reversed
        




def dijkstra_dfs_sieve(G, start, y, space_budget, final_state = None, start_frame = 0, last_frame = None, n_frames = None):
    
    
    '''
    dijkstra's algorithm determines the length from `start` to every other 
    vertex in the graph.
    The graph argument `G` should be a dict indexed by nodes.  The value 
    of each item `G[v]` should also a dict indexed by successor nodes.
    In other words, for any node `v`, `G[v]` is itself a dict, indexed 
    by the successors of `v`.  For any directed edge `v -> w`, `G[v][w]` 
    is the length of the edge from `v` to `w`.
        graph = {'a': {'b': 1}, 
                 'b': {'c': 2, 'b': 5}, 
                 'c': {'d': 1},
                 'd': {}}
        
    start: state state 
    y: vector of observation
    Returns two dicts, `dist` and `pred`:
        dist, pred = dijkstra(graph, start='a') 
    
   Returns 
   history: best path 
   '''
    
    D = defaultdict(float) 
    D[(start, start_frame)] = (- G.emission_probabilities[ start ][ y[start_frame] ] , -1)         # mapping of nodes to their dist from start
    Q = PQDict(D)           # priority queue for tracking min shortest path
    P = defaultdict(list)                # mapping of nodes to their direct predecessors
    #to_visit = set(  zip( G.adj.keys() , [0 for _ in range(len(G.adj.keys()))]  ) )       # unexplored node
    
    Q_by_state = [[] for _ in range(len(G.nodes))]  # we need to keep for each state an ordered list of (t, value) so that we can check the length and remove the best 
                                                    # each state will be dealt with by heapq
    new_medians = [(-1 , 1) for _ in range(len(G.nodes))]
    map_state2t = defaultdict(set)
    heapq.heappush( Q_by_state[start], (- G.emission_probabilities[ start ][ y[start_frame] ]   , start_frame) )
    map_state2t[start].add(start_frame) 
    
    visited = set() 
    success=False
    
    if last_frame == None:
        last_frame = len(y)-1 
       
    if n_frames == None:
        n_frames = len(y)
        
    T_floor = start_frame + floor((last_frame - start_frame + 1) / 2)
    
    #print("T floor " + str(T_floor))
        
    while Q:             
                                         # nodes yet to explore
        (k, d_pred) = Q.popitem()   
          # pop nodap_state2t[k[0]]e w min dist d on frontier in constant time 
        d, pred = d_pred #unpack distance and optimal predecessor 
        
        #print("visitng " + str(k))
        #print("k " + str(k)) 
        map_state2t[k[0]].remove(k[1])
        D[k]=d                           # est dijkstra greedy score
        visited.add(k)                   # remove from unexplored
        _ , _ = heapq.heappop(Q_by_state[k[0]]) # the gloabal best is also the best in that state 
        
        
        if k[1]==T_floor: 
            #print("new candidate ---- median " + str( (pred, k[0]) ))
            new_medians[k[0]] = (pred, k[0]) 

        elif k[1]>T_floor:
            new_medians[k[0]] = new_medians[pred]
                    
        #print("visiting " + str(k)) 
        #print("len Q " + str(len(Q)))
        
        if (k[1] == last_frame and final_state is None) or (k[0] == final_state and k[1] == last_frame): 
            #print("last frame " + str(last_frame))
            #print("final state " + str(final_state))
            #success=True 
            #print("start " + str(start))
            
            x_a, x_b =  new_medians[k[0]]
            
            #print("found median " + str( (x_a, x_b) ))
                        
            N_left = floor(n_frames/2)              
            # y_left = y[:N_left] 
            if N_left >1 : 
                 
                # use bfs to find ancestors in N_left-1 steps 
                #ancestors = BFS_ancestors(G, x_a,  N_left-1)
                # still using the entire state space 
                # continue recursion 
                
                #print("calling from the left with frames " + str(N_left) )
                dijkstra_dfs_sieve(G, start, y, space_budget, final_state = x_a, start_frame = 0, last_frame = N_left-1, n_frames = N_left)
            
            
            N_right = n_frames- N_left 
            #  y_right = y[-N_right:]
            # in order print 
            print("median pair " + str((x_a, x_b)))
            
            if N_right >1: 
                
                # use bfs to find descendants in N_right -1 steps 
                #descendants = BFS_descendants(G, x_b,  N_right-1)
                
                # continue recursion
                #print("calling from the right with frames " + str(N_right) )
                dijkstra_dfs_sieve(G, x_b, y, space_budget, final_state = final_state, start_frame =  T_floor  , last_frame = T_floor +  N_right - 1 , n_frames = N_right)
            
            break
            
        # break                  # if the frame k[1] is T we arrived to the end 
        # now consider the edges from v with an unexplored head -
        # we may need to update the dist of unexplored successors 


        for w in G.adj[k[0]]:                          # successors to v222, 333,
        
        
            #if k[0]==47 and start==0:
            #    print("w " + str(w))
        
            if k[1]+1 > last_frame: 
                break # break for loop go to next while loop iteration 
        
            if ( w , k[1]+1 ) not in visited:        
                
                # then w is a frontier node
                d = D[k] - G.adj[k[0]][w]  - G.emission_probabilities[ w ][ y[k[1]+1] ]      # dgs: dist of start -> v -> w
                
                
                
                
                tmp = Q.get((w,k[1]+1), float("inf"))
                if isinstance(tmp, tuple): 
                    tmp = tmp[0] 
                    
                #if k[0]==47 and start==0:
                #    print("d " + str(d) + " tmp " + str(tmp) + "k + 1 " + str(k[1]+1))
                
                if d < tmp:
                    
                    
                    if len(map_state2t[w]) > space_budget and k[1]+1 not in map_state2t[w]:
                        
                        #print("running dfs!") 
                        continue_flag = True 
                        Q, Q_by_state, map_state2t, continue_flag = dfs(G, k[0], w, k[1]+1, d, Q_by_state, Q, map_state2t, y, set(), space_budget, continue_flag, n_frames)  
                    #    print("dfs done!£")
                        
                        
                    else:
                        
                        # we already know that it is better 
                        Q[(w, k[1]+1)] = (d  , k[0])   
                        heapq.heappush( Q_by_state[w] , (d, k[1]+1) ) # but if there is already an entry for k[1]+1 this would be double counted 
                        map_state2t[w].add(k[1]+1)
                        #if len(Q_by_state[w]) > 1: # after adding w it has become 3 
                        #    print("running dfss") 
                        # run dfs 
                        #  this_d, this_t = heapq.heappop(Q_by_state[w])
                        
    
    r =  len(visited) / (len(G.nodes) * len(y))
    
    return D, r








def dijkstra_dfs_sieve_withPruning(G, start, y, final_state = None, start_frame = 0, last_frame = None, n_frames = None):
    
    
    '''
    dijkstra's algorithm determines the length from `start` to every other 
    vertex in the graph.
    The graph argument `G` should be a dict indexed by nodes.  The value 
    of each item `G[v]` should also a dict indexed by successor nodes.
    In other words, for any node `v`, `G[v]` is itself a dict, indexed 
    by the successors of `v`.  For any directed edge `v -> w`, `G[v][w]` 
    is the length of the edge from `v` to `w`.
        graph = {'a': {'b': 1}, 
                 'b': {'c': 2, 'b': 5}, 
                 'c': {'d': 1},
                 'd': {}}
        
    start: state state 
    y: vector of observation
    Returns two dicts, `dist` and `pred`:
        dist, pred = dijkstra(graph, start='a') 
    
   Returns 
   history: best path 
   '''
    
    D = defaultdict(float) 
    D[(start, start_frame)] = (- G.emission_probabilities[ start ][ y[start_frame] ] , -1)         # mapping of nodes to their dist from start
    Q = PQDict(D)           # priority queue for tracking min shortest path
    P = defaultdict(list)                # mapping of nodes to their direct predecessors
    #to_visit = set(  zip( G.adj.keys() , [0 for _ in range(len(G.adj.keys()))]  ) )       # unexplored node
    
    Q_by_state = [[] for _ in range(len(G.nodes))]  # we need to keep for each state an ordered list of (t, value) so that we can check the length and remove the best 
                                                    # each state will be dealt with by heapq
    new_medians = [(-1 , 1) for _ in range(len(G.nodes))]
    
    map_state2t = defaultdict(set)
    heapq.heappush( Q_by_state[start], (- G.emission_probabilities[ start ][ y[start_frame] ]   , start_frame) )
    map_state2t[start].add(start_frame) 
    
    visited = set() 
    success=False
    
    if last_frame == None:
        last_frame = len(y)-1 
       
    if n_frames == None:
        n_frames = len(y)
        
    T_floor = start_frame + floor((last_frame - start_frame + 1) / 2)

    print("T " + str(T_floor)) 
    print("start " + str(start))
    print("last frame " + str(last_frame)) 
    print("start frame " + str(start_frame)) 
    print("n frames " + str(n_frames)) 
    print("final state " + str(final_state))
    
    current_lower_bound = 0
    current_upper_bound = float("-inf") 
        
    while Q:             
                                         # nodes yet to explore
        (k, d_pred) = Q.popitem()   
          # pop nodap_state2t[k[0]]e w min dist d on frontier in constant time 
        d, pred = d_pred
        #print("k " + str(k)) 
        map_state2t[k[0]].remove(k[1])
        D[k]=d                           # est dijkstra greedy score
        visited.add(k)                   # remove from unexplored
        _ , _ = heapq.heappop(Q_by_state[k[0]]) # the gloabal best is also the best in that state 
        
        if k[1]==T_floor: 
            # print("new candidate ---- median " + str( (pred, k[0]) ))
            new_medians[k[0]] = (pred, k[0]) 

        elif k[1]>T_floor:
            new_medians[k[0]] = new_medians[pred]
        
        #print("visiting " + str(k)) 
        #print("len Q " + str(len(Q)))
        if (k[1] == last_frame and final_state is None) or (k[0] == final_state and k[1] == last_frame): 
            #success=True 
            x_a, x_b =  new_medians[k[0]]
            
            print("found median " + str((x_a, x_b)))
            
            N_left = floor(n_frames/2)              
            # y_left = y[:N_left] 
                        
            if N_left>1 : 
                 
                print("calling from the left with " + str(N_left) + " observations ") 
                # use bfs to find ancestors in N_left-1 steps 
                ancestors, adj , adj_inv,  current_lower_bound, current_upper_bound , current_lower_bound_reversed , current_upper_bound_reversed = BFS_ancestors(G, x_a,  N_left-1)
                # still using the entire state space 
                # continue recursion 
                
                #G2 = make_graph(all_nodes, all_edges)
                G.adj = adj
                G.adj_inv = adj_inv
                G.lower_bound = current_lower_bound
                G.upper_bound = current_upper_bound 
                G.lower_bound_reversed = current_lower_bound_reversed
                G.upper_bound_reversed = current_upper_bound_reversed
                dijkstra_dfs_sieve(G, start, y, final_state = x_a, start_frame = 0, last_frame = N_left-1, n_frames = N_left)
            
            
            N_right = n_frames- N_left 
          #  y_right = y[-N_right:]
            #in order print 
            print("median pair " + str((x_a, x_b)))
            
            if N_right >1: 
                
                print("calling from the right with " + str(N_right) + " observations ") 
                # use bfs to find descendants in N_right -1 steps 
                descendants, adj , adj_inv ,  current_lower_bound, current_upper_bound , current_lower_bound_reversed , current_upper_bound_reversed = BFS_descendants(G, x_b,  N_right-1)
                
                # continue recursion
                #G2 = make_graph(all_nodes, all_edges)
                G.adj = adj
                G.adj_inv = adj_inv
                G.lower_bound = current_lower_bound
                G.upper_bound = current_upper_bound 
                G.lower_bound_reversed = current_lower_bound_reversed
                G.upper_bound_reversed = current_upper_bound_reversed
                
                dijkstra_dfs_sieve(G, x_b, y, start_frame =  T_floor  , last_frame = T_floor +  N_right - 1 , n_frames = N_right)
            
            
            break
            
        # break                  # if the frame k[1] is T we arrived to the end 
        # now consider the edges from v with an unexplored head -
        # we may need to update the dist of unexplored successors 

        for w in G.adj[k[0]]:                          # successors to v
        
            if k[1]+1 > last_frame: 
                break # break for loop go to next while loop iteration 
        
            if ( w , k[1]+1 ) not in visited:        
                
                
                print("not visited")
                
                # then w is a frontier node
                d = D[k] - G.adj[k[0]][w]  - G.emission_probabilities[ w ][ y[k[1]+1] ]      # dgs: dist of start -> v -> w
                
                tmp = Q.get((w,k[1]+1), float("inf"))
                if isinstance(tmp, tuple): 
                    tmp = tmp[0] 
                
                if d < tmp:
                    
                    if len(map_state2t[w]) > 2 and k[1]+1 not in map_state2t[w]:
                        print("running dfs!!")                        
                        Q, Q_by_state, map_state2t = dfs(G, k[0], w, k[1]+1, d, Q_by_state, Q, map_state2t, y)  
                        print("dfs done")                            
                    
                    else:
                        # we already know that it is better 
                        Q[(w, k[1]+1)] = (d  , k[0])   
                        heapq.heappush( Q_by_state[w] , (d, k[1]+1) ) # but if there is already an entry for k[1]+1 this would be double counted 
                        map_state2t[w].add(k[1]+1)
                        #if len(Q_by_state[w]) > 1: # after adding w it has become 3 
                        #    print("running dfss") 
                        # run dfs 
                        #  this_d, this_t = heapq.heappop(Q_by_state[w])
                        
    
    r =  len(visited) / (len(G.nodes) * len(y))
    
    return D, r







def dijkstra_dfs_and_pruning(G, start, y):
    
    
    '''
    dijkstra's algorithm determines the length from `start` to every other 
    vertex in the graph.
    The graph argument `G` should be a dict indexed by nodes.  The value 
    of each item `G[v]` should also a dict indexed by successor nodes.
    In other words, for any node `v`, `G[v]` is itself a dict, indexed 
    by the successors of `v`.  For any directed edge `v -> w`, `G[v][w]` 
    is the length of the edge from `v` to `w`.
        graph = {'a': {'b': 1}, 
                 'b': {'c': 2, 'b': 5}, 
                 'c': {'d': 1},
                 'd': {}}
        
    start: state state 
    y: vector of observation
    Returns two dicts, `dist` and `pred`:
        dist, pred = dijkstra(graph, start='a') 
    
   Returns 
   history: best path 
   '''
    
    D = defaultdict(float) 
    D[(start, 0)] = - G.emission_probabilities[ start ][ y[0] ]          # mapping of nodes to their dist from start
    Q = PQDict(D)           # priority queue for tracking min shortest path
    P = defaultdict(list)                # mapping of nodes to their direct predecessors
    #to_visit = set(  zip( G.adj.keys() , [0 for _ in range(len(G.adj.keys()))]  ) )       # unexplored node
    
    Q_by_state = [[] for _ in range(len(G.nodes))]  # we need to keep for each state an ordered list of (t, value) so that we can check the length and remove the best 
                                                    # each state will be dealt with by heapq
    map_state2t = defaultdict(set)
    heapq.heappush( Q_by_state[start] , (- G.emission_probabilities[ start ][ y[0] ]   , 0) )
    map_state2t[start].add(0) 
    
    visited = set() 
    success=False
    T = len(y) 
    last_frame = T-1 
        
    while Q:             
                                         # nodes yet to explore
        (k, d) = Q.popitem()             # pop nodap_state2t[k[0]]e w min dist d on frontier in constant time 
        
        
        map_state2t[k[0]].remove(k[1])
        D[k]=d                           # est dijkstra greedy score
        visited.add(k)                   # remove from unexplored
       
        _ , _ = heapq.heappop(Q_by_state[k[0]]) # the gloabal best is also the best in that state 
        
        #print("visiting " + str(k)) 
        #print("len Q " + str(len(Q)))
        if k[1] == T-1: 
            success=True 
            break                  # if the frame k[1] is T we arrived to the end 

        # now consider the edges from v with an unexplored head -
        # we may need to update the dist of unexplored successors 
        for w in G.adj[k[0]]:                          # successors to v
        
            if ( w , k[1]+1 ) not in visited:        

                # then w is a frontier node
                d = D[k] - G.adj[k[0]][w]  - G.emission_probabilities[ w ][ y[k[1]+1] ]      # dgs: dist of start -> v -> w                
                
                if d < Q.get((w,k[1]+1), float("inf")):
                    
                    
                    if len(map_state2t[w]) > 25 and k[1]+1 not in map_state2t[w]:
                        
                        print("running dfs!!")
                        continue_flag = True 
                        Q, Q_by_state, map_state2t = dfs(G, k[0], w, k[1]+1, d, Q_by_state, Q, map_state2t, y, continue_flag) 
                        print("dfs done.. ")
                                                
                    else:
                        # we already know that it is better 
                        Q[(w, k[1]+1)] = d    
                        heapq.heappush( Q_by_state[w] , (d, k[1]+1) ) # but if there is already an entry for k[1]+1 this would be double counted 
                        map_state2t[w].add(k[1]+1)
                        
                        # prune if possible
                        Q_by_state, Q, map_state2t = prune_in_state(w, T, G, Q_by_state, Q, map_state2t)
                        
    
    r =  len(visited) / (len(G.nodes) * len(y))
 
    print("final node " + str(k[0]))
    
    return D, r





def prune_in_state(w, T, G, Q_by_state, Q, map_state2t): 
    # we can prune out longer path that even getting the best possible 
    # weight in the other frames are not 
    # capable of surviving 
    min_bound = float("inf") 
    min_t = 0 
    for cost, t in Q_by_state[w]: 
        # compute best possible scenario 
        this_bound = cost + (T - t) * G.upper_bound
        if this_bound < min_bound: 
            min_bound = this_bound
            min_t = t 
    
    # compare with all best case scenario 
    for cost, t in Q_by_state[w]: 
        this_bound = cost + (T - t) * G.lower_bound
        if this_bound > min_bound and t > min_t:  # the second condition makes sure that we are not pruning shorter paths 
                                                  # in case longer paths do not exist 
            # prune out
            Q_by_state[w].remove( (cost,t) )
            _ = Q.pop((w, t)) 
            map_state2t[w].remove(t) 
            
    
    return Q_by_state, Q, map_state2t





''' to do: we should make a class out of this ''' 
def dfs( G, pred, state, frame, cost, Q_by_state,  Q, map_state2t,  y, visited, space_budget, continue_flag, n_frames): 
                
    visited.add(state)
    
    if continue_flag:
        
        for neighbour in G.adj[state]:
            
            if frame < n_frames-1: 
                        
                # update cost 
                cost = cost - G.adj[state][neighbour]  - G.emission_probabilities[ state ][ y[frame+1] ]
    
                # check if we can insert state in the queue
                if len(map_state2t[neighbour])<space_budget or frame+1 in map_state2t[neighbour]:
                    
                    tmp = Q.get((neighbour,frame+1), float("inf"))
                    if isinstance(tmp, tuple): 
                        tmp = tmp[0] 
                        
                    if cost < tmp:
                        Q[(neighbour, frame+1)] = (cost , state)
                        heapq.heappush( Q_by_state[neighbour] , (cost, frame+1) )
                        map_state2t[neighbour].add(frame+1)
                        continue_flag=False
                    
                    return Q, Q_by_state,map_state2t, continue_flag
    
                else:
                    
                    if neighbour not in visited: 
                        dfs( G, state, neighbour, frame+1, cost, Q_by_state, Q, map_state2t, y, visited, space_budget, continue_flag , n_frames) # recurse 
    
            
            else: 
                
                # once last frame reached , we must terminate 
                # update cost in the last frame 
                tmp =  Q.get((state,frame), float("inf"))
                if isinstance(tmp, tuple): 
                    tmp = tmp[0] 
                                
                if cost < tmp: 
                    Q[(state, frame)] = (cost, pred)
                    heapq.heappush( Q_by_state[state] , (cost, frame) )
                    map_state2t[state].add(frame)
                    
                continue_flag=False
                                    
    return Q, Q_by_state, map_state2t, continue_flag ######## this should be the last visited node ################ 
    
    

>>>>>>> 153cb145f02d8e319d061cc3839923d57d7f34ce



def viterbi(G, start, y):
    """
<<<<<<< HEAD
    Viterbi algorithm - pull implementation
    G: graph  (graph)
    start: start node (int, str) 
    y: observation sequence (array) 
    
    Returns: 
    x: optimal paths (dict)
    T1; path probability table (array) 
=======
    Return the MAP estimate of state trajectory of Hidden Markov Model.

    Parameters
    
    Returns
    -------
>>>>>>> 153cb145f02d8e319d061cc3839923d57d7f34ce

    """
    
    
    K = len(G.nodes)
    T = len(y)
    T1 = np.zeros((K, T))
    T2 = np.zeros((K, T))


    # Initilaize the tracking tables from first observation
    Pi = np.array([float("-inf") if it!=start else 0 for it in G.nodes])
    
    T1[:, 0] = Pi +  G.emission_probabilities[:, y[0]] 
    T2[:, 0] = 0
    
<<<<<<< HEAD
=======
    
>>>>>>> 153cb145f02d8e319d061cc3839923d57d7f34ce
    # Iterate throught the observations updating the tracking tables
    for j in range(1, T): 
                
        for i in range(K): 
                        
            i_max = float("-inf") 
            best_neig = float("-inf")  
            for neig in G.adj_inv[i]: 
                this_likelihood = T1[neig , j - 1]  + G.adj_inv[i][neig] + G.emission_probabilities[i][y[j]] 
<<<<<<< HEAD
=======
                
>>>>>>> 153cb145f02d8e319d061cc3839923d57d7f34ce
                if this_likelihood > i_max: 
                    i_max = this_likelihood
                    best_neig = neig 
                
            T1[i,j] = i_max
            T2[i,j] = best_neig
                          
    # Build the output, optimal model trajectory by Backtracking 
    x = np.zeros(T, dtype=int)
<<<<<<< HEAD
    x[-1] = int(np.argmax(T1[:, T - 1]))
    for i in reversed(range(1, T)):
=======
    
    x[-1] = int(np.argmax(T1[:, T - 1]))

    for i in reversed(range(1, T)):
      
>>>>>>> 153cb145f02d8e319d061cc3839923d57d7f34ce
        x[i - 1] = T2[x[i], i]
    
    return x, T1



def viterbi_tp(G, start, y):
    """
<<<<<<< HEAD
    Viterbi algorithm - push (token-passing-like) implementation
    G: graph  (graph)
    start: start node (int, str) 
    y: observation sequence (array) 
    
    Returns: 
    x: optimal paths (dict)
    T1; path probability table (array) 
=======
    Return the MAP estimate of state trajectory of Hidden Markov Model.

    Parameters
    
    Returns
    -------

>>>>>>> 153cb145f02d8e319d061cc3839923d57d7f34ce
    """
    
    
    K = len(G.nodes)
    T = len(y)
    T1 = np.zeros((K, T))
    T2 = np.zeros((K, T))


    # Initilaize the tracking tables from first observation
    Pi = np.array([float("-inf") if it!=start else 0 for it in G.nodes])
    
    T1[:, 0] = Pi +  G.emission_probabilities[:, y[0]] 
<<<<<<< HEAD
    T2[:, 0] = 0    
=======
    T2[:, 0] = 0
    
>>>>>>> 153cb145f02d8e319d061cc3839923d57d7f34ce
    
    # Iterate through the observations updating the tracking tables
    for j in range(1, T): 
        
        this_j_T1 = [float("-inf")  for _ in range(K)] 
        this_j_T2 = [float("-inf")  for _ in range(K)] 
        
        for i in range(K): 
<<<<<<< HEAD
            for neig in G.adj[i]: 
                this_likelihood = T1[i , j - 1]  + G.adj[i][neig] + G.emission_probabilities[neig][y[j]] 
=======
            
            
            for neig in G.adj[i]: 
                this_likelihood = T1[i , j - 1]  + G.adj[i][neig] + G.emission_probabilities[neig][y[j]] 
                
>>>>>>> 153cb145f02d8e319d061cc3839923d57d7f34ce
                if this_likelihood > this_j_T1[neig]: 
                    this_j_T1[neig] = this_likelihood
                    this_j_T2[neig] = i  
            
<<<<<<< HEAD
=======
            
>>>>>>> 153cb145f02d8e319d061cc3839923d57d7f34ce
        T1[:,j] = this_j_T1
        T2[:,j] = this_j_T2
                      
    # Build the output, optimal model trajectory by Backtracking 
    x = np.zeros(T, dtype=int)
<<<<<<< HEAD
    x[-1] = int(np.argmax(T1[:, T - 1]))
    for i in reversed(range(1, T)):
=======
    
    x[-1] = int(np.argmax(T1[:, T - 1]))
    
    for i in reversed(range(1, T)):
      
>>>>>>> 153cb145f02d8e319d061cc3839923d57d7f34ce
        x[i - 1] = T2[x[i], i]
    
    return x, T1


<<<<<<< HEAD
def bidirectional_mint(G, start, y, last_nodes):
    '''
    bidirectional-mint algorithm
    Input: 
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
    D_f[(start, 0)] = - G.emission_probabilities[ start ][ y[0] ] 
    for node in last_nodes:
        D_b[(node, T-1)] = 0 
    Q_f = PQDict(D_f)          
    Q_b = PQDict(D_b)
    P_f = defaultdict(list)               
    P_b = defaultdict(list)  

    visited_f = set()      
    visited_b = set() 
    mu = float("inf") 
        
    while len(Q_f)>0 and len(Q_b)>0:     
        
        (k_f, d_f) = Q_f.popitem() 
        (k_b, d_b) = Q_b.popitem()                   
=======


def bidirectional_dijkstra(G, start, y, last_nodes):
    
    
    '''
    dijkstra's algorithm determines the length from `start` to every other 
    vertex in the graph.
    The graph argument `G` should be a dict indexed by nodes.  The value 
    of each item `G[v]` should also a dict indexed by successor nodes.
    In other words, for any node `v`, `G[v]` is itself a dict, indexed 
    by the successors of `v`.  For any directed edge `v -> w`, `G[v][w]` 
    is the length of the edge from `v` to `w`.
        graph = {'a': {'b': 1}, 
                 'b': {'c': 2, 'b': 5}, 
                 'c': {'d': 1},
                 'd': {}}
    Returns two dicts, `dist` and `pred`:
        dist, pred = dijkstra(graph, start='a') 
    
    `dist` is a dict mapping each node to its shortest distance from the
    specified starting node:
        assert dist == {'a': 0, 'c': 3, 'b': 1, 'd': 4}
    `pred` is a dict mapping each node to its predecessor node on the
    shortest path from the specified starting node:
        assert pred == {'b': 'a', 'c': 'b', 'd': 'c'}
    
    '''
    
    
    D_f = defaultdict(float) 
    D_b = defaultdict(float) 
    T = len(y) 
    D_f[(start, 0)] = - G.emission_probabilities[ start ][ y[0] ] # mapping of nodes to their dist from start
    for node in last_nodes:
        D_b[(node, T-1)] = - G.emission_probabilities[ node ][ y[-1] ]  # counting from 0 
    
    Q_f = PQDict(D_f)           # priority queue for tracking min shortest path
    Q_b = PQDict(D_b)
        
    P_f = defaultdict(list)                # mapping of nodes to their direct predecessors
    P_b = defaultdict(list)  
    #to_visit = set(  zip( G.adj.keys() , [0 for _ in range(len(G.adj.keys()))]  ) )       # unexplored node
    
    visited_f = set()       # unexplored node
    visited_b = set() 
    
    mu = float("inf") 
    #max_t_forw = 0 
    #max_t_back = T-1 
    #  print("backward start with " + str(Q_b))
    worst_emission = - np.min(G.emission_probabilities)
    
    #print("worst emission " + str(worst_emission)) 
        
    while len(Q_f)>0 and len(Q_b)>0:     
        
        #print("new iteration of the while loop .. ")
        
        #print("current forward queue " + str(Q_f))
        #print("current back queue " + str(Q_b))
        
        #print("current forward visited " + str(visited_f)) 
        #print("current backward visited " + str(visited_b))
                        #           # nodes yet to explore
        (k_f, d_f) = Q_f.popitem() 
        (k_b, d_b) = Q_b.popitem()                    # pop node w min dist d on frontier in constant time 
>>>>>>> 153cb145f02d8e319d061cc3839923d57d7f34ce
        D_f[k_f]=  d_f 
        D_b[k_b] = d_b      
        # update explored 
        visited_f.add(k_f)
<<<<<<< HEAD
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
            for w in G.adj_inv[k_b[0]]:  
=======
        visited_b.add(k_b)        
        
        #print("visiting in forward search " + str(k_f) + " of cost " + str(D_f[k_f])) 
        #print("visiting in backward search " + str(k_b) + " of cost " + str(D_b[k_b])) 
            
        if k_f[1] < T-1: 
            for w in G.adj[k_f[0]]:                          # successors to v
                if ( w , k_f[1]+1 ) not in visited_f:        
                    d = D_f[k_f] - G.adj[k_f[0]][w]  - G.emission_probabilities[ w ][ y[k_f[1]+1] ]      # dgs: dist of start -> v -> w
                    if d < Q_f.get((w,k_f[1]+1), float("inf")):
                        Q_f[(w, k_f[1]+1)] = d     
                        history = copy.deepcopy(P_f[k_f])
                        history.append(k_f)
                        P_f[(w, k_f[1]+1)] = history 
                        #if k_f[1]+1 > max_t_forw:
                        #    max_t_forw = k_f[1]+1np.min(B)
                                        
                
                if ( w , k_f[1]+1 ) in visited_b and D_f[k_f] - G.adj[k_f[0]][w]  + D_b[( w , k_f[1]+1 )] < mu and len(P_f[k_f]) + len(P_b[( w , k_f[1]+1 )]) == T-2:
                    #print("kf1 " + str(k_f[1]))
                    #print(" w " + str(w) + " t_back " + str(t_back)) 
                    mu = D_f[k_f] - G.adj[k_f[0]][w]  + D_b[( w , k_f[1]+1 )] 
                    best_path = P_f[k_f] + [(k_f[0], k_f[1]), (w, k_f[1]+1)] + P_b[( w , k_f[1]+1 )][::-1]
                    
                    #print("k_f " + str(k_f)) 
                    #print("k_b " + str( k_b))
                    
                    #print("found best path (forward) " + str(best_path) + " with cost " + str(mu)) 
                    
            
        if k_b[1] > 0: 
         
            for w in G.adj_inv[k_b[0]]:  # successors to v
                if ( w , k_b[1] - 1 ) not in visited_b: 
                    
                    d = D_b[k_b] - G.adj_inv[k_b[0]][w]  - G.emission_probabilities[ w ][ y[k_b[1]-1] ]      # dgs: dist of start -> v -> w
                    if d < Q_b.get((w,k_b[1]-1), float("inf")):
                        Q_b[(w, k_b[1]-1)] = d                             
                        history = copy.deepcopy(P_b[k_b])
                        history.append(k_b)
                        P_b[(w, k_b[1]-1)] = history  
                        #if k_b[1]-1 < max_t_back: 
                        #    max_t_back = k_b[1]-1
                                       
                if ( w, k_b[1]-1 ) in visited_f and D_b[k_b] - G.adj_inv[k_b[0]][w]  + D_f[(w, k_b[1]-1)] < mu and len(P_f[(w, k_b[1]-1)]) + len(P_b[k_b]) == T - 2:                    
                    
                    
                    mu = D_b[k_b] - G.adj_inv[k_b[0]][w]  + D_f[(w, k_b[1]-1)]
                    best_path = P_f[(w, k_b[1]-1)] +  [(w, k_b[1]-1), (k_b[0], k_b[1])] +  P_b[k_b][::-1] #P_f[(w, t_forw)] + [ (w,k_b[1]-1) , (k_b[0], k_b[1]) ] + P_b[k_b][::-1]
                   
                    #print("k_f " + str(k_f)) 
                    #print("k_b " + str( k_b))
                    #print("found best path (backward) " + str(best_path) + " with cost " + str(mu)) 
        
                    
        # check condition 
        if D_f[k_f] + D_b[k_b] >= mu + worst_emission : #and k_f[1] >= max_t_forw and k_b[1] <= max_t_back:
            print("Breaking!")
            break 
                 
    r =  len(visited_b.union(visited_f)) / (len(G.nodes) * len(y))
    
    return  best_path , r






def bidirectional_dijkstra2(G, start, y, last_nodes):
    '''
    dijkstra's algorithm determines the length from `start` to every other 
    vertex in the graph.
    The graph argument `G` should be a dict indexed by nodes.  The value 
    of each item `G[v]` should also a dict indexed by successor nodes.
    In other words, for any node `v`, `G[v]` is itself a dict, indexed 
    by the successors of `v`.  For any directed edge `v -> w`, `G[v][w]` 
    is the length of the edge from `v` to `w`.
        graph = {'a': {'b': 1}, 
                 'b': {'c': 2, 'b': 5}, 
                 'c': {'d': 1},
                 'd': {}}
    Returns two dicts, `dist` and `pred`:
        dist, pred = dijkstra(graph, start='a') 
    
    `dist` is a dict mapping each node to its shortest distance from the
    specified starting node:
        assert dist == {'a': 0, 'c': 3, 'b': 1, 'd': 4}
    `pred` is a dict mapping each node to its predecessor node on the
    shortest path from the specified starting node:
        assert pred == {'b': 'a', 'c': 'b', 'd': 'c'}
    '''
    
    D_f = defaultdict(float) 
    D_b = defaultdict(float) 
    T = len(y) 
    D_f[(start, 0)] = - G.emission_probabilities[ start ][ y[0] ] # mapping of nodes to their dist from start
    for node in last_nodes:
        D_b[(node, T-1)] = 0 
    Q_f = PQDict(D_f)           # priority queue for tracking min shortest path
    Q_b = PQDict(D_b)
    P_f = defaultdict(list)                # mapping of nodes to their direct predecessors
    P_b = defaultdict(list)  
    #to_visit = set(  zip( G.adj.keys() , [0 for _ in range(len(G.adj.keys()))]  ) )       # unexplored node
    visited_f = set()       # unexplored node
    visited_b = set() 
    mu = float("inf") 
    #max_t_forw = 0 
    #max_t_back = T-1 
    #  print("backward start with " + str(Q_b))            
    while len(Q_f)>0 and len(Q_b)>0:     
        #print("new iteration of the while loop .. ")
        # 
        #print("current forward queue " + str(Q_f))
        #print("current back queue " + str(Q_b))
        #print("current forward visited " + str(visited_f)) 
        #print("current backward visited " + str(visited_b))
        (k_f, d_f) = Q_f.popitem() 
        (k_b, d_b) = Q_b.popitem()                    # pop node w min dist d on frontier in constant time 
        D_f[k_f]=  d_f 
        D_b[k_b] = d_b      
        # update explored 
        visited_f.add(k_f)
        visited_b.add(k_b)   
        #print("visiting in forward search " + str(k_f) + " of cost " + str(D_f[k_f])) 
        #print("visiting in backward search " + str(k_b) + " of cost " + str(D_b[k_b]))              
        if k_f[1] < T-1: 
            for w in G.adj[k_f[0]]:                          # successors to v
                if ( w , k_f[1]+1 ) not in visited_f:        
                    d = D_f[k_f] - G.adj[k_f[0]][w]  - G.emission_probabilities[ w ][ y[k_f[1]+1] ]      # dgs: dist of start -> v -> w
                    if d < Q_f.get((w,k_f[1]+1), float("inf")):
                        Q_f[(w, k_f[1]+1)] = d     
                      #  history = copy.deepcopy(P_f[k_f])
                      #  history.append(k_f)
                        P_f[(w, k_f[1]+1)] = P_f[k_f] + [k_f] 
                        #if k_f[1]+1 > max_t_forw:
                        #    max_t_forw = k_f[1]+1np.min(B)
                                        
                
                if ( w , k_f[1]+1 ) in visited_b and D_f[k_f] - G.adj[k_f[0]][w]  + D_b[( w , k_f[1]+1 )] - G.emission_probabilities[ w ][ y[k_f[1]+1] ] < mu and len(P_f[k_f]) + len(P_b[( w , k_f[1]+1 )]) == T-2:
                    #print("kf1 " + str(k_f[1]))
                    #print(" w " + str(w) + " t_back " + str(t_back)) 
                    mu = D_f[k_f] - G.adj[k_f[0]][w]  + D_b[( w , k_f[1]+1 )]  - G.emission_probabilities[ w ][ y[k_f[1]+1] ]
                    best_path = P_f[k_f] + [(k_f[0], k_f[1]), (w, k_f[1]+1)] + P_b[( w , k_f[1]+1 )][::-1]
                    #print("k_f " + str(k_f)) 
                    #print("k_b " + str( k_b))w
                    
            
        if k_b[1] > 0: 
            for w in G.adj_inv[k_b[0]]:  # successors to v
>>>>>>> 153cb145f02d8e319d061cc3839923d57d7f34ce
                if ( w , k_b[1] - 1 ) not in visited_b: 
                    
                    d = D_b[k_b] - G.adj_inv[k_b[0]][w]  - G.emission_probabilities[ k_b[0] ][ y[k_b[1]] ]      # here we add the emission probability of the source and not of the destination 
                    if d < Q_b.get((w,k_b[1]-1), float("inf")):
                        Q_b[(w, k_b[1]-1)] = d                             
<<<<<<< HEAD
                        P_b[(w, k_b[1]-1)] = P_b[k_b] + [k_b]   
                       
                                       
                if ( w, k_b[1]-1 ) in visited_f and D_b[k_b] - G.emission_probabilities[ k_b[0] ][ y[k_b[1]] ] - G.adj_inv[k_b[0]][w]  + D_f[(w, k_b[1]-1)] < mu and len(P_f[(w, k_b[1]-1)]) + len(P_b[k_b]) == T - 2:                                        
                    mu = D_b[k_b]  - G.adj_inv[k_b[0]][w]  - G.emission_probabilities[ k_b[0] ][ y[k_b[1]] ] + D_f[(w, k_b[1]-1)] # new best path 
                    best_path = P_f[(w, k_b[1]-1)] +  [(w, k_b[1]-1), (k_b[0], k_b[1])] +  P_b[k_b][::-1] 
                   
        # check stopping condition 
        if D_f[k_f] + D_b[k_b] >= mu : 
            print("Breaking!")
            break 
                 
    
    return  best_path
=======
                       # history = copy.deepcopy(P_b[k_b])
                       # history.append(k_b)
                        P_b[(w, k_b[1]-1)] = P_b[k_b] + [k_b]   
                        #if k_b[1]-1 < max_t_back: 
                        #    max_t_back = k_b[1]-1
                                       
                if ( w, k_b[1]-1 ) in visited_f and D_b[k_b] - G.emission_probabilities[ k_b[0] ][ y[k_b[1]] ] - G.adj_inv[k_b[0]][w]  + D_f[(w, k_b[1]-1)] < mu and len(P_f[(w, k_b[1]-1)]) + len(P_b[k_b]) == T - 2:                                        
                    mu = D_b[k_b]  - G.adj_inv[k_b[0]][w]  - G.emission_probabilities[ k_b[0] ][ y[k_b[1]] ] + D_f[(w, k_b[1]-1)]
                    best_path = P_f[(w, k_b[1]-1)] +  [(w, k_b[1]-1), (k_b[0], k_b[1])] +  P_b[k_b][::-1] #P_f[(w, t_forw)] + [ (w,k_b[1]-1) , (k_b[0], k_b[1]) ] + P_b[k_b][::-1]
                    #print("k_f " + str(k_f)) 
                    #print("k_b " + str( k_b))
                   
        # check condition 
        if D_f[k_f] + D_b[k_b] >= mu : #and k_f[1] >= max_t_forw and k_b[1] <= max_t_back:
            print("Breaking!")
            break 
                 
    r =  len(visited_b.union(visited_f)) / (len(G.nodes) * len(y))
    
    return  best_path , r

>>>>>>> 153cb145f02d8e319d061cc3839923d57d7f34ce






def bidirectional_dijkstra_bound(G, start, y, last_nodes):
    '''
<<<<<<< HEAD
    bidirectional-mint-bound
    Input: 
    G: graph  (graph)
    start: start node (int, str) 
    y: observation sequence (array) 
    last_nodes: possible last states (e.g., set of nodes reachable in T steps) 
    
    Returns: 
    best path: path (list)
=======
    dijkstra's algorithm determines the length from `start` to every other 
    vertex in the graph.
    The graph argument `G` should be a dict indexed by nodes.  The value 
    of each item `G[v]` should also a dict indexed by successor nodes.
    In other words, for any node `v`, `G[v]` is itself a dict, indexed 
    by the successors of `v`.  For any directed edge `v -> w`, `G[v][w]` 
    is the length of the edge from `v` to `w`.
        graph = {'a': {'b': 1}, 
                 'b': {'c': 2, 'b': 5}, 
                 'c': {'d': 1},
                 'd': {}}
    Returns two dicts, `dist` and `pred`:
        dist, pred = dijkstra(graph, start='a') 
    
    `dist` is a dict mapping each node to its shortest distance from the
    specified starting node:
        assert dist == {'a': 0, 'c': 3, 'b': 1, 'd': 4}
    `pred` is a dict mapping each node to its predecessor node on the
    shortest path from the specified starting node:
        assert pred == {'b': 'a', 'c': 'b', 'd': 'c'}
>>>>>>> 153cb145f02d8e319d061cc3839923d57d7f34ce
    '''
    
    D_f = defaultdict(float) 
    D_b = defaultdict(float) 
    T = len(y) 
<<<<<<< HEAD
    D_f[(start, 0)] =  -G.upper_bound * T 
    for node in last_nodes:
        D_b[(node, T-1)] = -G.upper_bound_reversed * T 
    Q_f = PQDict(D_f)          
    Q_b = PQDict(D_b)
    P_f = defaultdict(list)    
    P_b = defaultdict(list)  
    visited_f = set()      
    visited_b = set() 
    mu = float("inf")       
    while len(Q_f)>0 and len(Q_b)>0:     
        (k_f, d_f) = Q_f.popitem() 
        (k_b, d_b) = Q_b.popitem()                   
=======
    D_f[(start, 0)] =  -G.upper_bound * T # - G.emission_probabilities[ start ][ y[0] ] # mapping of nodes to their dist from start
    for node in last_nodes:
        D_b[(node, T-1)] = -G.upper_bound_reversed * T 
    Q_f = PQDict(D_f)           # priority queue for tracking min shortest path
    Q_b = PQDict(D_b)
    P_f = defaultdict(list)     # mapping of nodes to their direct predecessors
    P_b = defaultdict(list)  
    #to_visit = set(  zip( G.adj.keys() , [0 for _ in range(len(G.adj.keys()))]  ) )       # unexplored node
    visited_f = set()       # unexplored node
    visited_b = set() 
    mu = float("inf") 
    #max_t_forw = 0 
    #max_t_back = T-1 
    #  print("backward start with " + str(Q_b))            
    while len(Q_f)>0 and len(Q_b)>0:     
        #print("new iteration of the while loop .. ")
        #print("current forward queue " + str(Q_f))
        #print("current back queue " + str(Q_b))
        #print("current forward visited " + str(visited_f)) 
        #print("current backward visited " + str(visited_b))
        (k_f, d_f) = Q_f.popitem() 
        (k_b, d_b) = Q_b.popitem()                    # pop node w min dist d on frontier in constant time 
>>>>>>> 153cb145f02d8e319d061cc3839923d57d7f34ce
        D_f[k_f]=  d_f 
        D_b[k_b] = d_b      
        # update explored 
        visited_f.add(k_f)
<<<<<<< HEAD
        visited_b.add(k_b)        
        if k_f[1] < T-1: 
            for w in G.adj[k_f[0]]:                          
                if ( w , k_f[1]+1 ) not in visited_f:        
                    d = D_f[k_f] + G.upper_bound - G.adj[k_f[0]][w]  - G.emission_probabilities[ w ][ y[k_f[1]+1] ]     
                    if d < Q_f.get((w,k_f[1]+1), float("inf")):
                        Q_f[(w, k_f[1]+1)] = d     
=======
        visited_b.add(k_b)   
        #print("visiting in forward search " + str(k_f) + " of cost " + str(D_f[k_f])) 
        #print("visiting in backward search " + str(k_b) + " of cost " + str(D_b[k_b]))              
        if k_f[1] < T-1: 
            for w in G.adj[k_f[0]]:                          # successors to v
                if ( w , k_f[1]+1 ) not in visited_f:        
                    d = D_f[k_f] + G.upper_bound - G.adj[k_f[0]][w]  - G.emission_probabilities[ w ][ y[k_f[1]+1] ]      # dgs: dist of start -> v -> w
                    if d < Q_f.get((w,k_f[1]+1), float("inf")):
                        Q_f[(w, k_f[1]+1)] = d     
                  #      history = copy.deepcopy(P_f[k_f])
                  #      history.append(k_f)
>>>>>>> 153cb145f02d8e319d061cc3839923d57d7f34ce
                        P_f[(w, k_f[1]+1)] = P_f[k_f] + [k_f]  
                                        
                
                actual_cost_forward = D_f[k_f] + (T - 1 - k_f[1]) * G.upper_bound
                actual_cost_backward = D_b[( w , k_f[1]+1 )] + (T - 1 -  k_f[1] - 1) * G.upper_bound
                if ( w , k_f[1]+1 ) in visited_b and actual_cost_forward - G.adj[k_f[0]][w]  + actual_cost_backward - G.emission_probabilities[ w ][ y[k_f[1]+1] ] < mu and len(P_f[k_f]) + len(P_b[( w , k_f[1]+1 )]) == T-2:
                    mu = actual_cost_forward - G.adj[k_f[0]][w]  + actual_cost_backward  - G.emission_probabilities[ w ][ y[k_f[1]+1] ]
                    best_path = P_f[k_f] + [(k_f[0], k_f[1]), (w, k_f[1]+1)] + P_b[( w , k_f[1]+1 )][::-1]
                    
            
        if k_b[1] > 0: 
<<<<<<< HEAD
            for w in G.adj_inv[k_b[0]]:  
=======
            for w in G.adj_inv[k_b[0]]:  # successors to v
>>>>>>> 153cb145f02d8e319d061cc3839923d57d7f34ce
                if ( w , k_b[1] - 1 ) not in visited_b: 
                    
                    d = D_b[k_b] + G.upper_bound - G.adj_inv[k_b[0]][w] - G.emission_probabilities[ k_b[0] ][ y[k_b[1]] ]      # here we add the emission probability of the source and not of the destination 
                    if d < Q_b.get((w,k_b[1]-1), float("inf")):
                        Q_b[(w, k_b[1]-1)] = d                             
<<<<<<< HEAD
=======
                       # history = copy.deepcopy(P_b[k_b])
                       # history.append(k_b)
>>>>>>> 153cb145f02d8e319d061cc3839923d57d7f34ce
                        P_b[(w, k_b[1]-1)] = P_b[k_b] + [k_b]   
                                     
                actual_cost_forward = D_f[(w, k_b[1]-1)] + (T - 1 - k_b[1] + 1) * G.upper_bound
                actual_cost_backward = D_b[k_b] + (T - 1 - k_b[1]) * G.upper_bound
                if ( w, k_b[1]-1 ) in visited_f and actual_cost_backward - G.emission_probabilities[ k_b[0] ][ y[k_b[1]] ] - G.adj_inv[k_b[0]][w]  + actual_cost_forward < mu and len(P_f[(w, k_b[1]-1)]) + len(P_b[k_b]) == T - 2:                                        
                    mu = actual_cost_backward - G.adj_inv[k_b[0]][w]  - G.emission_probabilities[ k_b[0] ][ y[k_b[1]] ] + actual_cost_forward
<<<<<<< HEAD
                    best_path = P_f[(w, k_b[1]-1)] +  [(w, k_b[1]-1), (k_b[0], k_b[1])] +  P_b[k_b][::-1] 

                    
        # check stopping condition 
        if D_f[k_f] + D_b[k_b] >= mu :
            print("Breaking!")
            break 
                 

    
    return  best_path 



=======
                    best_path = P_f[(w, k_b[1]-1)] +  [(w, k_b[1]-1), (k_b[0], k_b[1])] +  P_b[k_b][::-1] #P_f[(w, t_forw)] + [ (w,k_b[1]-1) , (k_b[0], k_b[1]) ] + P_b[k_b][::-1]

                    
        # check condition 
        if D_f[k_f] + D_b[k_b] >= mu : #and k_f[1] >= max_t_forw and k_b[1] <= max_t_back:
            print("Breaking!")
            break 
                 
    r =  len(visited_b.union(visited_f)) / (len(G.nodes) * len(y))
    
    return  best_path , r













def bidirectional_dijkstra_no_early_stop(G, start, y, last_nodes):
    
    
    '''
    dijkstra's algorithm determines the length from `start` to every other 
    vertex in the graph.
    The graph argument `G` should be a dict indexed by nodes.  The value 
    of each item `G[v]` should also a dict indexed by successor nodes.
    In other words, for any node `v`, `G[v]` is itself a dict, indexed 
    by the successors of `v`.  For any directed edge `v -> w`, `G[v][w]` 
    is the length of the edge from `v` to `w`.
        graph = {'a': {'b': 1}, 
                 'b': {'c': 2, 'b': 5}, 
                 'c': {'d': 1},
                 'd': {}}
    Returns two dicts, `dist` and `pred`:
        dist, pred = dijkstra(graph, start='a') 
    
    `dist` is a dict mapping each node to its shortest distance from the
    specified starting node:
        assert dist == {'a': 0, 'c': 3, 'b': 1, 'd': 4}
    `pred` is a dict mapping each node to its predecessor node on the
    shortest path from the specified starting node:
        assert pred == {'b': 'a', 'c': 'b', 'd': 'c'}
    
    '''
    
    
    D_f = defaultdict(float) 
    D_b = defaultdict(float) 
    T = len(y) 
    D_f[(start, 0)] = - G.emission_probabilities[ start ][ y[0] ] # mapping of nodes to their dist from start
    for node in last_nodes:
        D_b[(node, T-1)] = - G.emission_probabilities[ node ][ y[-1] ]  # counting from 0 
    
    Q_f = PQDict(D_f)           # priority queue for tracking min shortest path
    Q_b = PQDict(D_b)
        
    P_f = defaultdict(list)                # mapping of nodes to their direct predecessors
    P_b = defaultdict(list)  
    #to_visit = set(  zip( G.adj.keys() , [0 for _ in range(len(G.adj.keys()))]  ) )       # unexplored node
    
    visited_f = set()       # unexplored node
    visited_b = set() 
    
    mu = float("inf") 
    max_t_forw = 0 
    max_t_back = T-1 
    #  print("backward start with " + str(Q_b))
        
    while len(Q_f)>0 and len(Q_b)>0:     
        
        #print("new iteration of the while loop .. ")
        
        #print("current forward queue " + str(Q_f))
        #print("current back queue " + str(Q_b))
                                   # nodes yet to explore
        (k_f, d_f) = Q_f.popitem() 
        (k_b, d_b) = Q_b.popitem()                    # pop node w min dist d on frontier in constant time 
        D_f[k_f]=  d_f 
        D_b[k_b] = d_b      
        # update explored 
        visited_f.add(k_f)
        visited_b.add(k_b)
        
     
        #print("visiting in forward search " + str(k_f) + " of cost " + str(D_f[k_f])) 
        #print("visiting in backward search " + str(k_b) + " of cost " + str(D_b[k_b])) 

        
        
        
         
            
        if k_f[1] < T-1: 
            for w in G.adj[k_f[0]]:                          # successors to v
                if ( w , k_f[1]+1 ) not in visited_f:        
                    d = D_f[k_f] - G.adj[k_f[0]][w]  - G.emission_probabilities[ w ][ y[k_f[1]+1] ]      # dgs: dist of start -> v -> w
                    if d < Q_f.get((w,k_f[1]+1), float("inf")):
                        Q_f[(w, k_f[1]+1)] = d     
                        history = copy.deepcopy(P_f[k_f])
                        history.append(k_f)
                        P_f[(w, k_f[1]+1)] = history 
                        if k_f[1]+1 > max_t_forw:
                            max_t_forw = k_f[1]+1
                                        
                
                
                if ( w , k_f[1]+1 ) in visited_b and D_f[k_f] - G.adj[k_f[0]][w]  + D_b[( w , k_f[1]+1 )] < mu and len(P_f[k_f]) + len(P_b[( w , k_f[1]+1 )]) == T-2:
                    #print("kf1 " + str(k_f[1]))
                    #print(" w " + str(w) + " t_back " + str(t_back)) 
                    mu = D_f[k_f] - G.adj[k_f[0]][w]  + D_b[( w , k_f[1]+1 )] 
                    best_path = P_f[k_f] + [(k_f[0], k_f[1]), (w, k_f[1]+1)] + P_b[( w , k_f[1]+1 )][::-1]
                    
                    #print("k_f " + str(k_f)) 
                    #print("k_b " + str( k_b))
                    
                    #print("found best path (forward) " + str(best_path) + " with cost " + str(mu)) 
                    
            
        if k_b[1] > 0: 
         
            for w in G.adj_inv[k_b[0]]:  # successors to v
                if ( w , k_b[1] - 1 ) not in visited_b: 
                    
                    d = D_b[k_b] - G.adj_inv[k_b[0]][w]  - G.emission_probabilities[ w ][ y[k_b[1]-1] ]      # dgs: dist of start -> v -> w
                    if d < Q_b.get((w,k_b[1]-1), float("inf")):
                        Q_b[(w, k_b[1]-1)] = d                             
                        history = copy.deepcopy(P_b[k_b])
                        history.append(k_b)
                        P_b[(w, k_b[1]-1)] = history  
                        if k_b[1]-1 < max_t_back: 
                            max_t_back = k_b[1]-1
                                       
                if ( w, k_b[1]-1 ) in visited_f and D_b[k_b] - G.adj_inv[k_b[0]][w]  + D_f[(w, k_b[1]-1)] < mu and len(P_f[(w, k_b[1]-1)]) + len(P_b[k_b]) == T - 2:                    
                    
                    
                    mu = D_b[k_b] - G.adj_inv[k_b[0]][w]  + D_f[(w, k_b[1]-1)]
                    best_path = P_f[(w, k_b[1]-1)] +  [(w, k_b[1]-1), (k_b[0], k_b[1])] +  P_b[k_b][::-1] #P_f[(w, t_forw)] + [ (w,k_b[1]-1) , (k_b[0], k_b[1]) ] + P_b[k_b][::-1]
                   
                    #print("k_f " + str(k_f)) 
                    #print("k_b " + str( k_b))
                    #print("found best path (backward) " + str(best_path) + " with cost " + str(mu)) 
        
                    
       
    r =  len(visited_b.union(visited_f)) / (len(G.nodes) * len(y))
    
    return  best_path , r



def shortest_path(G, start, end):
    dist, pred = dijkstra(G, start, end)
    v = end
    path = [v]
    while v != start:
        v = pred[v]
        path.append(v)        
    path.reverse()
    return path


def prune_queue(G, Q, state, T): 
    
    '''
    we need to get all the elements that have the same state as the input one and prune accordingly
    
    '''
    
    pairs = Q[state] # dict frame to cost 
    # find minimum frame by value 
    min_frame = min(pairs.items(), key=lambda x: x[1] + (T-x[0]) * G.upper_bound)[0]
    #for k,v in pairs.items(): 
    best_bound = G.upper_bound * (T - min_frame) + Q[state][min_frame] 
        
    for k,v in pairs.items(): 
        if k!=min_frame: 
            this_value =  G.lower_bound * (T - k[0]) + v 
            if this_value >= best_bound: 
                Q[state].pop(k)
        
    
    
    return Q 
    
    


def make_graph(nodes, edges):
    '''
    for testing 
    '''
    G = Graph() 
    for node in nodes: 
        G.add_node(node) 
    for edge in edges: 
        G.add_edge(edge[0], edge[1], edge[2]) 
                
    return G 
>>>>>>> 153cb145f02d8e319d061cc3839923d57d7f34ce
    
