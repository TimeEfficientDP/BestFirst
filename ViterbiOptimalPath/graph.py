from collections import defaultdict 
import numpy as np 

class Graph: 
    
    def __init__(self): 
        self.adj = defaultdict(dict) 
        self.adj_inv = defaultdict(dict) 
        self.emission_probabilities = None
        self.lower_bound = None
        self.upper_bound = None 
        self.nodes = [] 
        self.node2idx = dict() 
        
            
    def add_node(self, u): 
        self.adj[u] = dict() # not really needed 
        self.adj_inv[u] = dict() 
        self.nodes.append(u) 

    def add_edge(self, u, v, cost: float): 
        self.adj[u][v] = cost 
        self.adj_inv[v][u] = cost 
                    
    def evaluate_edge_min(self, u,v): 
        return self.adj[u][v] + min(self.emission_probabilities[v]) 
        
        
    def evaluate_edge_max(self, u,v): 
        return self.adj[u][v] + max(self.emission_probabilities[v])
    
    def evaluate_edge_min_reversed(self, u,v): 
        return self.adj[u][v] + min(self.emission_probabilities[u]) 
        
        
    def evaluate_edge_max_reversed(self, u,v): 
        return self.adj[u][v] + max(self.emission_probabilities[u])
    
    
    
    def evaluate_edge_min_dict(self, u,v): 
        if v in self.emission_probabilities.keys(): 
            return self.adj[u][v] + min(self.emission_probabilities[v].values())
        else:
            return self.adj[u][v] 
        
        
    def evaluate_edge_max_dict(self, u,v): 
        if v in self.emission_probabilities.keys(): 
            return self.adj[u][v] + max(self.emission_probabilities[v].values())
        else:
            return self.adj[u][v]
        
        
        
    def evaluate_edge_min_dict_reversed(self, u,v): 
        if v in self.emission_probabilities.keys(): 
            return self.adj[u][v] + min(self.emission_probabilities[u].values())
        else:
            return self.adj[u][v] 
        
        
        
    def evaluate_edge_max_dict_reversed(self, u,v): 
        if u in self.emission_probabilities.keys(): 
            return self.adj[u][v] + max(self.emission_probabilities[u].values())
        else:
            return self.adj[u][v]
        
        
    def evaluate_edge_max_dict_decoding(self, u,v, new_emission_probabilities): 
        if (u,v) in self.emission_probabilities.keys(): 
             return self.adj[u][v] + max(new_emission_probabilities[(u,v)].values())
        else:
             return self.adj[u][v]
    
     
    def evaluate_edge_max_dict_reversed_decoding(self, u,v, new_emission_probabilities): 
        if (u,v) in self.emission_probabilities.keys(): 
            return self.adj[u][v] + max(new_emission_probabilities[(u,v)].values())
        else:
            return self.adj[u][v]
        
         
        
    
    def find_bounds(self, s):
        
        ''' scan the input graph in order to find the best and worst probabilities of transition to a given node (which 
        also include emission probabilities if the destination node is emitting '''
         
        # Mark all the vertices as not visited
        visited = set()
 
        # Create a queue for BFS
        queue = []
 
        # Mark the source node as
        # visited and enqueue it
        queue.append(s)
        visited.add(s) 
        
        current_lower_bound = 0
        current_upper_bound = float("-inf") 
        
        
        
        while queue:
            
 
            # Dequeue a vertex from
            # queue and print it
            s = queue.pop(0) 
            # Get all adjacent vertices of the
            # dequeued vertex s. If a adjacent
            # has not been visited, then mark it
            # visited and enqueue it
                        
            for i in self.adj[s]:
                                
                this_edge_cost_min = self.evaluate_edge_min(s, i) 
                                
                if this_edge_cost_min < current_lower_bound: 
                    current_lower_bound=this_edge_cost_min
                    
                this_edge_cost_max = self.evaluate_edge_max(s, i) 
                                
                if this_edge_cost_max > current_upper_bound: 
                    current_upper_bound = this_edge_cost_max
                
                if i not in visited:
                    queue.append(i)
                    
            visited.add(s) 
                        
        
        return current_lower_bound, current_upper_bound
    
    
    
    
    
    def find_bounds_reversed(self, s):
        
        ''' scan the input graph in order to find the best and worst probabilities of transition to a given node (which 
        also include emission probabilities if the destination node is emitting '''
         
        # Mark all the vertices as not visited
        visited = set()
        # Create a queue for BFS
        queue = []
        # Mark the source node as
        # visited and enqueue it
        queue.append(s)
        visited.add(s) 
        
        current_lower_bound = 0
        current_upper_bound = float("-inf") 
                
        while queue:
            # Dequeue a vertex from
            # queue and print it
            s = queue.pop(0) 
            # Get all adjacent vertices of the
            # dequeued vertex s. If a adjacent
            # has not been visited, then mark it
            # visited and enqueue it
            for i in self.adj[s]:
                
                this_edge_cost_min = self.evaluate_edge_min_reversed(s, i) 
                if this_edge_cost_min < current_lower_bound: 
                    current_lower_bound=this_edge_cost_min
                  
                this_edge_cost_max = self.evaluate_edge_max_reversed(s, i) 
                if this_edge_cost_max > current_upper_bound: 
                    current_upper_bound = this_edge_cost_max
                
                if i not in visited:
                    queue.append(i)
                    
            visited.add(s) 
                        
        
        return current_lower_bound, current_upper_bound
    
    
    
    

    
    def find_bounds_multiple_sources(self):
        
        ''' scan the input graph in order to find the best and worst probabilities of transition to a given node (which 
        also include emission probabilities if the destination node is emitting '''
        
        
        current_lower_bound=0
        current_upper_bound=float("-inf") 
        
        current_lower_bound_reversed=0
        current_upper_bound_reversed=float("-inf") 
        
        for k, v in self.adj.items():  
            for k2,v2 in v.items(): 

                this_edge_cost_min = self.evaluate_edge_min_dict(k, k2) 
                                
                if this_edge_cost_min < current_lower_bound: 
                    current_lower_bound=this_edge_cost_min
                    
                this_edge_cost_max = self.evaluate_edge_max_dict(k, k2) 
                
                this_edge_cost_max_reversed = self.evaluate_edge_max_dict_reversed(k, k2) 
                                                
                if this_edge_cost_max > current_upper_bound: 
                    current_upper_bound = this_edge_cost_max
                    
                if this_edge_cost_max_reversed > current_upper_bound_reversed:
                    current_upper_bound_reversed = this_edge_cost_max_reversed
                        
        return current_upper_bound, current_upper_bound_reversed
    
    
    
    def find_bounds_decoding(self, new_emission_probabilities):
        
        ''' scan the input graph in order to find the best and worst probabilities of transition to a given node (which 
        also include emission probabilities if the destination node is emitting '''
                
        current_lower_bound=0
        current_upper_bound=float("-inf") 
        
        current_lower_bound_reversed=0
        current_upper_bound_reversed=float("-inf") 
        
        for k, v in self.adj.items():  
            for k2,v2 in v.items(): 
                    
                this_edge_cost_max = self.evaluate_edge_max_dict_decoding(k, k2, new_emission_probabilities) 
                
                
                this_edge_cost_max_reversed = self.evaluate_edge_max_dict_reversed_decoding(k, k2, new_emission_probabilities) 
                                                
                if this_edge_cost_max > current_upper_bound: 
                    current_upper_bound = this_edge_cost_max
                    
                if this_edge_cost_max_reversed > current_upper_bound_reversed:
                    current_upper_bound_reversed = this_edge_cost_max_reversed
                        
        return current_upper_bound, current_upper_bound_reversed
    
    
    
    
    def find_refined_bounds_decoding(self, T, sources, n):
        
        ''' scan the input graph in order to find the best and worst probabilities of transition to a given node (which 
        also include emission probabilities if the destination node is emitting '''
        
        frame = dict() 
        
        # these are upper bounds in terms of negative log-likelihood (lower bound on cost)
        UB = np.ones(n) * float("-inf") 
        UB_reversed = np.ones(n) *float("-inf") 
        
        
        for start in sources: 
            frame[start] = 0 
        
        visited = set() 
        queue = [] 
        visited.update(set(sources)) 
        queue.extend(sources) 
        
        predecessors = defaultdict(set) 
        
        for start in sources: 
            frame[start] = 0 
        
        while len(queue)>0: 
            v = queue.pop(0) 
            for neig in self.adj[v]: 
                
                if frame[v]<T: # make sure we do not add neighbours once we reach K states 
                    
                    frame[neig] = frame[v]+1
                    
                    predecessors[neig].add(v) 
                    predecessors[neig].update( predecessors[v] )
                    
                 
                    this_edge_cost = self.adj[v][neig] + self.emission_probabilities[frame[neig]][(v,neig)]
                    
                    this_edge_cost_reversed = self.adj[v][neig] + self.emission_probabilities[frame[v]][(v,neig)]
                    
                    if this_edge_cost > UB[v]: 
                        UB[v] = this_edge_cost
                    if this_edge_cost_reversed > UB_reversed[v]: 
                        UB_reversed[v] = this_edge_cost_reversed
                        
                    # we need now to udpate all predecessors of V 
                    for pred in predecessors[v]: 
                        if this_edge_cost > UB[pred]: 
                            UB[pred] = this_edge_cost
                        if this_edge_cost_reversed > UB_reversed[pred]: 
                            UB_reversed[pred] = this_edge_cost_reversed
                            
                            
                    queue.append(neig) 
                    visited.add(neig)
                    
                        
        return UB, UB_reversed
    
    
    
    
    def find_T_reachability_set_no(self, s, T):
        
        ''' scan the input graph in order to find the best and worst probabilities of transition to a given node (which 
        also include emission probabilities if the destination node is emitting '''
         
        output_nodes = []
        # Mark all the vertices as not visited
        visited = set()
        # Create a queue for BFS
        queue = []
        # Mark the source node as
        # visited and enqueue it
        queue.append(  (s,0) )
        visited.add(  (s,0)  )       
        ########################Se esco prima delle 21 si ma non son sicuro

        while queue:
            # Dequeue a vertex from
            # queue and print it
            s,t = queue.pop(0) 
            print("s " + str(s) + " t " + str(t)) 
            # Get all adjacent vertices of the
            # dequeued vertex s. If a adjacent
            # has not been visited, then mark it
            # visited and enqueue it
            for i in self.adj[s]:
                if i not in visited:        
                    queue.append((i, t+1))
                    visited.add( (i, t+1) ) 
                    if t+1 == T: 
                        output_nodes.append( i )
            
                        
        return output_nodes



    def find_T_reachability_set(self, source, T): 
        
       ''' 
       Compute T hop neighbours of source  
        
       params: 
           source: starting state 
           T : number of hops 
            
       return: 
           reachable_T_set : set 
       ''' 
       
       reachable_T_set = set() 
       
       current_set_of_neighbours = set() 
       for k in self.adj[source]: 
           current_set_of_neighbours.add(k) 
           
       for i in range(T-2):
           
           new_neighbours = set() 
           for node in current_set_of_neighbours:
               for k in self.adj[node]: 
                   new_neighbours.add(k) 
        
           current_set_of_neighbours = new_neighbours    
           #print("current set of neighbours " + str(current_set_of_neighbours) +" iteration " + str(i+2))
       
       return current_set_of_neighbours
   
    
   
    def find_T_reachability_set_multiple_sources(self, sources, T): 
        
       ''' 
       Compute T hop neighbours of source  
        
       params: 
           source: starting state 
           T : number of hops 
            
       return: 
           reachable_T_set : set 
       ''' 
       
       
       all_reachable = set() 
       
       for source in sources: 
       
           current_set_of_neighbours = set() 
           for k in self.adj[source]: 
               current_set_of_neighbours.add(k) 
               
           for i in range(T-2):
               
               new_neighbours = set() 
               for node in current_set_of_neighbours:
                   for k in self.adj[node]: 
                       new_neighbours.add(k) 
            
               current_set_of_neighbours = new_neighbours    
               #print("current set of neighbours " + str(current_set_of_neighbours) +" iteration " + str(i+2))
           
           all_reachable.update(current_set_of_neighbours) 
           
       return all_reachable