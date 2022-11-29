'''
this script provides a minimal example comparing mint and Viterbi in synthetic ER data and testing
the correctness of MINT 
'''
import numpy as np 
import random
from bestFirstViterbi import mint, viterbi, viterbi_tp, mint_bound, bidirectional_mint, bidirectional_mint_bound
import time 
from graph import Graph

def generate_ER(n_nodes: int, sd:int , prob:float): 
    
    np.random.seed(sd) 
    all_nodes = [x for x in range(n_nodes)]
    all_edges = []
    
    for state in range(n_nodes): 
        
        edge_per_node=np.random.binomial(n_nodes-1, p=prob, size=None)
                
        #we sample @edge_per_node edges to connect to current state 
        state_connections = np.random.choice(all_nodes, size=edge_per_node)
        
        #sample probabilities  
        ps = -1 * np.random.uniform(0.001,1000, size=edge_per_node)
        
        for i in range(edge_per_node): 
            connection = state_connections[i]
            p = ps[i]
            all_edges.append((state, connection, p))
            
    return all_nodes, all_edges



def create_B(n_observables = 50, n_states = 100, sd = 1, lb = 0.001, ub = 10): 
    
    ''' create matrix of uniform emission probabilities '''
    
    np.random.seed(sd) 
        
    B = np.random.uniform(low=lb, high=ub, size=(n_states,n_observables))
    
    return -1*B 


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

        
        
if __name__ == '__main__': 
      
     density = 0.01
     n_paths = 100
     space_budget = 500 
     n_observables  = 50 
     K = 500 
     T = 10 
    
     random.seed(1)
     # vector of observations
     y = [random.randint(0,n_observables-1) for _ in range(T)] 
     # generate simple data 
     all_nodes, all_edges = generate_ER(n_nodes = K, sd=1, prob = density)                    
     G = make_graph(all_nodes, all_edges)
     B = create_B(n_observables = n_observables, n_states = K, sd = 1, lb = 0.001, ub = 1000)
     G.emission_probabilities = B 
     G.lower_bound, G.upper_bound = G.find_bounds(0)
     G.lower_bound_reversed, G.upper_bound_reversed = G.find_bounds_reversed(0)
     
     print("Starting Vanilla Viterbi .. \n ") 
     start_time_vanilla = time.time() 
     P_Viterbi, T1 = viterbi_tp(G,0,y)
     final_time_vanilla = time.time() - start_time_vanilla 
    
     print("Starting Mint.. \n ") 
     start_time_dij = time.time() 
     D, P = mint(G, 0, y)
     final_time_dij = time.time() - start_time_dij
      
     P_Dij = [x[0] for x in P]
     assert list(P_Viterbi) == P_Dij # throw error if the paths are different 
      
   