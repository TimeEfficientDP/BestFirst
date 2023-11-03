'''
this script provides a comparison between different algorithms for Optimal Viterbi path

'''

import numpy as np 
import random
from BestFirstViterbi import mint, viterbi_tp, mint_bound, bidirectional_mint, bidirectional_mint_bound
from graph import Graph

def generate_ER(n_nodes: int, sd:int , prob:float): 
    ''' generate synthetic data with ER transitions ''' 
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


def create_G_by_paths(n_nodes: int, sd:int, T:int, n_paths: int, power_a: float): 
    ''' generate synthetic data with skewed transition probabilities ''' 
    #
    np.random.seed(sd) 
    all_nodes = [x for x in range(n_nodes)]
    all_edges = []
    #
    # draw path probabilities 
    path_probabilities = np.random.power(power_a, size=n_paths) * 10000
    #
    for path in range(n_paths): 
        #
        # generate one path of lenght T 
        path_nodes = [0] + random.choices(all_nodes, k=T-1)
        #
        this_path_probability = path_probabilities[path] 
        #
        probability_per_edge = - this_path_probability / (T-1) 
        #
        for i in range(len(path_nodes)-1): 
            all_edges.append( ( path_nodes[i], path_nodes[i+1], probability_per_edge) )
        #
        #
        #
    return all_nodes, all_edges


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
     n_observables  = 50 

    
     for K in [500, 1000, 2000, 4000, 8000, 16000]: # state space size 
            for T in [10]: #path length
                for i in range(5): # number of repetitions 
                     random.seed(i)
                     # vector of observations
                     y = [random.randint(0,n_observables-1) for _ in range(T)] 
                     # generate simple data 
                     all_nodes, all_edges = generate_ER(n_nodes = K, sd=1, prob = density)                    
                     #all_nodes, all_edges = create_G_by_paths(n_nodes = K , sd = 1 , T = T , n_paths = 100, power_a = 1) use this in order to generate data with skewed path likelihoods
                     G = make_graph(all_nodes, all_edges)
                     B = create_B(n_observables = n_observables, n_states = K, sd = 1, lb = 0.001, ub = 1000)
                     G.emission_probabilities = B 
                     G.lower_bound, G.upper_bound = G.find_bounds(0)
                     G.lower_bound_reversed, G.upper_bound_reversed = G.find_bounds_reversed(0)
                     
                     print("Starting Vanilla Viterbi .. \n ") 
                     P_Viterbi, T1 = viterbi_tp(G,0,y)
                    
                     print("Starting Mint.. \n ") 
                     D, P = mint(G, 0, y)
                     
                     print("Starting MintBound Viterbi .. \n ") 
                     D, P = mint_bound(G, 0, y)
                      
                     print("Starting Bi-directional Mint .. \n ") 
                     last_nodes = G.find_T_reachability_set(0,T)
                     P , optimal_cost = bidirectional_mint(G, 0, y, list(last_nodes)) # here for simplicity we assume last node is known 
                
                     print("Starting Bi-directional Mint Bound .. \n ") 
                     P_Dij_bidi_bound , optimal_cost = bidirectional_mint_bound(G, 0, y, list(last_nodes)) # here for simplicity we assume last node is known 
                   
                 
    
      
     
       
