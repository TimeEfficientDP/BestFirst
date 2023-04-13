'''
this script provides a comparison between different algorithms for Optimal Viterbi path
in decoding data 
'''

from BestFirstViterbi_Decoding import mint, viterbi_tp, mint_bound, bidirectional_mint, bidirectional_mint_bound
from graph import Graph 
from collections import defaultdict



def forcedAlignment2Graph(transitions_dict_out, emission_probabilities, allstates):
    
    '''
    utility for decoding data   
    '''
    
    G = Graph()
    for k, v in transitions_dict_out.items(): 
        for neig in v: 
            G.adj[k][neig[0]] = neig[1] 
            G.adj_inv[neig[0]][k] = neig[1] 

    G.emission_probabilities = emission_probabilities
    
    new_emission_probabilities = invert_double_dict(emission_probabilities)
    G.upper_bound, G.upper_bound_reversed = G.find_bounds_decoding(new_emission_probabilities)    
    idx = 0 
    for state in allstates: 
        G.node2idx[state] = idx 
        idx+=1 
        G.nodes.append(state)
    return G 


def invert_double_dict(emission_probabilities):
    '''
    utility for decoding data   
    '''
    new_emission_probabilities = defaultdict(lambda: defaultdict(float))
    for k,v in emission_probabilities.items(): 
        for k2 in emission_probabilities[k]: 
            new_emission_probabilities[k2][k] = emission_probabilities[k][k2]
    return new_emission_probabilities



def read_acustic_path(acustic_filepath):
    '''
    Read data for decoding
    '''
       
    all_acoustic_costs = defaultdict(lambda: defaultdict(float))

    f = open(acustic_filepath, "r") 
    for row in f.readlines(): 
        splitted = row.split(" ") 
        source, dest, frame, weight = splitted 
        all_acoustic_costs[int(frame)][(int(source), int(dest))] = -float(weight)
    return all_acoustic_costs     
        

def read_kaldi_path(filepath): 
    
    '''
    Read data for decoding    
    '''
    
    
    transition_dict_in = defaultdict(list)  
    transition_dict_out = defaultdict(list)  
    all_states = set() 
    
    f = open(filepath, "r") 
    for row in f.readlines(): 
        splitted = row.split(" ") 
        
        source, dest, weight = splitted[:3] 
        
        transition_dict_in[int(dest)].append( (int(source), -float( weight) ) )
        transition_dict_out[int(source)].append( (int(dest), -float(weight)  ) )
        
        all_states.add(int(source))
        all_states.add(int(dest))
        
    return transition_dict_in, transition_dict_out,  all_states



def create_subset_graph(sources, G, K): 
    
    '''
    This function is used for experimenting with smaller values of K (smaller state spacse)
    '''
    
    new_transition_dict_out = defaultdict(list) 
    new_emission_likelihood = defaultdict(lambda: defaultdict(float))
    
    visited = set() 
    queue = [] 
    visited.update(set(sources)) 
    queue.extend(sources) 
    frame=dict() 
    for start in sources: 
        frame[start] = 0 
    
    while len(queue)>0 and len(visited) <= K: 
        v = queue.pop(0) 
        for neig in G.adj[v]: 
            
            if len(visited)<K: # make sure we do not add neighbours once we reach K states 
                
            
                frame[neig] = frame[v]+1
                new_transition_dict_out[v].append((neig, G.adj[v][neig]))
                new_emission_likelihood[frame[neig]][(v,neig)] = G.emission_probabilities[frame[neig]][(v,neig)] 
                
                if neig not in visited:
                    queue.append(neig) 
                    visited.add(neig)
                
    
    return new_transition_dict_out , new_emission_likelihood, list(visited)





if __name__ == '__main__': 
    
    
        # read data 
        acustic_filepath = "dataOptimalPath/acousticCost.txt"  
        all_acoustic_costs = read_acustic_path(acustic_filepath)
        filepath = "dataOptimalPath/graphFile.txt"  
        transition_dict_in, transition_dict_out, allstates = read_kaldi_path(filepath)
        sources = list(set(allstates).difference(set(transition_dict_in.keys()))) # start state zero does not have incoming 
        K = len(allstates)   
        allstates = list(allstates)  
        
        for T in [5, 7, 10, 15, 22, 32, 47, 69, 100]: 
            for i in range(5): 
                                
                G = forcedAlignment2Graph(transition_dict_out, all_acoustic_costs, allstates)
                
                print("Starting Vanilla Viterbi .. \n ") 
                P_Viterbi, T1 = viterbi_tp(G,sources,T)
                
                print("Starting Dijstra Viterbi .. \n ") 
                D, P = mint(G, sources,T)
                
                print("Starting DijstraBound Viterbi .. \n ") 
                D, P = mint_bound(G, sources,T)
                
                print("Starting Bi-directional Dijstra Viterbi .. \n ") 
                last_nodes = G.find_T_reachability_set(0,T)
                P , best_cost   = bidirectional_mint(G, sources, T, list(last_nodes))
                 
                print("Starting Bi-directional Dijstra Bound .. \n ") 
                P , best_cost  = bidirectional_mint_bound(G, sources, T, list(last_nodes))
              