from BestFirstHistogram import V_optimal_histogram, tech, compute_bounds, compute_bounds_reversed, tech_bound, bidirectional_tech, bidirectional_tech_bound

def read_data(target_l):
    f = open("dataHistogram/dowJones.txt")
    sequence = []
    l = 0
    for row in f: 
        sequence.append(float(row.split("\t")[-1][:-1]))
        l+=1 
        if l >= target_l: 
            break 
        
    return sequence

        
if __name__ == '__main__': 
    
      
       '''Perform experiments comparing different algorithms for histogram construction
       using a subset of DowJones data'''
      	
      
       # number of observable symbols 
       n_observables = 50
       
       # sequence length 
       for l in [1001]:
           
           sequence = read_data(l)
           
           # number of buckets 
           for B in [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]:
               # repeat a fixed number of times for robustness 
                for i in range(5): 
                     
                    # precompute bounds
                     LB , UB = compute_bounds(sequence, B) 
                     LB_reversed , UB_reversed = compute_bounds_reversed(sequence, B) 
                     
                     print("Starting Vanilla V-Opt Histogram .. \n ") 
                     final_error = V_optimal_histogram(sequence, B) 
                     
                     print("Starting Dijkstra Histogram .. \n ") 
                     final_error_dijkstra = tech(sequence, B)
                     
                     print("Starting Bound Dijkstra Histogram .. \n " ) 
                     final_error_dijkstra_bound = tech_bound(sequence, B, LB)
                     
                     print("Starting Bidirectional Dijkstra Histogram .. \n")
                     final_error_bidirectional_dijkstra = bidirectional_tech(sequence, B) 
                     
                     print("Starting Bidirectional-Bound Dijkstra Histogram  .. \n")
                     final_error_bidirectional_dijkstra_bound = bidirectional_tech_bound(sequence, B, UB, UB_reversed) 
                  
