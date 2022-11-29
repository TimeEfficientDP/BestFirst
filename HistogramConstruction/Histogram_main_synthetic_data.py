import random
from BestFirstHistogram import V_optimal_histogram, tech, compute_bounds, tech_bound, bidirectional_tech,compute_bounds_reversed, bidirectional_tech_bound

        
if __name__ == '__main__': 
    
  
      
       '''Perform experiments comparing different algorithms for histogram construction
       in synthetic data'''
      	
      
       # number of observable symbols 
       n_observables = 50
       # sequence length 
       for l in [1001]:
           # number of buckets 
           for B in [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]:
               # repeat a fixed number of times for robustness 
                for i in range(5): 
                     
                     # generate data 
                     random.seed(i)
                     sequence = [random.randint(0,n_observables-1) for _ in range(l)] 
                     
                     # precompute bounds
                     LB , UB = compute_bounds(sequence, B) 
                     LB_reversed , UB_reversed = compute_bounds_reversed(sequence, B) 
                     
                     print("Starting Vanilla V-Opt Histogram .. \n ") 
                     final_error = V_optimal_histogram(sequence, B) 
                     
                     print("Starting Tech Histogram .. \n ") 
                     final_error_dijkstra = tech(sequence, B)
                     
                     print("Starting Tech Bound Histogram .. \n " ) 
                     final_error_dijkstra_bound = tech_bound(sequence, B, LB)
                     
                     print("Starting Bidirectional Tech Histogram .. \n")
                     final_error_bidirectional_dijkstra = bidirectional_tech(sequence, B) 
                     
                     print("Starting Bidirectional-Bound Tech Histogram  .. \n")
                     final_error_bidirectional_dijkstra_bound = bidirectional_tech_bound(sequence, B, UB, UB_reversed) 
                  
