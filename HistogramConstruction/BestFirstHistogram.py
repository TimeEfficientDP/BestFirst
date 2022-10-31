from pqdict import PQDict
import numpy as np 
from collections import defaultdict 
import copy 
import logging
<<<<<<< HEAD
import heapq
from math import ceil , floor 
from graph import Graph




def V_optimal_histogram(sequence, B: int):
    
    ''' 
    V-optimal histogram DP algorithm 
    Parameters: 
        sequence: data (list) 
        B: number of buckets (int)
        
    Returns: 
        terr: optimal errors (array)
    
    ''' 
    
    
    n = len(sequence) 
    sums = np.zeros(n)
    squared_sums = np.zeros(n) 
    terr = np.full((n, B), np.inf)
    sums[0] = sequence[0]
    squared_sums[0]= sequence[0]**2 
    
    for i in range(1,n): 
        sums[i] = sums[i-1] + sequence[i] 
        squared_sums[i] = squared_sums[i-1] + sequence[i]**2
        
    start = 0 
    for j in range(n): 
        terr[j,0] = (squared_sums[j] -   squared_sums[start]) - (sums[j]  - sums[start] )**2 / (j - start + 1) # start enquining the zero split 
    
        
        for k in range(1, B): 
            for i in range(k-1, j): 
                squared_error = (squared_sums[j] - squared_sums[i]) - (sums[j] - sums[i])**2  /(j - i + 1)
                terr[j,k] = min( terr[j,k] ,  terr[i, k-1]  + squared_error  )
                
                
<<<<<<< HEAD
    return terr 



def tech(sequence, B:int):
    ''' 
    tech algorithm 
    Parameters: 
        sequence: data (list) 
        B: number of buckets (int)
        
    Returns: 
        D: optimal errors (dict)
    
    ''' 
   
=======
    return terr #terr[n-1, B-1] 



def V_optimal_histogram_dijkstra(sequence, B):
>>>>>>> 153cb145f02d8e319d061cc3839923d57d7f34ce
    
    D = defaultdict(dict) 
    D[(0, 0)] = 0 
    Q = PQDict(D)    
    visited = set() 
    n = len(sequence) 
    sums = np.zeros(n)
    squared_sums = np.zeros(n) 
    sums[0] = sequence[0]
<<<<<<< HEAD
    squared_sums[0]= sequence[0]**2     
=======
    squared_sums[0]= sequence[0]**2 
    #success = False
    
>>>>>>> 153cb145f02d8e319d061cc3839923d57d7f34ce
    
    for i in range(1,n): 
        sums[i] = sums[i-1] + sequence[i] 
        squared_sums[i] = squared_sums[i-1] + sequence[i]**2
        
    
    start = 0 
    for j in range(1,n-(B-1)+1):        
        Q[(j,0)] = (squared_sums[j] -   squared_sums[start]) - (sums[j]  - sums[start] )**2 / (j - start + 1)
<<<<<<< HEAD
    
    while Q: 
=======
      #(squared_sums[j] -   squared_sums[start]) - (sums[j]  - sums[start] )**2 / (j - start + 1) # start enquining the zero split 
    
    while Q: 
        #
>>>>>>> 153cb145f02d8e319d061cc3839923d57d7f34ce
        (key, d) = Q.popitem() 
        D[key]= d
        visited.add(key)  
                
        if key[0] == n-1 and key[1] == B-1: 
            break                
        
        if key[1] < B-1:
<<<<<<< HEAD
            for neig in range(key[0]+1, n-(B-1-key[1])+1): 
=======
            
            for neig in range(key[0]+1, n-(B-1-key[1])+1): 
            #for neig in range(key[0]+1, n): 
>>>>>>> 153cb145f02d8e319d061cc3839923d57d7f34ce
                if (neig,key[1]+1) not in visited:
                    new_d = d + (squared_sums[neig] - squared_sums[key[0]]) - (sums[neig] - sums[key[0]])**2  / (neig - key[0]  + 1)
                    if new_d < Q.get((neig,key[1]+1), float("inf")):
                        Q[(neig, key[1]+1)] = new_d 
        
<<<<<<< HEAD
    return D



def tech_bound(sequence, B, LB):
    ''' 
    tech bound algorithm 
    Parameters: 
        sequence: data (list) 
        B: number of buckets (int)
        
    Returns: 
        D: optimal errors (dict)
    ''' 
    
    D = defaultdict(dict) 
=======
    return D #D[key]




def V_optimal_histogram_dijkstra_2(sequence, B):
    
    D = defaultdict(dict) 
    D[(0, 0)] = 0 
    Q = PQDict(D)    
    visited = set() 
    n = len(sequence) 
    sums = np.zeros(n)
    squared_sums = np.zeros(n) 
    sums[0] = sequence[0]
    squared_sums[0]= sequence[0]**2 
    #success = False
    
    
    for i in range(1,n): 
        sums[i] = sums[i-1] + sequence[i] 
        squared_sums[i] = squared_sums[i-1] + sequence[i]**2
        
    
    start = 0 
    for j in range(1,n-(B-1)+1): 
        Q[(j,0)] = (squared_sums[j] -   squared_sums[start]) - (sums[j]  - sums[start] )**2 / (j - start + 1) # start enquining the zero split 
    
    while Q: 
        
        (key, d) = Q.popitem() 
        D[key]= d
        visited.add(key)  
                
        if key[0] == n-1 and key[1] == B-1: 
            #success=True 
            break                
        
        if key[1] < B-2:
            
            for neig in range(key[0]+1, n-(B-1-key[1])+1): 
            #for neig in range(key[0]+1, n): 
                if (neig,key[1]+1) not in visited:
                                        
                    #print("neig " + str(neig))
                    squared_error = (squared_sums[neig] - squared_sums[key[0]]) - (sums[neig] - sums[key[0]])**2  / (neig - key[0]  + 1)
                    new_d = d + squared_error 
                    if new_d < Q.get((neig,key[1]+1), float("inf")):
                        Q[(neig, key[1]+1)] = new_d 

        
        elif key[1] == B-2: 
            squared_error = (squared_sums[n-1] - squared_sums[key[0]]) - (sums[n-1] - sums[key[0]])**2  / (n - key[0]  )
            new_d = d + squared_error 
            if new_d < Q.get((n-1,key[1]+1), float("inf")):
                Q[(n-1, key[1]+1)] = new_d 
        
    #if not success:
    #    logging.warning('Algorithm terminated before last frame was reached.')        
    return D[key] #, len(visited) / (n * B)







def V_optimal_histogram_dijkstra_bound(sequence, B, LB):
    
    D = defaultdict(dict) 
    #D_actual_costs = defaultdict(dict) 
>>>>>>> 153cb145f02d8e319d061cc3839923d57d7f34ce
    D[(0, 0)] = LB[1,B]# (state, frame) - cost 
    Q = PQDict(D)    
    visited = set() 
    n = len(sequence) 
    sums = np.zeros(n)
    squared_sums = np.zeros(n) 
    sums[0] = sequence[0]
    squared_sums[0]= sequence[0]**2 
    
    for i in range(1,n): 
        sums[i] = sums[i-1] + sequence[i] 
        squared_sums[i] = squared_sums[i-1] + sequence[i]**2
        
        
    start = 0 
    for j in range(1,n-(B-1)+1): 
        Q[(j,0)] = (squared_sums[j] -   squared_sums[start]) - (sums[j]  - sums[start] )**2 / (j - start + 1) + LB[j+1,B] # start enquining the zero split 
    

        
    while Q: 
        (key, d) = Q.popitem() 
        D[key]= d
        visited.add(key)  
        if key[0] == n-1 and key[1] == B-1: 
            break                

        if key[1] < B-1:
            
            for neig in range(key[0]+1, n-(B-1-key[1])+1): 
                
                if (neig,key[1]+1) not in visited and neig <= n-1:
                    
                    squared_error = (squared_sums[neig] - squared_sums[key[0]]) - (sums[neig] - sums[key[0]])**2  / (neig - key[0]  + 1)
                    new_d = d + squared_error - LB[ key[0]+1, B - key[1]] + LB[ neig+1 , B - key[1] - 1]  # must be zero when neig+1 is n-1
                        
                    if new_d < Q.get((neig,key[1]+1), float("inf")):
                        Q[(neig, key[1]+1)] = new_d 
               
<<<<<<< HEAD
               
=======
                   # if D_actual_costs[key] + squared_error  < D_actual_costs.get((neig,key[1]+1), float("inf")): 
                   #     D_actual_costs[(neig,key[1]+1)] = D_actual_costs[key] + squared_error

    #if not success:
    #    logging.warning('Algorithm terminated before last frame was reached.')
        
>>>>>>> 153cb145f02d8e319d061cc3839923d57d7f34ce
    return D[n-1, B-1]



<<<<<<< HEAD


=======
def compute_bounds(sequence, B): 
    
    n = len(sequence)
    s = ceil(n / B) 
    LB = np.zeros((n+1,B+1)) # in this case B is exactly the number of buckets not split 
   # UB = np.matrix(np.ones((n+1,B+1)) * np.inf)
    sums = np.zeros(n)
    squared_sums = np.zeros(n) 
    sums[0] = sequence[0]
    squared_sums[0]= sequence[0]**2 
    
    for i in range(1,n): 
        sums[i] = sums[i-1] + sequence[i] 
        squared_sums[i] = squared_sums[i-1] + sequence[i]**2
    
    for i in range(n-1): 
        #LB[i,0] = squared_sums[i] - sums[i]**2/(i + 1)     
        
       # print("i " + str(i))
        
        left_elements = n - i #- 1
        
       # print("left elements " + str(left_elements))
        
        for k in range(1, B+1): 
            
            s =  floor(left_elements  / k)
            
            #print("k " + str(k) + " s " + str(s))
            
            if s > 0: 
                # compute the SSE for each bucket and sum them 
                all_sse = [] 
                previous_boundary = i 
                j = i
                while j < n: 
                    j+=s 
                    if j <n:
                        
                       # print("sum of squares between " + str(j-1) + " and " + str(previous_boundary))
                        squared_error = (squared_sums[j-1] - squared_sums[previous_boundary]) - (sums[j-1] - sums[previous_boundary])**2  / (j - 1 - previous_boundary + 1)
                        all_sse.append(squared_error) 
                        previous_boundary = copy.deepcopy(j) 
                    
                # adjust for last bin 
                if previous_boundary != n:
                    #print("sum of squares between " + str(n-1) + " and " + str(previous_boundary))
                    squared_error = (squared_sums[n-1] - squared_sums[previous_boundary]) - (sums[n-1] - sums[previous_boundary])**2  / (n - 1 - previous_boundary + 1)
                    all_sse.append(squared_error) 
                    
                LB[i, k] = min(all_sse) 
               # UB[i,k] = sum(all_sse)
            
    return LB 




def compute_bounds_upper(sequence, B): 
    
    n = len(sequence)
    s = ceil(n / B) 
    UB = np.matrix(np.ones((n+1,B+1)) * np.inf)
    sums = np.zeros(n)
    squared_sums = np.zeros(n) 
    sums[0] = sequence[0]
    squared_sums[0]= sequence[0]**2 
    
    for i in range(1,n): 
        sums[i] = sums[i-1] + sequence[i] 
        squared_sums[i] = squared_sums[i-1] + sequence[i]**2
    
    for i in range(n-1): 
        #LB[i,0] = squared_sums[i] - sums[i]**2/(i + 1)     UB
        
       # print("i " + str(i))
        
        left_elements = n - i #- 1
        
       # print("left elements " + str(left_elements))
        
        for k in range(1, B+1): 
            
            s =  ceil(left_elements  / k)
            
            #print("k " + str(k) + " s " + str(s))
            
            if s > 0: 
                # compute the SSE for each bucket and sum them 
                all_sse = [] 
                previous_boundary = i 
                j = i
                while j < n: 
                    j+=s 
                    if j <n:
                        
                       # print("sum of squares between " + str(j-1) + " and " + str(previous_boundary))
                        squared_error = (squared_sums[j-1] - squared_sums[previous_boundary]) - (sums[j-1] - sums[previous_boundary])**2  / (j - 1 - previous_boundary + 1)
                        all_sse.append(squared_error) 
                        previous_boundary = copy.deepcopy(j) 
                    
                # adjust for last bin 
                if previous_boundary != n:
                    #print("sum of squares between " + str(n-1) + " and " + str(previous_boundary))
                    squared_error = (squared_sums[n-1] - squared_sums[previous_boundary]) - (sums[n-1] - sums[previous_boundary])**2  / (n - 1 - previous_boundary + 1)
                    all_sse.append(squared_error) 
                    
                    
                UB[i,k] = sum(all_sse)
            
    return UB 



def compute_bounds_reversed(sequence, B): 
    
    n = len(sequence)
    s = ceil(n / B) 
    LB_reversed = np.zeros((n+1,B+1)) # in this case B is exactly the number of buckets not split 
    sums = np.zeros(n)
    squared_sums = np.zeros(n) 
    sums[0] = sequence[0]
    squared_sums[0]= sequence[0]**2 
    
    for i in range(1,n): 
        sums[i] = sums[i-1] + sequence[i] 
        squared_sums[i] = squared_sums[i-1] + sequence[i]**2
    
    
    for i in range(n-1): 
        
        
        #LB[i,0] = squared_sums[i] - sums[i]**2/(i + 1)        
        elements_to_split = i + 1
                
        
        for k in range(1, B+1): 
            
            
            s =  ceil(elements_to_split / k)
            # compute the SSE for each bucket and sum them 
            all_sse = [] 
            previous_boundary = i + 1
          
            j =  i + 1
            #print("i " + str(i)) 
            while j >= 0:  
                j-=s 
                #print("s " + str(s) + " " + str("j " ) + str(j))
                if j >=0:
                    squared_error = (squared_sums[previous_boundary] - squared_sums[j]) - (sums[previous_boundary] - sums[j])**2  / (previous_boundary - j + 1)
                    all_sse.append(squared_error) 
                    previous_boundary = copy.deepcopy(j) 
                    
                
            # adjust for last bin 
            if previous_boundary > 0:
                squared_error = (squared_sums[previous_boundary] - squared_sums[0]) - (sums[previous_boundary] - sums[0])**2  / (previous_boundary)
                all_sse.append(squared_error) 
                    
                
            LB_reversed[i, k] = min(all_sse) 
            
   
    return LB_reversed 
>>>>>>> 153cb145f02d8e319d061cc3839923d57d7f34ce









<<<<<<< HEAD

def bidirectional_tech(sequence, B):
   ''' 
   bidirectional tech algorithm 
   Parameters: 
       sequence: data (list) 
       B: number of buckets (int)
        
   Returns: 
       mu: optimal error 
   ''' 

   D_f = defaultdict(float) 
   D_b = defaultdict(float) 
   squared_error = dict() 
   n = len(sequence) 
   D_f[(0, 0)] = 0  
   D_b[(n-1, 0)] = 0 
   Q_f = PQDict(D_f)    
   Q_b = PQDict(D_b) 
   visited_b = set() 
   visited_f = set() 
   sums = np.zeros(n)
   squared_sums = np.zeros(n) 
   sums[0] = sequence[0]
   squared_sums[0]= sequence[0]**2 
      
   start = 0 
   for j in range(1, n-(B-1)+1): 
       sse = (squared_sums[j] - squared_sums[start]) - (sums[j]  - sums[start] )**2 / (j - start + 1)
       Q_f[(j,0)] = sse # start enquining the zero split 
       squared_error[(start,j)] = sse
       
   end = n-1
   for j in reversed(range(B-1, n-1)): 
      sse = (squared_sums[end] - squared_sums[j]) - ( sums[end] - sums[j]  )**2  / (end - j + 1)
      Q_b[(j,0)] = sse      
      squared_error[(j,end)] = sse
      
   mu = float("inf") 
   while len(Q_f)>0 and len(Q_b)>0:
       
       (k_f, d_f) = Q_f.popitem() 
       (k_b, d_b) = Q_b.popitem()                    # pop node w min dist d on frontier in constant time 
       D_f[k_f]=  d_f 
       D_b[k_b] = d_b      
       # update explored 
       visited_f.add(k_f)
       visited_b.add(k_b)       
       
       if k_f[1] < B-1:
           for neig in range(k_f[0]+1,  n-(B-1-k_f[1])+1): 
               
               if (neig, k_f[1]+1) not in visited_f:
                   if (k_f[0], neig) not in squared_error:
                       squared_error[(k_f[0], neig)] = (squared_sums[neig] - squared_sums[k_f[0]]) - (sums[neig] - sums[k_f[0]])**2  / (neig - k_f[0] + 1)
                   new_d = d_f + squared_error[(k_f[0], neig)] 
                   if new_d < Q_f.get((neig,k_f[1]+1), float("inf")):
                       Q_f[(neig, k_f[1]+1)] = new_d 
   
               if (neig , B - k_f[1]  - 2) in visited_b and D_f[k_f] + squared_error[(k_f[0], neig)] + D_b[( neig , B - k_f[1]  - 2 )] < mu:
                   mu = D_f[k_f] + squared_error[(k_f[0], neig)] + D_b[( neig , B - k_f[1]  - 2 )]
   
       if k_b[1] < B-1:
           for neig_back in reversed(range(B-1-k_b[1], k_b[0])): 
               
               if ( neig_back, k_b[1] + 1) not in visited_b:
                   if (neig_back, k_b[0]) not in squared_error:
                       squared_error[(neig_back, k_b[0])] = (squared_sums[k_b[0]] - squared_sums[neig_back]) - (sums[k_b[0]] - sums[neig_back])**2  / (k_b[0] - neig_back +1)
                   new_d = d_b + squared_error[(neig_back, k_b[0])]  
                   if new_d < Q_b.get((neig_back , k_b[1]+1), float("inf")):
                       Q_b[(neig_back, k_b[1]+1)] = new_d 
                   
               if (neig_back , B - k_b[1] - 2) in visited_f and D_b[k_b] + squared_error[(neig_back, k_b[0])] + D_f[( neig_back ,  B - k_b[1] - 2)] < mu :                                        
                   mu = D_b[k_b] + squared_error[(neig_back, k_b[0])]  + D_f[( neig_back ,  B - k_b[1] - 2 )]

       # check stopping condition 
       if D_f[k_f] + D_b[k_b] >= mu :
           break 
   
   return mu




def b_directional_tech_bound(sequence, B, LB, LB_reversed):
    ''' 
    bidirectional tech bound algorithm 
    Parameters: 
        sequence: data (list) 
        B: number of buckets (int)
         
    Returns: 
        mu: optimal error 
    ''' 

=======
def V_optimal_histogram_dijkstra_bound_old(sequence, B, LB):
    
    D = defaultdict(dict) 
    #D_actual_costs = defaultdict(dict) 
    #D[(0, 0)] =  LB[0,B] # (state, frame) - cost 
    #D_actual_costs[(0, 0)] = 0 # (state, frame) - cost 
    Q = PQDict(D)    
    visited = set() 
    n = len(sequence) 
    sums = np.zeros(n)
    squared_sums = np.zeros(n) 
    sums[0] = sequence[0]
    squared_sums[0]= sequence[0]**2 
    success = False
    
    for i in range(1,n): 
        sums[i] = sums[i-1] + sequence[i] 
        squared_sums[i] = squared_sums[i-1] + sequence[i]**2
        
    
    start = 0 
    for j in range(n): 
        Q[(j,0)] = (squared_sums[j] -   squared_sums[start]) - (sums[j]  - sums[start] )**2 / (j - start + 1) + LB[j+1,B] # start enquining the zero split 

        
    while Q: 
        (key, d) = Q.popitem() 
        D[key]= d
        visited.add(key)  
        if key[0] == n-1 and key[1] == B-1: 
            success=True 
            break                

        if key[1] < B-1:
            for neig in range(key[0]+1, n): 
                
                if (neig,key[1]+1) not in visited:
                    squared_error = (squared_sums[neig] - squared_sums[key[0]]) - (sums[neig] - sums[key[0]])**2  / ( neig - key[0] + 1 )
                    new_d = d + squared_error - LB[ key[0]+1, B - key[1]] + LB[ neig+1 , B - key[1] - 1]  # must be zero when neig+1 is n-1
                    if new_d < Q.get((neig,key[1]+1), float("inf")):
                        Q[(neig, key[1]+1)] = new_d 
               
                   # if D_actual_costs[key] + squared_error  < D_actual_costs.get((neig,key[1]+1), float("inf")): 
                   #     D_actual_costs[(neig,key[1]+1)] = D_actual_costs[key] + squared_error

    if not success:
        logging.warning('Algorithm terminated before last frame was reached.')
        
    return D[key]



def V_optimal_histogram_dijkstra_bidirectional(sequence, B):
    
    D_f = defaultdict(float) 
    D_b = defaultdict(float) 
    squared_error = dict() 
    n = len(sequence) 
    D_f[(0, 0)] = 0  
    D_b[(n-1, 0)] = 0 
    Q_f = PQDict(D_f)    
    Q_b = PQDict(D_b) 
    visited_b = set() 
    visited_f = set() 
    sums = np.zeros(n)
    squared_sums = np.zeros(n) 
    #sums_reversed = np.zeros(n)
    #squared_sums_reversed = np.zeros(n) 
    sums[0] = sequence[0]
    squared_sums[0]= sequence[0]**2 
    #sums_reversed[-1] = sequence[-1] 
    #squared_sums_reversed[-1] = sequence[-1]**2 
    for i in range(1,n): 
        sums[i] = sums[i-1] + sequence[i] 
        squared_sums[i] = squared_sums[i-1] + sequence[i]**2
    #for i in reversed(range(n-1)): 
    #    print("i " + str(i))
    #    sums_reversed[i] = sums_reversed[i+1] + sequence[i] 
    #    squared_sums_reversed[i] = squared_sums_reversed[i+1] + sequence[i]**2     
    # sum computed, now starting enquing fromt the start and from the end     
    start = 0 
    for j in range(1, n-(B-1)+1): 
        sse = (squared_sums[j] - squared_sums[start]) - (sums[j]  - sums[start] )**2 / (j - start + 1)
        Q_f[(j,0)] = sse # start enquining the zero split 
        squared_error[(start,j)] = sse
        
    end = n-1
    for j in reversed(range(B-1, n-1)): 
       sse = (squared_sums[end] - squared_sums[j]) - ( sums[end] - sums[j]  )**2  / (end - j + 1)
       Q_b[(j,0)] = sse # now its correct we dont use reversed sums also in the backward case         
       squared_error[(j,end)] = sse
       
    mu = float("inf") 
    while len(Q_f)>0 and len(Q_b)>0:
        
        (k_f, d_f) = Q_f.popitem() 
        (k_b, d_b) = Q_b.popitem()                    # pop node w min dist d on frontier in constant time 
        D_f[k_f]=  d_f 
        D_b[k_b] = d_b      
        # update explored 
        visited_f.add(k_f)
        visited_b.add(k_b)       
        
        if k_f[1] < B-1:
            for neig in range(k_f[0]+1,  n-(B-1-k_f[1])+1): 
                
                if (neig, k_f[1]+1) not in visited_f:
                    if (k_f[0], neig) not in squared_error:
                        squared_error[(k_f[0], neig)] = (squared_sums[neig] - squared_sums[k_f[0]]) - (sums[neig] - sums[k_f[0]])**2  / (neig - k_f[0] + 1)
                    new_d = d_f + squared_error[(k_f[0], neig)] 
                    if new_d < Q_f.get((neig,k_f[1]+1), float("inf")):
                        Q_f[(neig, k_f[1]+1)] = new_d 
    
                #if ( k_f[0] , B - k_f[1]  - 1 ) in visited_b and D_f[k_f] + D_b[( k_f[0] , B - k_f[1]  - 1 )] < mu :
                if (neig , B - k_f[1]  - 2) in visited_b and D_f[k_f] + squared_error[(k_f[0], neig)] + D_b[( neig , B - k_f[1]  - 2 )] < mu:
                    mu = D_f[k_f] + squared_error[(k_f[0], neig)] + D_b[( neig , B - k_f[1]  - 2 )]
    
        if k_b[1] < B-1:
            for neig_back in reversed(range(B-1-k_b[1], k_b[0])): 
                
                if ( neig_back, k_b[1] + 1) not in visited_b:
                    if (neig_back, k_b[0]) not in squared_error:
                        squared_error[(neig_back, k_b[0])] = (squared_sums[k_b[0]] - squared_sums[neig_back]) - (sums[k_b[0]] - sums[neig_back])**2  / (k_b[0] - neig_back +1)
                    new_d = d_b + squared_error[(neig_back, k_b[0])]  
                    if new_d < Q_b.get((neig_back , k_b[1]+1), float("inf")):
                        Q_b[(neig_back, k_b[1]+1)] = new_d 
                    
                #squared_error = (squared_sums[k_b[0]] - squared_sums[neig_back]) - (sums[k_b[0]] - sums[neig_back])**2  / (k_b[0] - neig_back + 1)
                if (neig_back , B - k_b[1] - 2) in visited_f and D_b[k_b] + squared_error[(neig_back, k_b[0])] + D_f[( neig_back ,  B - k_b[1] - 2)] < mu :                                        
                    mu = D_b[k_b] + squared_error[(neig_back, k_b[0])]  + D_f[( neig_back ,  B - k_b[1] - 2 )]

        # check condition 
        if D_f[k_f] + D_b[k_b] >= mu :
            #print("Breaking!")
            break 
    
    return mu



def V_optimal_histogram_dijkstra_bidirectional_bound(sequence, B, LB, LB_reversed):
    
>>>>>>> 153cb145f02d8e319d061cc3839923d57d7f34ce
    
    D_f = defaultdict(float) 
    D_b = defaultdict(float) 
    squared_error = dict() 
    n = len(sequence) 
    D_f[(0, 0)] = LB[1,B]
    D_b[(n-1, 0)] = LB_reversed[n-2,B]
    Q_f = PQDict(D_f)    
    Q_b = PQDict(D_b) 
    visited_b = set() 
    visited_f = set() 
    sums = np.zeros(n)
    squared_sums = np.zeros(n) 
<<<<<<< HEAD
    sums[0] = sequence[0]
    squared_sums[0]= sequence[0]**2 
    
    for i in range(1,n): 
        sums[i] = sums[i-1] + sequence[i] 
        squared_sums[i] = squared_sums[i-1] + sequence[i]**2
    
=======
    #sums_reversed = np.zeros(n)
    #squared_sums_reversed = np.zeros(n) 
    sums[0] = sequence[0]
    squared_sums[0]= sequence[0]**2 
    #sums_reversed[-1] = sequence[-1] 
    #squared_sums_reversed[-1] = sequence[-1]**2 
    for i in range(1,n): 
        sums[i] = sums[i-1] + sequence[i] 
        squared_sums[i] = squared_sums[i-1] + sequence[i]**2
    #for i in reversed(range(n-1)): 
    #    print("i " + str(i))
    #    sums_reversed[i] = sums_reversed[i+1] + sequence[i] 
    #    squared_sums_reversed[i] = squared_sums_reversed[i+1] + sequence[i]**2     
    # sum computed, now starting enquing fromt the start and from the end     
>>>>>>> 153cb145f02d8e319d061cc3839923d57d7f34ce
    start = 0 
    for j in range(1, n-(B-1)+1): 
        sse = (squared_sums[j] - squared_sums[start]) - (sums[j]  - sums[start] )**2 / (j - start + 1) 
        Q_f[(j,0)] = sse + LB[j+1,B]  # start enquining the zero split 
        squared_error[(start,j)] = sse
        
    end = n-1
    for j in reversed(range(B-1, n-1)): 
       sse = (squared_sums[end] - squared_sums[j]) - ( sums[end] - sums[j]  )**2  / (end - j + 1)
<<<<<<< HEAD
       Q_b[(j,0)] = sse  + LB_reversed[j-1,B]        
=======
       Q_b[(j,0)] = sse  + LB_reversed[j-1,B]  # now its correct we dont use reversed sums also in the backward case         
>>>>>>> 153cb145f02d8e319d061cc3839923d57d7f34ce
       squared_error[(j,end)] = sse
       
    mu = float("inf") 
    while len(Q_f)>0 and len(Q_b)>0:
        
        (k_f, d_f) = Q_f.popitem() 
        (k_b, d_b) = Q_b.popitem()                    # pop node w min dist d on frontier in constant time 
        D_f[k_f]=  d_f 
        D_b[k_b] = d_b      
        # update explored 
        visited_f.add(k_f)
        visited_b.add(k_b)       
        
        if k_f[1] < B-1:
            for neig in range(k_f[0]+1,  n-(B-1-k_f[1])+1): 
                
                if (neig, k_f[1]+1) not in visited_f:
                    if (k_f[0], neig) not in squared_error:
                        squared_error[(k_f[0], neig)] = (squared_sums[neig] - squared_sums[k_f[0]]) - (sums[neig] - sums[k_f[0]])**2  / (neig - k_f[0] + 1)
                    new_d = d_f + squared_error[(k_f[0], neig)]  - LB[ k_f[0]+1, B - k_f[1]] + LB[ neig+1 , B - k_f[1] - 1] 
                    if new_d < Q_f.get((neig,k_f[1]+1), float("inf")):
                        Q_f[(neig, k_f[1]+1)] = new_d 
    
<<<<<<< HEAD
=======
                #if ( k_f[0] , B - k_f[1]  - 1 ) in visited_b and D_f[k_f] + D_b[( k_f[0] , B - k_f[1]  - 1 )] < mu :
>>>>>>> 153cb145f02d8e319d061cc3839923d57d7f34ce
                if (neig , B - k_f[1]  - 2) in visited_b and D_f[k_f] - LB[ k_f[0]+1, B - k_f[1]]  + squared_error[(k_f[0], neig)] + D_b[( neig , B - k_f[1]  - 2 )] - LB_reversed[ neig-1 , k_f[1]  + 2] < mu:
                    mu = D_f[k_f] - LB[ k_f[0]+1, B - k_f[1]]  + squared_error[(k_f[0], neig)] + D_b[( neig , B - k_f[1]  - 2 )] - LB_reversed[ neig-1 , k_f[1]  + 2]
    
        if k_b[1] < B-1:
            for neig_back in reversed(range(B-1-k_b[1], k_b[0])): 
                
                if ( neig_back, k_b[1] + 1) not in visited_b:
                    if (neig_back, k_b[0]) not in squared_error:
                        squared_error[(neig_back, k_b[0])] = (squared_sums[k_b[0]] - squared_sums[neig_back]) - (sums[k_b[0]] - sums[neig_back])**2  / (k_b[0] - neig_back +1)
                    new_d = d_b + squared_error[(neig_back, k_b[0])]  - LB_reversed[ k_b[0]-1, B - k_b[1]] + LB_reversed[ neig_back-1 , B - k_b[1] - 1]  
                    if new_d < Q_b.get((neig_back , k_b[1]+1), float("inf")):
                        Q_b[(neig_back, k_b[1]+1)] = new_d 
                    
<<<<<<< HEAD
                if (neig_back , B - k_b[1] - 2) in visited_f and D_b[k_b]  - LB_reversed[ k_b[0]-1, B - k_b[1]]  + squared_error[(neig_back, k_b[0])] + D_f[( neig_back ,  B - k_b[1] - 2)] - LB[neig_back+1 , k_b[1] + 2 ]  < mu :                                        
                    mu = D_b[k_b]  - LB_reversed[ k_b[0]-1, B - k_b[1]]  + squared_error[(neig_back, k_b[0])] + D_f[( neig_back ,  B - k_b[1] - 2)] - LB[neig_back+1 , k_b[1] + 2 ]

        # check stopping condition 
        if D_f[k_f] + D_b[k_b] >= mu :
            break 
    
    return mu
=======
                #squared_error = (squared_sums[k_b[0]] - squared_sums[neig_back]) - (sums[k_b[0]] - sums[neig_back])**2  / (k_b[0] - neig_back + 1)
                if (neig_back , B - k_b[1] - 2) in visited_f and D_b[k_b]  - LB_reversed[ k_b[0]-1, B - k_b[1]]  + squared_error[(neig_back, k_b[0])] + D_f[( neig_back ,  B - k_b[1] - 2)] - LB[neig_back+1 , k_b[1] + 2 ]  < mu :                                        
                    mu = D_b[k_b]  - LB_reversed[ k_b[0]-1, B - k_b[1]]  + squared_error[(neig_back, k_b[0])] + D_f[( neig_back ,  B - k_b[1] - 2)] - LB[neig_back+1 , k_b[1] + 2 ]

        # check condition 
        if D_f[k_f] + D_b[k_b] >= mu :
            #print("Breaking!")
            break 
    
    return mu
























'''


def V_optimal_histogram_dijkstra_bidirectional_bound(sequence, B,  LB, LB_reversed):
        
    
    print("ippp")

    D_f = defaultdict(float) 
    D_b = defaultdict(float) 
    squared_error = dict() 
    n = len(sequence) 
    
    #D_f[(0, 0)] = 0  
    #D_b[(n-1, 0)] = 0 
    Q_f = PQDict(D_f)    
    Q_b = PQDict(D_b) 
    visited_b = set() 
    visited_f = set() 
    
    sums = np.zeros(n)
    squared_sums = np.zeros(n) 
    #sums_reversed = np.zeros(n)
    #squared_sums_reversed = np.zeros(n) 
    sums[0] = sequence[0]
    squared_sums[0]= sequence[0]**2 
    #sums_reversed[-1] = sequence[-1] 
    #squared_sums_reversed[-1] = sequence[-1]**2 
    
    for i in range(1,n): 
        sums[i] = sums[i-1] + sequence[i] 
        squared_sums[i] = squared_sums[i-1] + sequence[i]**2
    #for i in reversed(range(n-1)): 
    #    print("i " + str(i))
    #    sums_reversed[i] = sums_reversed[i+1] + sequence[i] 
    #    squared_sums_reversed[i] = squared_sums_reversed[i+1] + sequence[i]**2     
    # sum computed, now starting enquing fromt the start and from the end     
    start = 0 
    for j in range(1, n-(B-1)+1): 
        sse = (squared_sums[j] -   squared_sums[start]) - (sums[j]  - sums[start] )**2 / (j - start + 1) + LB[j+1,B]
        Q_f[(j,0)] = sse # start enquining the zero split 
        squared_error[(start,j)] = sse
        
        
    end = n-1
    for j in reversed(range(B-1, n-1)): 
       sse = (squared_sums[end] - squared_sums[j]) - ( sums[end] - sums[j]  )**2  / (end - j + 1)
       Q_b[(j,0)] = sse # now its correct we dont use reversed sums also in the backward case      + LB[j+1,B]   
       squared_error[(j,end)] = sse
       
       
    mu = float("inf") 
  
    while len(Q_f)>0 and len(Q_b)>0:
        
        
        (k_f, d_f) = Q_f.popitem() 
        (k_b, d_b) = Q_b.popitem()                    # pop node w min dist d on frontier in constant time 
        D_f[k_f]=  d_f 
        D_b[k_b] = d_b      
        # update explored 
        visited_f.add(k_f)
        visited_b.add(k_b)   
        
    
        
        if k_f[1] < B-1:
            for neig in range(k_f[0]+1,  n-(B-1-k_f[1])+1): 
                if (neig, k_f[1]+1) not in visited_f:
                    
                    if (k_f[0], neig) not in squared_error:
                        squared_error[(k_f[0], neig)] = (squared_sums[neig] - squared_sums[k_f[0]]) - (sums[neig] - sums[k_f[0]])**2  / (neig - k_f[0] + 1)
                    new_d = d_f + squared_error[(k_f[0], neig)] - LB[ k_f[0]+1, B - k_f[1]] + LB[ neig+1 , B - k_f[1] - 1] 
                    if new_d < Q_f.get((neig,k_f[1]+1), float("inf")):
                        Q_f[(neig, k_f[1]+1)] = new_d 
    
                #if ( k_f[0] , B - k_f[1]  - 1 ) in visited_b and D_f[k_f] + D_b[( k_f[0] , B - k_f[1]  - 1 )] < mu :
                if (neig , B - k_f[1]  - 2) in visited_b and D_f[k_f] - LB[ k_f[0]+1, B - k_f[1]]  + squared_error[(k_f[0], neig)] + D_b[( neig , B - k_f[1]  - 2 )]  - LB_reversed[ neig-1 ,k_f[1]  - 2]  < mu:
                    mu = D_f[k_f] - LB[ k_f[0]+1, B - k_f[1]]  + squared_error[(k_f[0], neig)] + D_b[( neig , B - k_f[1]  - 2 )]  - LB_reversed[ neig-1 , k_f[1]  - 2] 
    
        if k_b[1] < B-1:
            for neig_back in reversed(range(B-1-k_b[1], k_b[0])): 
                if ( neig_back, k_b[1] + 1) not in visited_b:
                    if (neig_back, k_b[0]) not in squared_error:
                        squared_error[(neig_back, k_b[0])] = (squared_sums[k_b[0]] - squared_sums[neig_back]) - (sums[k_b[0]] - sums[neig_back])**2  / (k_b[0] - neig_back +1)
                    new_d = d_b + squared_error[(neig_back, k_b[0])] - LB_reversed[ k_b[0]-1, B - k_b[1]] + LB_reversed[ neig_back-1 , B - k_b[1] - 1]  
                    if new_d < Q_b.get((neig_back , k_b[1]+1), float("inf")):
                        Q_b[(neig_back, k_b[1]+1)] = new_d 
                    
                #squared_error = (squared_sums[k_b[0]] - squared_sums[neig_back]) - (sums[k_b[0]] - sums[neig_back])**2  / (k_b[0] - neig_back + 1)
                if ( neig_back , B - k_b[1] - 2 ) in visited_f and D_b[k_b] - LB_reversed[ k_b[0]-1, B - k_b[1]]  + squared_error[(neig_back, k_b[0])] + D_f[( neig_back ,  B - k_b[1] - 2 )] - LB[neig_back+1 ,k_b[1] - 2 ]  < mu:                                        
                    mu = D_b[k_b] - LB_reversed[ k_b[0]-1, B - k_b[1]]  + squared_error[(neig_back, k_b[0])] + D_f[( neig_back ,  B - k_b[1] - 2 )] - LB[neig_back+1 , k_b[1] - 2 ]
                    
        # check condition 
        if D_f[k_f] + D_b[k_b] >= mu : #and k_f[1] >= max_t_forw and k_b[1] <= max_t_back:
            #print("Breaking!")
            break 
    

    return mu









def V_optimal_histogram_dijkstra_bidirectional_bound_no(sequence, B, LB, LB_reversed):
    

    D_f = defaultdict(float) 
    D_b = defaultdict(float) 
    
    n = len(sequence) 
    
    Q_f = PQDict(D_f)    
    Q_b = PQDict(D_b) 
    visited_b = set() 
    visited_f = set() 
    
    sums = np.zeros(n)
    squared_sums = np.zeros(n) 
    #sums_reversed = np.zeros(n)
    #squared_sums_reversed = np.zeros(n) 
    sums[0] = sequence[0]
    squared_sums[0]= sequence[0]**2 
    #sums_reversed[-1] = sequence[-1] 
    #squared_sums_reversed[-1] = sequence[-1]**2 
    success = False
    
    for i in range(1,n): 
        sums[i] = sums[i-1] + sequence[i] 
        squared_sums[i] = squared_sums[i-1] + sequence[i]**2
    #for i in reversed(range(n-1)): 
    #    print("i " + str(i))
    #    sums_reversed[i] = sums_reversed[i+1] + sequence[i] 
    #    squared_sums_reversed[i] = squared_sums_reversed[i+1] + sequence[i]**2     
    # sum computed, now starting enquing fromt the start and from the end     
    start = 0 
    for j in range(n): 
        Q_f[(j,0)] = (squared_sums[j] -   squared_sums[start]) - (sums[j]  - sums[start] )**2 / (j - start + 1) + LB[j+1,B]  # start enquining the zero split 
        
    end = n-1
    for j in reversed(range(n)): 
       Q_b[(j,0)] = squared_sums[end] - squared_sums[j] - ( sums[end] - sums[j]  )**2  / (end - j + 1)  +  LB_reversed[j-1,B]  # now its correct we dont use reversed sums also in the backward case         
  
    mu = float("inf") 
  
    while len(Q_f)>0 and len(Q_b)>0:
        
        (k_f, d_f) = Q_f.popitem() 
        (k_b, d_b) = Q_b.popitem()                    # pop node w min dist d on frontier in constant time 
        D_f[k_f]=  d_f 
        D_b[k_b] = d_b      
        # update explored 
        visited_f.add(k_f)
        visited_b.add(k_b)   
        
        if k_f[1] < B-1:
            for neig in range(k_f[0]+1, n): 
                
                if (neig, k_f[1]+1) not in visited_f:
          
                    squared_error = (squared_sums[neig] - squared_sums[k_f[0]]) - (sums[neig] - sums[k_f[0]])**2  / (neig - k_f[0])
                    new_d = d_f + squared_error - LB[ k_f[0]+1, B - k_f[1]] + LB[ neig+1 , B - k_f[1] - 1] 
                    if new_d < Q_f.get((neig,k_f[1]+1), float("inf")):
                        Q_f[(neig, k_f[1]+1)] = new_d 
    
                    

                if ( k_f[0]+1 , B - k_f[1]  - 1 ) in visited_b and D_f[k_f]  - LB[ k_f[0]+1, B - k_f[1]]  + D_b[( k_f[0]+1 , B - k_f[1]  - 1 )]  - LB_reversed[ k_f[0], k_f[1] + 1 ]  < mu :
                    mu = D_f[k_f]  - LB[ k_f[0]+1, B - k_f[1]]  + D_b[( k_f[0]+1 , B - k_f[1]  - 1 )]  - LB_reversed[ k_f[0], k_f[1] + 1 ]
         
    
        if k_b[1] < B-1:
            for neig_back in reversed(range(k_b[0])): 
                
                if ( neig_back, k_b[1] + 1) not in visited_b: 
                                        
                    squared_error = (squared_sums[k_b[0]] - squared_sums[neig_back]) - (sums[k_b[0]] - sums[neig_back])**2  / (k_b[0] - neig_back)
                    new_d = d_b + squared_error - LB_reversed[ k_b[0]-1, B - k_b[1]] + LB_reversed[ neig-1 , B - k_b[1] - 1] 
                    if new_d < Q_b.get((neig_back , k_b[1]+1), float("inf")):
                        Q_b[(neig_back, k_b[1]+1)] = new_d 
                    
                if ( k_b[0]-1 , B - k_b[1] - 1 ) in visited_f and D_b[k_b] - LB_reversed[ k_b[0]-1, B - k_b[1]]  + D_f[( k_b[0]-1 ,  B - k_b[1] - 1 )]  + LB[k_b[0],k_b[1] + 1]  < mu :                                        
                    mu = D_b[k_b] + D_f[(k_b[0]-1, B - k_b[1] - 1 )]
         
                    
        # check condition 
        if D_f[k_f] - LB[ k_f[0]+1, B - k_f[1]]  + D_b[k_b] -LB_reversed[k_b[0]-1, B-k_b[1]] >= mu : #and k_f[1] >= max_t_forw and k_b[1] <= max_t_back:
            print("Breaking!")
            break 
    
        

    return mu
'''


















def bidirectional_dijkstra_bound(G, start, y, last_nodes):
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
                    d = D_f[k_f] + G.upper_bound - G.adj[k_f[0]][w]  - G.emission_probabilities[ w ][ y[k_f[1]+1] ]      # dgs: dist of start -> v -> w
                    if d < Q_f.get((w,k_f[1]+1), float("inf")):
                        Q_f[(w, k_f[1]+1)] = d     
                  #      history = copy.deepcopy(P_f[k_f])
                  #      history.append(k_f)
                        P_f[(w, k_f[1]+1)] = P_f[k_f] + [k_f]  
                                        
                
                actual_cost_forward = D_f[k_f] + (T - 1 - k_f[1]) * G.upper_bound
                actual_cost_backward = D_b[( w , k_f[1]+1 )] + (T - 1 -  k_f[1] - 1) * G.upper_bound
                if ( w , k_f[1]+1 ) in visited_b and actual_cost_forward - G.adj[k_f[0]][w]  + actual_cost_backward - G.emission_probabilities[ w ][ y[k_f[1]+1] ] < mu and len(P_f[k_f]) + len(P_b[( w , k_f[1]+1 )]) == T-2:
                    mu = actual_cost_forward - G.adj[k_f[0]][w]  + actual_cost_backward  - G.emission_probabilities[ w ][ y[k_f[1]+1] ]
                    best_path = P_f[k_f] + [(k_f[0], k_f[1]), (w, k_f[1]+1)] + P_b[( w , k_f[1]+1 )][::-1]
                    
            
        if k_b[1] > 0: 
            for w in G.adj_inv[k_b[0]]:  # successors to v
                if ( w , k_b[1] - 1 ) not in visited_b: 
                    
                    d = D_b[k_b] + G.upper_bound - G.adj_inv[k_b[0]][w] - G.emission_probabilities[ k_b[0] ][ y[k_b[1]] ]      # here we add the emission probability of the source and not of the destination 
                    if d < Q_b.get((w,k_b[1]-1), float("inf")):
                        Q_b[(w, k_b[1]-1)] = d                             
                       # history = copy.deepcopy(P_b[k_b])
                       # history.append(k_b)
                        P_b[(w, k_b[1]-1)] = P_b[k_b] + [k_b]   
                                     
                actual_cost_forward = D_f[(w, k_b[1]-1)] + (T - 1 - k_b[1] + 1) * G.upper_bound
                actual_cost_backward = D_b[k_b] + (T - 1 - k_b[1]) * G.upper_bound
                if ( w, k_b[1]-1 ) in visited_f and actual_cost_backward - G.emission_probabilities[ k_b[0] ][ y[k_b[1]] ] - G.adj_inv[k_b[0]][w]  + actual_cost_forward < mu and len(P_f[(w, k_b[1]-1)]) + len(P_b[k_b]) == T - 2:                                        
                    mu = actual_cost_backward - G.adj_inv[k_b[0]][w]  - G.emission_probabilities[ k_b[0] ][ y[k_b[1]] ] + actual_cost_forward
                    best_path = P_f[(w, k_b[1]-1)] +  [(w, k_b[1]-1), (k_b[0], k_b[1])] +  P_b[k_b][::-1] #P_f[(w, t_forw)] + [ (w,k_b[1]-1) , (k_b[0], k_b[1]) ] + P_b[k_b][::-1]

                    
        # check condition 
        if D_f[k_f] + D_b[k_b] >= mu : #and k_f[1] >= max_t_forw and k_b[1] <= max_t_back:
            print("Breaking!")
            break 
                 
    r =  len(visited_b.union(visited_f)) / (len(G.nodes) * len(y))
    
    return  best_path , r



























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
            # print("new candidate ---- median " + str( (pred, k[0]) ))
            new_medians[k[0]] = (pred, k[0]) 

        elif k[1]>T_floor:
            new_medians[k[0]] = new_medians[pred]
                    
        #print("visiting " + str(k)) 
        #print("len Q " + str(len(Q)))
        if (k[1] == last_frame and final_state is None) or (k[0] == final_state and k[1] == last_frame): 
            #success=True 
            x_a, x_b =  new_medians[k[0]]
                        
            N_left = floor(n_frames/2)              
            # y_left = y[:N_left] 
            if N_left >1 : 
                 
                # use bfs to find ancestors in N_left-1 steps 
                #ancestors = BFS_ancestors(G, x_a,  N_left-1)
                # still using the entire state space 
                # continue recursion 
                dijkstra_dfs_sieve(G, start, y, space_budget, final_state = x_a, start_frame = 0, last_frame = N_left-1, n_frames = N_left)
            
            
            N_right = n_frames- N_left 
            #  y_right = y[-N_right:]
            # in order print 
            print("median pair " + str((x_a, x_b)))
            
            if N_right >1: 
                
                # use bfs to find descendants in N_right -1 steps 
                #descendants = BFS_descendants(G, x_b,  N_right-1)
                
                # continue recursion
                dijkstra_dfs_sieve(G, x_b, y, space_budget,  start_frame =  T_floor  , last_frame = T_floor +  N_right - 1 , n_frames = N_right)
            
            break
            
        # break                  # if the frame k[1] is T we arrived to the end 
        # now consider the edges from v with an unexplored head -
        # we may need to update the dist of unexplored successors 


        for w in G.adj[k[0]]:                          # successors to v
        
            if k[1]+1 > last_frame: 
                break # break for loop go to next while loop iteration 
        
            if ( w , k[1]+1 ) not in visited:        
                
                # then w is a frontier node
                d = D[k] - G.adj[k[0]][w]  - G.emission_probabilities[ w ][ y[k[1]+1] ]      # dgs: dist of start -> v -> w
                
                tmp = Q.get((w,k[1]+1), float("inf"))
                if isinstance(tmp, tuple): 
                    tmp = tmp[0] 
                
                if d < tmp:
                    
                    
                    if len(map_state2t[w]) > space_budget and k[1]+1 not in map_state2t[w]:
                        
                      #  print("running dfs!") 
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
    
    




def viterbi(G, start, y):
    """
    Return the MAP estimate of state trajectory of Hidden Markov Model.

    Parameters
    
    Returns
    -------

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
    Return the MAP estimate of state trajectory of Hidden Markov Model.

    Parameters
    
    Returns
    -------

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
                if ( w , k_b[1] - 1 ) not in visited_b: 
                    
                    d = D_b[k_b] - G.adj_inv[k_b[0]][w]  - G.emission_probabilities[ k_b[0] ][ y[k_b[1]] ]      # here we add the emission probability of the source and not of the destination 
                    if d < Q_b.get((w,k_b[1]-1), float("inf")):
                        Q_b[(w, k_b[1]-1)] = d                             
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







def bidirectional_dijkstra_bound(G, start, y, last_nodes):
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
                    d = D_f[k_f] + G.upper_bound - G.adj[k_f[0]][w]  - G.emission_probabilities[ w ][ y[k_f[1]+1] ]      # dgs: dist of start -> v -> w
                    if d < Q_f.get((w,k_f[1]+1), float("inf")):
                        Q_f[(w, k_f[1]+1)] = d     
                  #      history = copy.deepcopy(P_f[k_f])
                  #      history.append(k_f)
                        P_f[(w, k_f[1]+1)] = P_f[k_f] + [k_f]  
                                        
                
                actual_cost_forward = D_f[k_f] + (T - 1 - k_f[1]) * G.upper_bound
                actual_cost_backward = D_b[( w , k_f[1]+1 )] + (T - 1 -  k_f[1] - 1) * G.upper_bound
                if ( w , k_f[1]+1 ) in visited_b and actual_cost_forward - G.adj[k_f[0]][w]  + actual_cost_backward - G.emission_probabilities[ w ][ y[k_f[1]+1] ] < mu and len(P_f[k_f]) + len(P_b[( w , k_f[1]+1 )]) == T-2:
                    mu = actual_cost_forward - G.adj[k_f[0]][w]  + actual_cost_backward  - G.emission_probabilities[ w ][ y[k_f[1]+1] ]
                    best_path = P_f[k_f] + [(k_f[0], k_f[1]), (w, k_f[1]+1)] + P_b[( w , k_f[1]+1 )][::-1]
                    
            
        if k_b[1] > 0: 
            for w in G.adj_inv[k_b[0]]:  # successors to v
                if ( w , k_b[1] - 1 ) not in visited_b: 
                    
                    d = D_b[k_b] + G.upper_bound - G.adj_inv[k_b[0]][w] - G.emission_probabilities[ k_b[0] ][ y[k_b[1]] ]      # here we add the emission probability of the source and not of the destination 
                    if d < Q_b.get((w,k_b[1]-1), float("inf")):
                        Q_b[(w, k_b[1]-1)] = d                             
                       # history = copy.deepcopy(P_b[k_b])
                       # history.append(k_b)
                        P_b[(w, k_b[1]-1)] = P_b[k_b] + [k_b]   
                                     
                actual_cost_forward = D_f[(w, k_b[1]-1)] + (T - 1 - k_b[1] + 1) * G.upper_bound
                actual_cost_backward = D_b[k_b] + (T - 1 - k_b[1]) * G.upper_bound
                if ( w, k_b[1]-1 ) in visited_f and actual_cost_backward - G.emission_probabilities[ k_b[0] ][ y[k_b[1]] ] - G.adj_inv[k_b[0]][w]  + actual_cost_forward < mu and len(P_f[(w, k_b[1]-1)]) + len(P_b[k_b]) == T - 2:                                        
                    mu = actual_cost_backward - G.adj_inv[k_b[0]][w]  - G.emission_probabilities[ k_b[0] ][ y[k_b[1]] ] + actual_cost_forward
                    best_path = P_f[(w, k_b[1]-1)] +  [(w, k_b[1]-1), (k_b[0], k_b[1])] +  P_b[k_b][::-1] #P_f[(w, t_forw)] + [ (w,k_b[1]-1) , (k_b[0], k_b[1]) ] + P_b[k_b][::-1]

                    actual_cost_backward
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
