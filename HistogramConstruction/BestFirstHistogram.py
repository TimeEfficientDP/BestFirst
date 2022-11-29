from pqdict import PQDict
import numpy as np 
from collections import defaultdict 
from math import ceil , floor 
import copy 


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
    
    D = defaultdict(dict) 
    D[(0, 0)] = 0 
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
        Q[(j,0)] = (squared_sums[j] -   squared_sums[start]) - (sums[j]  - sums[start] )**2 / (j - start + 1)
    
    while Q: 
        (key, d) = Q.popitem() 
        D[key]= d
        visited.add(key)  
                
        if key[0] == n-1 and key[1] == B-1: 
            break                
        
        if key[1] < B-1:
            for neig in range(key[0]+1, n-(B-1-key[1])+1): 
                if (neig,key[1]+1) not in visited:
                    new_d = d + (squared_sums[neig] - squared_sums[key[0]]) - (sums[neig] - sums[key[0]])**2  / (neig - key[0]  + 1)
                    if new_d < Q.get((neig,key[1]+1), float("inf")):
                        Q[(neig, key[1]+1)] = new_d 
        
    return D 
   




def tech_bound(sequence, B, LB):
    ''' 
    tech bound algorithm 
    Parameters: 
        sequence: data (list) 
        B: number of buckets (int)
        LB: lower bounds (array)
        
    Returns: 
        D: optimal errors (dict)
    ''' 
    
    D = defaultdict(dict) 
    D[(0, 0)] = LB[1,B]
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
        Q[(j,0)] = (squared_sums[j] -   squared_sums[start]) - (sums[j]  - sums[start] )**2 / (j - start + 1) + LB[j+1,B] 
        
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
                       
    return D
    
  

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
   for i in range(1,n): 
       sums[i] = sums[i-1] + sequence[i] 
       squared_sums[i] = squared_sums[i-1] + sequence[i]**2
 
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
 


def bidirectional_tech_bound(sequence, B, LB, LB_reversed):
    ''' 
    bidirectional tech bound algorithm 
    Parameters: 
        sequence: data (list) 
        B: number of buckets (int)
        LB: bounds (array) 
        LB_reversed: bounds in reversed sequence (array)
         
    Returns: 
        mu: optimal error 
    ''' 
    
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
    sums[0] = sequence[0]
    squared_sums[0]= sequence[0]**2 
    for i in range(1,n): 
        sums[i] = sums[i-1] + sequence[i] 
        squared_sums[i] = squared_sums[i-1] + sequence[i]**2
 
    start = 0 
    for j in range(1, n-(B-1)+1): 
        sse = (squared_sums[j] - squared_sums[start]) - (sums[j]  - sums[start] )**2 / (j - start + 1) 
        Q_f[(j,0)] = sse + LB[j+1,B]  # start enquining the zero split 
        squared_error[(start,j)] = sse
        
    end = n-1
    for j in reversed(range(B-1, n-1)): 
       sse = (squared_sums[end] - squared_sums[j]) - ( sums[end] - sums[j]  )**2  / (end - j + 1)
       Q_b[(j,0)] = sse  + LB_reversed[j-1,B]  # now its correct we dont use reversed sums also in the backward case         
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
                    
                if (neig_back , B - k_b[1] - 2) in visited_f and D_b[k_b]  - LB_reversed[ k_b[0]-1, B - k_b[1]]  + squared_error[(neig_back, k_b[0])] + D_f[( neig_back ,  B - k_b[1] - 2)] - LB[neig_back+1 , k_b[1] + 2 ]  < mu :                                        
                    mu = D_b[k_b]  - LB_reversed[ k_b[0]-1, B - k_b[1]]  + squared_error[(neig_back, k_b[0])] + D_f[( neig_back ,  B - k_b[1] - 2)] - LB[neig_back+1 , k_b[1] + 2 ]

        # check stopping condition 
        if D_f[k_f] + D_b[k_b] >= mu :
            break 
    
    return mu



def splitLeftElements(start, end, bins):
    ''' helper subroutine used for bounds computation ''' 
    return list(np.ceil(np.linspace(start=start, stop=end, num=bins + 1)).astype(np.int32))



def compute_bounds(sequence, B): 
    
    ''' utility for the pre-computation of bounds to be used for priority computation '''
    
    n = len(sequence)
    LB = np.zeros((n+1,B+1)) # in this case B is exactly the number of buckets not split 
    UB = np.matrix(np.ones((n+1,B+1)) * np.inf)
    sums = np.zeros(n)
    squared_sums = np.zeros(n) 
    sums[0] = sequence[0]
    squared_sums[0]= sequence[0]**2 
    
    for i in range(1,n): 
        sums[i] = sums[i-1] + sequence[i] 
        squared_sums[i] = squared_sums[i-1] + sequence[i]**2
    
    for i in range(n-1): 
      
        left_elements = n - i #- 1
                        
        for k in range(1, B+1): 
            
            if k < left_elements: 
            
                bins = splitLeftElements(i, n, k)
                                
                all_sse = [] 
                for h in range(len(bins)-1): 
                    low_bin = max(0,  bins[h]-1)
                    top_bin = bins[h+1]-1
                    squared_error = (squared_sums[top_bin] - squared_sums[low_bin]) - (sums[top_bin] - sums[low_bin])**2  / (top_bin - low_bin + 1)
                    all_sse.append(squared_error) 
        
                
                LB[i,k] = min(all_sse) 
                UB[i,k] = sum(all_sse)
                
            
    return LB , UB


def compute_bounds_reversed(sequence, B): 
    
    
    '''´ utility for the pre-computation of bounds to be used for priority computation
    in the backward search'''
    
    
    n = len(sequence)
    LB = np.zeros((n+1,B+1)) # in this case B is exactly the number of buckets not split 
    UB = np.matrix(np.ones((n+1,B+1)) * np.inf)
    sums = np.zeros(n)
    squared_sums = np.zeros(n) 
    sums[0] = sequence[0]
    squared_sums[0]= sequence[0]**2 
    
    for i in range(1,n): 
        sums[i] = sums[i-1] + sequence[i] 
        squared_sums[i] = squared_sums[i-1] + sequence[i]**2
    
    for i in range(n): 
           
        elements_to_split = i 
                        
        for k in range(1, B+1): 

            if k < elements_to_split: 
            
                bins = splitLeftElements(0, i, k)
                
                all_sse = [] 
                for h in range(len(bins)-1): 
                    low_bin = max(0,  bins[h])
                    top_bin = bins[h+1]
                    squared_error = (squared_sums[top_bin] - squared_sums[low_bin]) - (sums[top_bin] - sums[low_bin])**2  / (top_bin - low_bin + 1)
                    all_sse.append(squared_error) 
      
                   
                LB[i,k] = min(all_sse) 
                UB[i,k] = sum(all_sse)
                
            
    return LB , UB






