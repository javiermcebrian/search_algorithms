#!/usr/bin/python
# -*- coding: utf-8 -*-


#########
# NOTES #
#########
# Algorithms (knapsack): Greedy (fast), Dynamic Programing (memory limits), BnB (find fastest way to compute: relaxation/bound)
#########
# Algorithms (getting started):
# - Step 1: Greedy
# - Step 2: Smarter greedy, DP, CP, LS, MIP
# - Step 3: Relax (give quality guarantee)
#########
# Algorithms (getting better):
# - Step 1: Baseline (greedy)
# - Step 2:
#       Quality: CP, MIP
#       Scalability: Local Search
# - Step 3: Hybrids
#########
# Reward: Scalability + Quality
#########






from collections import namedtuple
import numpy as np


Item = namedtuple("Item", ['index', 'value', 'weight'])




# Node: could be seen as a physical knapsack:
# - Level: depth (number of items evaluated - 1)
# - Value: accumulated value inside the knapsack
# - Room: space left in the knapsack
# - Taken: items in the knapsack (if "variable" Node has "value" = taken, that Node's level is appended to Taken list)
class Node(object):
    def __init__(self, level, value, room, taken):
        self.level = level
        self.value = value
        self.room = room
        self.taken = taken









class DFS_BnB(object):
    def __init__(self, items, capacity):
        self.items = sorted(items, key = lambda k: float(k.value)/float(k.weight), reverse = True)
        self.capacity = capacity
        self.selected_items = []
        self.objective = 0
        
    def linear_relaxation(self, node):
        value = node.value        
        room = node.room
        for item in self.items[node.level:]:
            if room - item.weight >= 0:
                value += item.value
                room -= item.weight
            else:
                return value + (float(item.value) / float(item.weight)) * room
        return value
        
    def solve(self):
        root = Node(0, 0, self.capacity, [])
        best = root
        stack = [root] # LIFO queue: explore tree first in depth rather than width -> DFS?. Keep nodes that needs to be branched.
        while len(stack) > 0:
            current = stack.pop()
            # Doesn't matter if leaf or not: check if better than best            
            if current.value > best.value:
                best = current
            # Check if not leaf            
            if current.level < len(self.items):
                # RIGHT
                right = Node(current.level + 1, current.value, current.room, list(current.taken)) # First append right to LIFO, due to we want to explore first in depth
                bound = self.linear_relaxation(right)
                if bound > best.value: # If optimistic estimate is above best's value, append to LIFO
                    stack.append(right)
                # LEFT
                item = self.items[current.level]                
                if item.weight <= current.room: # Check if item to consider fits in knapsack
                    taken = list(current.taken)
                    taken.append(current.level) # Update taken Nodes
                    left = Node(current.level + 1, current.value + item.value, current.room - item.weight, taken)
                    bound = self.linear_relaxation(left)
                    if bound > best.value: # If optimistic estimate is above best's value, append to LIFO
                        stack.append(left)
        # Solution
        self.selected_items = [self.items[id] for id in best.taken]
        self.objective = best.value
        return len(best.taken)









class BFS_BnB(object):
    def __init__(self, items, capacity):
        self.items = sorted(items, key = lambda k: float(k.value)/float(k.weight), reverse = True)
        self.capacity = capacity
        self.selected_items = []
        self.objective = 0
        
    def linear_relaxation(self, node):
        value = node.value        
        room = node.room
        for item in self.items[node.level:]:
            if room - item.weight >= 0:
                value += item.value
                room -= item.weight
            else:
                return value + (float(item.value) / float(item.weight)) * room
        return value

    def solve(self):
        root = Node(0, 0, self.capacity, [])
        root.bound = self.linear_relaxation(root)
        best = root
        parent = root
        space = [root] # Search Space
        while parent != None: # [STOP CONDITION] : If all nodes in space are worse (node.bound < best.value) than a found solution, stop it
            # LEFT
            item = self.items[parent.level] 
            if item.weight <= parent.room: # Check if item to consider fits in knapsack
                taken = list(parent.taken)
                taken.append(parent.level) # Update taken Nodes
                left = Node(parent.level + 1, parent.value + item.value, parent.room - item.weight, taken)
                left.bound = self.linear_relaxation(left)
                if left.bound > best.value: # Check if it is worthwhile to add it to search space
                    space += [left]
                    if left.value > best.value: # Check if best
                        best = left
            # RIGHT
            right = Node(parent.level + 1, parent.value, parent.room, list(parent.taken))
            right.bound = self.linear_relaxation(right)
            if right.bound > best.value: # Check if it is worthwhile to add it to search space
                space += [right]
                if right.value > best.value: # Check if best
                    best = right            
            # Remove wortheless nodes            
            space = [node for node in space if node.bound > best.value]
            # Get best parent
            space = sorted(space, key = lambda k: k.bound, reverse = True)
            parent = None
            for node in space:
                if node.level < len(self.items):
                    parent = node
                    space.remove(node)
                    break      
        # Solution
        self.selected_items = [self.items[id] for id in best.taken]
        self.objective = best.value
        return len(best.taken)












# https://pdfs.semanticscholar.org/33c2/3911062d41500b10b5718b9c0c07414fc847.pdf
# https://www.aaai.org/Papers/AAAI/1996/AAAI96-043.pdf
class LDS_BnB(object):
    def __init__(self, items, capacity):
        self.items = sorted(items, key = lambda k: float(k.value)/float(k.weight), reverse = True)
        self.capacity = capacity
        self.selected_items = []
        self.objective = 0
        
    def linear_relaxation(self, node):
        value = node.value        
        room = node.room
        for item in self.items[node.level:]:
            if room - item.weight >= 0:
                value += item.value
                room -= item.weight
            else:
                return value + (float(item.value) / float(item.weight)) * room
        return value
    
    def wave(self, node, k, best):
        # LEAF
        item = self.items[node.level]
        if node.level == len(self.items) or self.linear_relaxation(node) <= best.value:
            return best
        # LEFT
        if item.weight <= node.room:
            taken = list(node.taken)
            taken.append(node.level) # Update taken Nodes
            left = Node(node.level + 1, node.value + item.value, node.room - item.weight, taken)
            # BEST
            if left.value > best.value:
                best = left
            if self.linear_relaxation(left) > best.value:
                best = self.wave(left, k, best)
        # RIGHT
        if k > 0:
            right = Node(node.level + 1, node.value, node.room, list(node.taken))
            # BEST        
            if right.value > best.value:
                best = right
            if self.linear_relaxation(right) > best.value:
                best = self.wave(right, k - 1, best)
        return best
        
    def solve(self):
        root = Node(0, 0, self.capacity, [])
        best = root
        # Iterate LDS
        for k in range(len(self.items) + 1):
            best = self.wave(root, k, best)
            if best.value == self.linear_relaxation(best): # so fast!!!! --> HEURISTICS ??? / LOWER BOUND
                break
        # Solution
        self.selected_items = [self.items[id] for id in best.taken]
        self.objective = best.value
        return len(best.taken)





                    
            
        





class knapsack_dp(object):
    def __init__(self, items, capacity):
        self.items = items
        self.capacity = capacity
        self.selected_items = []
        self.objective = 0
    def O(self, k, j):
        if j==0:
            return 0
        elif self.items[j].weight <= k:
            return max(self.O(k,j-1), self.items[j].value+self.O(k-self.items[j].weight,j-1))
        else:
            return self.O(k,j-1)
    def solve(self):
        # construct table
        table = [[0] * (self.capacity + 1) for _ in range(len(self.items) + 1)]
        for i, item in enumerate(self.items):
            for c in range(self.capacity + 1):
                if c < item.weight:
                    table[i+1][c] = table[i][c]
                else:
                    table[i+1][c] = max(table[i][c], item.value + table[i][c - item.weight])
        self.objective = table[len(self.items)][self.capacity]
        # trace back table
        selected = []
        capacity_status = self.capacity
        for i in range(len(self.items), 0, -1):
            if table[i][capacity_status] != table[i-1][capacity_status]:
                selected.append(self.items[i-1])
                capacity_status -= self.items[i-1].weight
        self.selected_items = selected
        # return number of items selected
        return len(selected)
            




def solve_it(input_data, variant):
    # Modify this code to run your optimization algorithm

    # parse the input
    lines = input_data.split('\n')

    firstLine = lines[0].split()
    item_count = int(firstLine[0])
    capacity = int(firstLine[1])

    items = []

    for i in range(1, item_count+1):
        line = lines[i]
        parts = line.split()
        items.append(Item(i-1, int(parts[0]), int(parts[1])))
    
    if variant == 'greedy':
        # a trivial greedy algorithm for filling the knapsack
        # it takes items in-order until the knapsack is full
        value = 0
        weight = 0
        taken = [0]*len(items)
        # Sort by descendent density
        items = sorted(items, key = lambda k: float(k.value)/float(k.weight), reverse = True)
        # Sort by descendent value
#        items = sorted(items, key = lambda k: k.value, reverse = True)
        # Sort by ascendent weight
#        items = sorted(items, key = lambda k: k.weight, reverse = False)
        # Sort by descendent density with penalty squared: V / (W/C)^2
#        items = sorted(items, key = lambda k: float(k.value)/(float(k.weight)/float(capacity))**2, reverse = True)
        # Sort by descendent density with penalty tanh: V / tanh(W/C)
#        items = sorted(items, key = lambda k: float(k.value)/np.tanh(0.01 * float(k.weight)/float(capacity)), reverse = True)
        # Sequence items in order, and add them if fit
        for item in items:
            if weight + item.weight <= capacity:
                taken[item.index] = 1
                value += item.value
                weight += item.weight
        nb_selected = sum(taken)

    if variant == 'knapsack_dp':
        solver = knapsack_dp(items, capacity)
        nb_selected = solver.solve()
        value = solver.objective
        taken = [0]*len(items)
        for item in solver.selected_items:
            taken[item.index] = 1
    
    if variant == 'DFS_BnB':
        solver = DFS_BnB(items, capacity)
        nb_selected = solver.solve()
        value = solver.objective
        taken = [0]*len(items)
        for item in solver.selected_items:
            taken[item.index] = 1
    if variant == 'BFS_BnB':
        solver = BFS_BnB(items, capacity)
        nb_selected = solver.solve()
        value = solver.objective
        taken = [0]*len(items)
        for item in solver.selected_items:
            taken[item.index] = 1
    if variant == 'LDS_BnB':
        solver = LDS_BnB(items, capacity)
        nb_selected = solver.solve()
        value = solver.objective
        taken = [0]*len(items)
        for item in solver.selected_items:
            taken[item.index] = 1
    
    # prepare the solution in the specified output format
    output_data = str(value) + ' ' + str(0) + '\n'
    output_data += ' '.join(map(str, taken))
    return nb_selected, value, taken#output_data





if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        file_location = sys.argv[1].strip()
        with open(file_location, 'r') as input_data_file:
            input_data = input_data_file.read()
#        print(solve_it(input_data, 'knapsack_dp'))
#        print(solve_it(input_data, 'DFS_BnB'))
#        print(solve_it(input_data, 'BFS_BnB'))
#        print(solve_it(input_data, 'LDS_BnB'))
            res = solve_it(input_data, 'knapsack_dp')
    else:
        print('This test requires an input file.  Please select one from the data directory. (i.e. python solver.py ./data/ks_4_0)')



#if __name__ == '__main__':
#    import sys
#    if len(sys.argv) > 1:
#        file_location = sys.argv[1].strip()
#        with open(file_location, 'r') as input_data_file:
#            input_data = input_data_file.read()
#        print(solve_it(input_data))
#    else:
#        print('This test requires an input file.  Please select one from the data directory. (i.e. python solver.py ./data/ks_4_0)')

