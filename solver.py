#!/usr/bin/python
# -*- coding: utf-8 -*-



from collections import namedtuple
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





def solve_it(input_data):
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

    # DFS_BnB
    solver = DFS_BnB(items, capacity)
    nb_selected = solver.solve()
    value = solver.objective
    taken = [0]*len(items)
    for item in solver.selected_items:
        taken[item.index] = 1
    
    # prepare the solution in the specified output format
    output_data = str(value) + ' ' + str(1) + '\n'
    output_data += ' '.join(map(str, taken))
    return output_data


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        file_location = sys.argv[1].strip()
        with open(file_location, 'r') as input_data_file:
            input_data = input_data_file.read()
        print(solve_it(input_data))
    else:
        print('This test requires an input file.  Please select one from the data directory. (i.e. python solver.py ./data/ks_4_0)')

