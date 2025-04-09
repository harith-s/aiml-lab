import heapq

available_locations = []

for i in range(8):
    for j in range(8):
        available_locations.append([i, j])

def get_neighbours(state, available_locations):
    neighbours = []
    for position in available_locations:
        next_state = state.copy()
        next_state.append(position)
        new_available_locations = available_locations.copy()
        for remposi in available_locations:
            if ( position[0] == remposi[0] ) or ( position[1] == remposi[1] ) or ( position[0] + position[1] == remposi[0] + remposi[1] ) or ( position[0] - position[1] == remposi[0] - remposi[1] ):
                new_available_locations.remove(remposi)
        neighbours.append([next_state, new_available_locations])
    return neighbours

def g(state):
    return len(state)

def h1(state, available_locations):
    return 8 - len(state)

def h2(state, available_locations):
    return len(available_locations) / 28

def astar(available_locations, h):
    
    openList = []
    clsoedList = []

    for first_position in available_locations:
        state = []
        state.append(first_position)
        new_available_locations = available_locations.copy()
        for remposi in available_locations:
            if ( first_position[0] == remposi[0] ) or ( first_position[1] == remposi[1] ) or ( first_position[0] + first_position[1] == remposi[0] + remposi[1] ) or ( first_position[0] - first_position[1] == remposi[0] - remposi[1] ):
                new_available_locations.remove(remposi)
        heapq.heappush(openList, (h(first_position, new_available_locations) + g(first_position), (state, new_available_locations)))

    count = 0

    while(openList):
        priority, ( state, available_locations ) = heapq.heappop(openList)
        print(state)
        if len(state) == 8:
            print(state)
            print(count)
            break
        if state not in clsoedList:
            count = count + 1
            for next_state, new_available_locations in get_neighbours(state, available_locations):
                if next_state not in clsoedList:
                    heapq.heappush(openList, (g(next_state) + h(next_state, new_available_locations), (next_state, new_available_locations)))
        clsoedList.append(state)

astar(available_locations, h1)

available_locations = []

for i in range(8):
    for j in range(8):
        available_locations.append([i, j])

astar(available_locations, h2)