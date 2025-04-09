import numpy as np
from collections import deque
import heapq
from typing import List, Tuple, Set, Dict
"""
Do not import any other package unless allowed by te TAs in charge of the lab.
Do not change the name of any of the functions below.
"""

def findNeighbors(row, col, n):
    l = []
    if (row < n - 1): l.append((row + 1, col))
    if (col < n - 1): l.append((row, col + 1))
    if (row > 0): l.append((row - 1, col))
    if (col > 0): l.append((row, col + 1))

def encode(node: np.ndarray):
    d = {}
    for row in range(3):
        for col in range(3):
            d[node[row][col]] = row * 3 + col
    encoded = ""
    for i in range(0, 9):
        encoded += str(d[i])
    return encoded

def decode(state : str):
    result = np.zeros((3,3))
    for i in range(len(state)):
        s = int(state[i])
        result[s // 3][s % 3] = i

    return result.astype(int)

def identity(x):
    return x

def zerofunction(goal, curr):
    return 0

def reciprocal(x):
    return 1 / (x + 1)

def get_dt(goal, curr):
    count = 9
    for r in range(3):
        for c in range(3):
            if goal[r][c] == curr[r][c]:
                count -= 1
    return count

def get_md(goal, curr):
    g_e = encode(goal)
    c_e = encode(curr)

    md = 0
    for i in range(len(g_e)):
        g_row, g_col = int(g_e[i]) // 3, int(g_e[i]) % 3
        c_row, c_col = int(c_e[i]) // 3, int(c_e[i]) % 3

        md += abs(g_col - c_col) + abs(c_row - g_row)
    return md

def isSolvable(initial: np.ndarray, goal:np.ndarray):
    arr = initial.flatten()
    icount = 0
    for j in range(8):
        for k in range(j + 1, 9):
            if (arr[j] and arr[k] and (arr[j] > arr[k])):
                icount+= 1
    arr = goal.flatten()
    gcount = 0
    for j in range(8):
        for k in range(j + 1, 9):
            if (arr[j] and arr[k] and (arr[j] > arr[k])):
                gcount+= 1
    return gcount % 2 == icount % 2

def find_zero_index(matrix):
    for row_idx in [0,1,2]:
        for col_idx in [0,1,2]:
            if matrix[row_idx][col_idx] == 0:
                return (row_idx, col_idx)

def nextBoardPositions(initial: np.ndarray) -> List[np.ndarray]:
    nextBoardPosi = []

    # Find the position of the blank tile (0)
    blank_pos = find_zero_index(initial)
    possibleMoves = [(-1, 0, 'U'), (1, 0, 'D'), (0, -1, 'L'), (0, 1, "R")]

    for move in possibleMoves:
        new_row = blank_pos[0] + move[0]
        new_column = blank_pos[1] + move[1]

        if 0 <= new_row < 3 and 0 <= new_column < 3:
            new_board = initial.copy()
            new_board[blank_pos[0], blank_pos[1]], new_board[new_row, new_column] = (
                new_board[new_row, new_column],
                new_board[blank_pos[0], blank_pos[1]],
            )

            nextBoardPosi.append((new_board, move[2]))
    
    return nextBoardPosi

def astar(initial: np.ndarray, goal: np.ndarray, h, g) -> Tuple[List[str], int, int]:

    if not isSolvable(initial, goal):
        return ([], 0, 0)
    # Initialise an empty priority queue
    openlist = []
    closedlist = []

    count = 1

    # Push the first element
    heapq.heappush(openlist, (g(0) + h(goal, initial), (encode(initial), "")))

    while openlist:
        value, (encodedstate, movestaken) = heapq.heappop(openlist)
        if encodedstate not in closedlist: 
            state = decode(encodedstate)
            nextBoardPosi = nextBoardPositions(state)

            for nextstate, movetype in nextBoardPosi:
                if encode(nextstate) == encode(goal):
                    path = list(movestaken)
                    path.append(movetype)
                    numberofnodes = count + 1
                    return (path, numberofnodes, len(path))

                if encode(nextstate) not in closedlist:
                    heapq.heappush(openlist, (g(len(movestaken) + 1) + h(goal, nextstate), (encode(nextstate), movestaken+movetype)))
                    count += 1  
            closedlist.append(encode(state))

def bfs(initial: np.ndarray, goal: np.ndarray) -> Tuple[List[str], int]:
    """
    Implement Breadth-First Search algorithm to solve 8-puzzle problem.
    
    Args:
        initial (np.ndarray): Initial state of the puzzle as a 3x3 numpy array.
                            Example: np.array([[1, 2, 3], [4, 0, 5], [6, 7, 8]])
                            where 0 represents the blank space
        goal (np.ndarray): Goal state of the puzzle as a 3x3 numpy array.
                          Example: np.array([[1, 2, 3], [4, 5, 6], [7, 8, 0]])
    
    Returns:
        Tuple[List[str], int]: A tuple containing:
            - List of moves to reach the goal state. Each move is represented as
              'U' (up), 'D' (down), 'L' (left), or 'R' (right), indicating how
              the blank space should move
            - Number of nodes expanded during the search

    Example return value:
        (['R', 'D', 'R'], 12) # Means blank moved right, down, right; 12 nodes were expanded
              
    """
    # TODO: Implement this function
    
    directions, moves, cost =  astar(initial, goal, zerofunction, identity)
    return directions, moves

def dfs(initial: np.ndarray, goal: np.ndarray) -> Tuple[List[str], int]:
    """
    Implement Depth-First Search algorithm to solve 8-puzzle problem.
    
    Args:
        initial (np.ndarray): Initial state of the puzzle as a 3x3 numpy array
        goal (np.ndarray): Goal state of the puzzle as a 3x3 numpy array
    
    Returns:
        Tuple[List[str], int]: A tuple containing:
            - List of moves to reach the goal state
            - Number of nodes expanded during the search
    """
    # TODO: Implement this function

    directions, moves, cost =   astar(initial, goal, zerofunction, reciprocal)
    return directions, moves

def dijkstra(initial: np.ndarray, goal: np.ndarray) -> Tuple[List[str], int, int]:
    """
    Implement Dijkstra's algorithm to solve 8-puzzle problem.
    
    Args:
        initial (np.ndarray): Initial state of the puzzle as a 3x3 numpy array
        goal (np.ndarray): Goal state of the puzzle as a 3x3 numpy array
    
    Returns:
        Tuple[List[str], int, int]: A tuple containing:
            - List of moves to reach the goal state
            - Number of nodes expanded during the search
            - Total cost of the path for transforming initial into goal configuration
            
    """
    # TODO: Implement this function

    return astar(initial, goal, zerofunction, identity)

def astar_dt(initial: np.ndarray, goal: np.ndarray) -> Tuple[List[str], int, int]:
    """
    Implement A* Search with Displaced Tiles heuristic to solve 8-puzzle problem.
    
    Args:
        initial (np.ndarray): Initial state of the puzzle as a 3x3 numpy array
        goal (np.ndarray): Goal state of the puzzle as a 3x3 numpy array
    
    Returns:
        Tuple[List[str], int, int]: A tuple containing:
            - List of moves to reach the goal state
            - Number of nodes expanded during the search
            - Total cost of the path for transforming initial into goal configuration
              
    
    """
    # TODO: Implement this function

    return astar(initial, goal, get_dt, identity)

def astar_md(initial: np.ndarray, goal: np.ndarray) -> Tuple[List[str], int, int]:
    """
    Implement A* Search with Manhattan Distance heuristic to solve 8-puzzle problem.
    
    Args:
        initial (np.ndarray): Initial state of the puzzle as a 3x3 numpy array
        goal (np.ndarray): Goal state of the puzzle as a 3x3 numpy array
    
    Returns:
        Tuple[List[str], int, int]: A tuple containing:
            - List of moves to reach the goal state
            - Number of nodes expanded during the search
            - Total cost of the path for transforming initial into goal configuration
    """
    # TODO: Implement this function
    
    return astar(initial, goal, get_md, identity)


# Example test case to help verify your implementation
if __name__ == "__main__":
    # Example puzzle configuration
    initial_state = np.array([
        [1, 2, 3],
        [4, 0, 5],
        [6, 7, 8]
    ])

    goal_state = np.array([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 0]
    ])
    
    # Test each algorithm
    print("Testing BFS...")
    bfs_moves, bfs_expanded = bfs(initial_state, goal_state)
    print(f"BFS Solution: {bfs_moves}")
    print(f"Nodes expanded: {bfs_expanded}")
    
    print("\nTesting DFS...")
    dfs_moves, dfs_expanded = dfs(initial_state, goal_state)
    print(f"DFS Solution: {dfs_moves}")
    print(f"Nodes expanded: {dfs_expanded}")
    
    print("\nTesting Dijkstra...")
    dijkstra_moves, dijkstra_expanded, dijkstra_cost = dijkstra(initial_state, goal_state)
    print(f"Dijkstra Solution: {dijkstra_moves}")
    print(f"Nodes expanded: {dijkstra_expanded}")
    print(f"Total cost: {dijkstra_cost}")
    
    print("\nTesting A* with Displaced Tiles...")
    dt_moves, dt_expanded, dt_fscore = astar_dt(initial_state, goal_state)
    print(f"A* (DT) Solution: {dt_moves}")
    print(f"Nodes expanded: {dt_expanded}")
    print(f"Total cost: {dt_fscore}")
    
    print("\nTesting A* with Manhattan Distance...")
    md_moves, md_expanded, md_fscore = astar_md(initial_state, goal_state)
    print(f"A* (MD) Solution: {md_moves}")
    print(f"Nodes expanded: {md_expanded}")
    print(f"Total cost: {md_fscore}")