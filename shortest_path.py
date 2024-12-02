"""
discrete computer project
"""

import argparse
import random
from datetime import datetime
import heapq
import numpy as np
from numba import jit

start_time = datetime.now()

def create_node(position: tuple[int, int], g: float = float('inf'),
                h: float = 0.0, parent: dict = None) -> dict:
    """
    Create a node for the A* algorithm.
    Args:
        position: (x, y) coordinates of the node
        g: Cost from start to this node (default: infinity)
        h: Estimated cost from this node to goal (default: 0)
        parent: Parent node (default: None)
    
    Returns:
        Dictionary containing node information
    """
    return {
        'position': position,
        'g': g,
        'h': h,
        'f': g + h,
        'parent': parent
    }

@jit(nopython=True)
def neighbor_nodes(point, rows, cols) -> list:
    """Finds neighbor points.
    Args: 
        point (tuple): Current position (x, y).
        rows (int): an amount of rows in matrix.
        cols (int): an amount of rows in matrix.
    Returns:
        list: list of neighbour points.
    """
    totresult = []
    result = [(point[0] + 1, point[1]), (point[0] - 1, point[1]),
              (point[0], point[1] + 1), (point[0], point[1] - 1)]
    for pair in result:
        if 0 <= pair[0] < rows and 0 <= pair[1] < cols:
            totresult.append(pair)
    return totresult 

@jit(nopython=True)
def h_x(curr_point, goal):
    """Calculates the heuristic (h(x)) from the current point to the goal
    Args:
        curr_point (tuple): Current position (x, y).
        goal (tuple): Goal position (x, y).

    Returns:
        int: The Manhattan distance between the current point and the goal."""
    return abs(curr_point[0] - goal[0]) + abs(curr_point[1] - goal[1])

@jit(nopython=True)
def g_x(curr_height, start_height, step=1):
    """
    Finds heuristic cost of current point from start.
    Args:
        matr (list): matrix og heights.
        curr_point (tuple): current position (x, y).
        start (tuple): start position (x, y).

    Returns:
        int: The hueristic cost of current point.
    """
    distance = curr_height - start_height
    return (step ** 2 + distance ** 2) ** 0.5

def astar_search(matr, start, goal):
    """
    Implements the A* pathfinding algorithm to find the shortest path 
    in a grid from start to goal.
    Args:
        matr (list[int]): matrix of heights.
        start tuple(int): start position (x, y).
        goal: tuple(int): goal position (x, y).
    Returns:
        list[tuple[int]]: the shortest path form start to goal.
    """
    rows, cols = matr.shape
    closed_set = set()
    start_node = create_node(position=start,g=0,
        h=h_x(start, goal))
    open_set = [(start_node['f'], start)]
    node_map = {start: start_node}

    while open_set:
        _, current_pos = heapq.heappop(open_set)
        current_node = node_map[current_pos]

        if current_pos == goal:
            path = []
            while current_node:
                path.insert(0, current_node["position"])
                current_node = current_node["parent"]
            return path

        closed_set.add(current_node["position"])

        for neighbor in neighbor_nodes(current_node["position"], rows, cols):
            if neighbor in closed_set:
                continue

            cost = current_node["g"] + g_x(matr[current_node["position"]], matr[neighbor])
            neighbor_node = create_node(position=neighbor, g=cost, h=h_x(neighbor, goal),
                    parent=current_node)
            neighbor_node["parent"] = current_node

            if neighbor not in node_map or node_map[neighbor]["g"] > cost:
                node_map[neighbor] = neighbor_node
                heapq.heappush(open_set, (neighbor_node['f'], neighbor))

    return None

@jit(nopython=True)
def generate_matrix(rows, cols):
    """Generates matrix of random numbers with specific size.
    Args:
        rows (int): an amount of rows in matrix.
        cols (int): an amount of rows in matrix.
    Returns:
        list[int]: a matrix which is consisted of random numbers, which represent
        the height of a point.
    """
    return np.random.randint(1, 10001, size=(rows, cols))

def main():
    parser = argparse.ArgumentParser(description="A* Pathfinding Algorithm")
    parser.add_argument("--rows", type=int, default=10000, help="Number of rows in the matrix")
    parser.add_argument("--cols", type=int, default=10000, help="Number of columns in the matrix")

    args = parser.parse_args()

    rows, cols = args.rows, args.cols

    matrix = generate_matrix(rows, cols)

    start = (random.randint(0, rows - 1), random.randint(0, cols - 1))
    goal = (random.randint(0, rows - 1), random.randint(0, cols - 1))

    print("Start:", start)
    print("Goal:", goal)

    path = astar_search(matrix, start, goal)

    if path:
        print("Path found:", len(path), path)
    else:
        print("No path found.")

    end_time = datetime.now()
    print(f"Execution time: {end_time - start_time}")

if __name__ == "__main__":
    main()
