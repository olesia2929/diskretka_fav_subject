"""Implementation of A Star algorithm"""
import argparse
import random
from datetime import datetime
import heapq
import numpy as np
from numba import jit

start_time = datetime.now()

def create_node(position: tuple, g: float = float('inf'),
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
    Examples:
    >>> node = create_node((0, 0), g=5, h=3, parent=None)
    >>> node['position']
    (0, 0)
    >>> node['g']
    5
    >>> node['h']
    3
    >>> node['f']
    8
    >>> node['parent'] is None
    True
    """
    return {
        'position': position,
        'g': g,
        'h': h,
        'f': g + h,
        'parent': parent
    }

@jit(nopython=True)
def neighbor_nodes(point: tuple, rows: int, cols: int) -> list:
    """Finds neighbor points.
    Args: 
        point (tuple): Current position (x, y).
        rows (int): an amount of rows in matrix.
        cols (int): an amount of rows in matrix.
    Returns:
        list: list of neighbour points.
    >>> neighbor_nodes((1, 1), 3, 3)
    [(2, 1), (0, 1), (1, 2), (1, 0)]
    >>> neighbor_nodes((0, 0), 3, 3)
    [(1, 0), (0, 1)]
    """
    totresult = []
    result = [(point[0] + 1, point[1]), (point[0] - 1, point[1]),
              (point[0], point[1] + 1), (point[0], point[1] - 1)]
    for pair in result:
        if 0 <= pair[0] < rows and 0 <= pair[1] < cols:
            totresult.append(pair)
    return totresult 

@jit(nopython=True)
def h_x(curr_point: tuple, goal: tuple):
    """Calculates the heuristic (h(x)) from the current point to the goal
    Args:
        curr_point (tuple): Current position (x, y).
        goal (tuple): Goal position (x, y).

    Returns:
        int: The Manhattan distance between the current point and the goal.
    >>> h_x((0, 0), (2, 2))
    4
    >>> h_x((1, 1), (1, 1))
    0
    >>> h_x((5, 5), (2, 3))
    5
    """
    return abs(curr_point[0] - goal[0]) + abs(curr_point[1] - goal[1])

@jit(nopython=True)
def g_x(curr_height: int, start_height: int, step=1):
    """
    Finds heuristic cost of current point from start.
    Args:
        matr (list): matrix og heights.
        curr_point (tuple): current position (x, y).
        start (tuple): start position (x, y).

    Returns:
        int: The hueristic cost of current point.
    >>> g_x(10, 5)
    5.0990195135927845
    >>> g_x(7, 7)
    1.0
    >>> g_x(15, 5)
    10.04987562112089
    """
    distance = curr_height - start_height
    return (step ** 2 + distance ** 2) ** 0.5

def astar_search(matr: list[list], start: tuple, goal: tuple):
    """
    Implements the A* pathfinding algorithm to find the shortest path 
    in a grid from start to goal.
    Args:
        matr (list[int]): matrix of heights.
        start tuple(int): start position (x, y).
        goal: tuple(int): goal position (x, y).
    Returns:
        list[tuple[int]]: the shortest path form start to goal or None.
    >>> matr = np.array([
    ...     [1, 1, 1, 1],
    ...     [1, 1000, 1000, 1],
    ...     [1, 1, 1, 1],
    ...     [1, 1, 1, 1],
    ... ])
    >>> start = (0, 0)
    >>> goal = (3, 3)
    >>> path = astar_search(matr, start, goal)
    >>> path[0] == start
    True
    >>> path[-1] == goal
    True
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
def generate_matrix(rows: int, cols: int):
    """Generates matrix of random numbers with specific size.
    Args:
        rows (int): an amount of rows in matrix.
        cols (int): an amount of rows in matrix.
    Returns:
        list[int]: a matrix which is consisted of random numbers, which represent
        the height of a point.
    >>> matrix = generate_matrix(2, 2)
    >>> matrix.shape
    (2, 2)
    >>> (matrix >= 1).all() and (matrix <= 10000).all()
    np.True_
    """
    return np.random.randint(1, 10001, size=(rows, cols))

def main():
    parser = argparse.ArgumentParser(description="A* Pathfinding Algorithm")
    parser.add_argument("--rows", type=int, default=10000, help="Number of rows in the matrix")
    parser.add_argument("--cols", type=int, default=10000, help="Number of columns in the matrix")
    parser.add_argument("--start", type=lambda s: tuple(map(int, s.split(','))),
    default=(random.randint(0, 10000), random.randint(0, 10000)), help="Coordinates of start point")
    parser.add_argument("--goal", type=lambda s: tuple(map(int, s.split(','))),
    default=(random.randint(0, 10000), random.randint(0, 10000)), help="Coordinates of goal point")

    args = parser.parse_args()
    rows, cols = args.rows, args.cols
    start, goal = args.start, args.goal

    matrix = generate_matrix(rows, cols)

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
if __name__ == '__main__':
    import doctest
    print(doctest.testmod())
