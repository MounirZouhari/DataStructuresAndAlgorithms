class Graph:
    """
    A class representing a graph using an adjacency list.

    Args:
        directed (bool, optional): If True, the graph is directed. Defaults to False.
        weighted (bool, optional): If True, the graph is weighted. Defaults to False.

    Attributes:
        directed (bool): Indicates whether the graph is directed.
        weighted (bool): Indicates whether the graph is weighted.
        graph (dict): Dictionary storing adjacency lists.
                      - If weighted: {node: {neighbor: weight, ...}}
                      - If unweighted: {node: {neighbor1, neighbor2, ...}} (set)
    """

    def __init__(self, directed=False, weighted=False):
        """
        Initializes the graph.

        Args:
            directed (bool, optional): If True, the graph is directed. Defaults to False.
            weighted (bool, optional): If True, the graph is weighted. Defaults to False.
        """
        self.directed = directed
        self.weighted = weighted
        self.graph = {}  # Dictionary to store adjacency lists
 
    def add_vertex(self, u):
        """
        Adds a vertex to the graph if it doesn't already exist.

        Args:
            u (hashable): The vertex to be added. Can be a string, number, or object.

        Returns:
            bool: True if the vertex was added, False if it already exists.
        """
        if u in self.graph:
            return False  # Vertex already exists, no need to add it
        self.graph[u] = {} if self.weighted else set()
        return True  # Vertex successfully added

    def add_edge(self, u, v, weight=None):
        """
        Adds an edge between two vertices.

        - If the graph is weighted, assigns the given weight.
        - If the graph is unweighted, stores only the existence of the edge.
        - Automatically adds missing vertices.

        Args:
            u (hashable): The starting vertex.
            v (hashable): The ending vertex.
            weight (int, optional): The weight of the edge. Required for weighted graphs.

        Raises:
            ValueError: If an invalid weight is provided.
            ValueError: If the edge already exists.
        """
        if u not in self.graph:
            self.add_vertex(u)
        if v not in self.graph:
            self.add_vertex(v)

        if self.weighted and weight is None:
            raise ValueError(f"Graph is weighted, but edge ({u}, {v}) has no weight.")
        if not self.weighted and weight is not None:
            raise ValueError(f"Graph is unweighted, but edge ({u}, {v}) has a weight.")

        if v in self.graph[u]:
            raise ValueError(f"Edge ({u}, {v}) already exists.")

        if self.weighted:
            self.graph[u][v] = weight
        else:
            self.graph[u].add(v)

        if not self.directed:
            if self.weighted:
                self.graph[v][u] = weight
            else:
                self.graph[v].add(u)

    def has_vertex(self, u):
        """
        Checks if a vertex exists in the graph.

        Args:
            u (hashable): The vertex to check.

        Returns:
            bool: True if the vertex exists, False otherwise.
        """
        return u in self.graph

    def has_edge(self, u, v):
        """
        Checks if an edge exists between two vertices.

        Args:
            u (hashable): The starting vertex.
            v (hashable): The ending vertex.

        Returns:
            bool: True if the edge exists, False otherwise.
        """
        if not (self.has_vertex(u) and self.has_vertex(v)):  #  checking for missing vertices
            return False
        return v in self.graph[u]  

    
    def get_weight(self, u, v):
        """
        Retrieves the weight of an edge between two vertices.

        Args:
            u (hashable): The starting vertex.
            v (hashable): The ending vertex.

        Returns:
            int or float: The weight of the edge.

        Raises:
            ValueError: If the graph is unweighted.
            KeyError: If the edge does not exist.
        """
        if not self.weighted:
            raise ValueError("Graph is unweighted")
        if not self.has_edge(u, v):  
            raise KeyError(f"Edge ({u}, {v}) does not exist")
        
        return self.graph[u][v]
    
    def remove_vertex(self,u):
        return
    def remove_edge(self,u,v):
        return
    def get_neighbours(self,u):
        return
    def size(self):
        """
        Returns the number of vertices and edges in the graph.

        Returns:
            tuple: (num_vertices, num_edges)
        """
        num_vertices = len(self.graph)

        num_edges = sum(len(neighbors) for neighbors in self.graph.values())

        # If the graph is undirected, each edge is counted twice
        if not self.directed:
            num_edges //= 2

        return num_vertices, num_edges

    def is_empty(self):
        return
    def display(self):
        return
    def degree(self,u):
        return
    def clear(self):
        return
    def copy(self):
        return
    def to_adjacency_matrix(self):
        return
    def to_incidence_matrix(self):
        return
    
    def to_edge_list(self):
        return
    
    def is_connected(self):
        return
    def is_directed(self):
        return
    def is_weighhted(self):
        return
    def bfs(self,start):
        return
    def dfs(self,start):
        return
    
    def djikstra(self,start,end):
        return
    
    def floyd_warshall(self):
        return 
    
    def bellman_ford(self, source):
        """
        Finds shortest paths from source to all vertices using Bellman-Ford algorithm.
        Detects negative cycles if present.

        Args:
            source (hashable): The starting vertex.

        Returns:
            dict: Shortest distances from source to each vertex.

        Raises:
            ValueError: If a negative cycle is detected.
        """
        numVertices, numEdges = self.size()
        
        # Step 1: Initialize distances
        dist = {node: float('inf') for node in self.graph}
        dist[source] = 0

        # Step 2: Relax all edges (V-1 times)
        for _ in range(numVertices - 1):  
            for vertex in self.graph:  
                for neighbour in self.graph[vertex]:  
                    weight = self.graph[vertex][neighbour] if self.weighted else 1
                    if dist[vertex] + weight < dist[neighbour]:
                        dist[neighbour] = dist[vertex] + weight

        # Step 3: Negative Cycle Detection 
        for vertex in self.graph:
            for neighbour in self.graph[vertex]:
                weight = self.graph[vertex][neighbour] if self.weighted else 1
                if dist[vertex] + weight < dist[neighbour]:  
                    raise ValueError(f"There is a negative cycle involving edge ({vertex}, {neighbour})")

        return dist

    def bellman_ford(self, source, destination):
        """
        Computes the shortest path from the source to the destination 
        using the Bellman-Ford algorithm.

        Args:
            source (hashable): The starting vertex.
            destination (hashable): The target vertex.

        Returns:
            float: The shortest distance from `source` to `destination`.
                Returns `float('inf')` if `destination` is not reachable.
        """
        distances = self.bellman_ford(source)  # Compute shortest paths from source
        return distances[destination]  # Return the shortest distance (or float('inf') if unreachable)




                    






        




    



