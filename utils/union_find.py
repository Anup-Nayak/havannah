class UnionFind:
    def __init__(self, dim):
        self.parent = {}
        self.rank = {}
        self.corner_bits = {}
        self.edge_bits = {}
        self.dim = dim

    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x, y):
        rootX = self.find(x)
        rootY = self.find(y)

        if rootX != rootY:
            if self.rank[rootX] > self.rank[rootY]:
                self.parent[rootY] = rootX
                self.corner_bits[rootX] |= self.corner_bits[rootY]
                self.edge_bits[rootX] |= self.edge_bits[rootY]
            elif self.rank[rootX] < self.rank[rootY]:
                self.parent[rootX] = rootY
                self.corner_bits[rootY] |= self.corner_bits[rootX]
                self.edge_bits[rootY] |= self.edge_bits[rootX]
            else:
                self.parent[rootY] = rootX
                self.rank[rootX] += 1
                self.corner_bits[rootX] |= self.corner_bits[rootY]
                self.edge_bits[rootX] |= self.edge_bits[rootY]

    def add_stone(self, pos, is_corner, is_edge):
        if pos not in self.parent:
            self.parent[pos] = pos
            self.rank[pos] = 0
            self.corner_bits[pos] = is_corner
            self.edge_bits[pos] = is_edge

    def check_win_condition(self, pos):
        root = self.find(pos)
        corner_bits = self.corner_bits[root]
        edge_bits = self.edge_bits[root]
        
        # Check for Bridge (2 corners connected)
        if bin(corner_bits).count('1') >= 2:
            return "Bridge"

        # Check for Fork (3 edges connected)
        if bin(edge_bits).count('1') >= 3:
            return "Fork"

        return None

# Example to map positions to corners and edges
def is_corner(pos, dim):
    # Assume corner positions are predefined for a given dimension
    corners = [(0, 0), (0, dim - 1), (dim - 1, 0), (dim - 1, 2 * (dim - 1)), (2 * (dim - 1), dim - 1), (2 * (dim - 1), 2 * (dim - 1))]
    return pos in corners

def is_edge(pos, dim):
    # Example: Checking if a position is on the edge
    row, col = pos
    return row == 0 or row == 2 * (dim - 1) or col == 0 or col == 2 * (dim - 1)
