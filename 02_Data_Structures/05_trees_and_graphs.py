"""
Module 02: Data Structures
Topic: Binary Search Trees (BST), Traversals, and Graph BFS / DFS
"""

from collections import deque


# =====================================================================
# 1. BINARY SEARCH TREE (BST)
# =====================================================================
class TreeNode:
    """Represents a node in a Binary Tree."""

    def __init__(self, val: int):
        self.val = val
        self.left = None
        self.right = None


class BinarySearchTree:
    """BST supporting insertion, search, and depth-first traversals."""

    def __init__(self):
        self.root = None

    def insert(self, val: int):
        """Inserts a value maintaining BST invariants (Left < Root < Right)."""
        if not self.root:
            self.root = TreeNode(val)
            return

        curr = self.root
        while True:
            if val < curr.val:
                if curr.left is None:
                    curr.left = TreeNode(val)
                    break
                curr = curr.left
            elif val > curr.val:
                if curr.right is None:
                    curr.right = TreeNode(val)
                    break
                curr = curr.right
            else:
                break  # Duplicate values ignored

    def search(self, val: int) -> bool:
        """Searches value in O(log n) average time, O(n) worst case."""
        curr = self.root
        while curr:
            if val == curr.val:
                return True
            curr = curr.left if val < curr.val else curr.right
        return False

    def inorder_traversal(self) -> list[int]:
        """In-order traversal (Left -> Root -> Right): Returns sorted sequence."""
        res = []

        def _dfs(node):
            if not node:
                return
            _dfs(node.left)
            res.append(node.val)
            _dfs(node.right)

        _dfs(self.root)
        return res


# =====================================================================
# 2. GRAPH REPRESENTATION & TRAVERSALS (BFS & DFS)
# =====================================================================
class Graph:
    """Graph implementation using Adjacency List."""

    def __init__(self):
        self.adj_list: dict[str, list[str]] = {}

    def add_edge(self, u: str, v: str, bidirectional: bool = True):
        self.adj_list.setdefault(u, []).append(v)
        self.adj_list.setdefault(v, [])
        if bidirectional:
            self.adj_list[v].append(u)

    def bfs(self, start_node: str) -> list[str]:
        """Breadth-First Search (Level-Order / Shortest path in unweighted graphs)."""
        visited = set([start_node])
        queue = deque([start_node])
        traversal_order = []

        while queue:
            node = queue.popleft()
            traversal_order.append(node)

            for neighbor in self.adj_list.get(node, []):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)

        return traversal_order

    def dfs(self, start_node: str) -> list[str]:
        """Depth-First Search (Explores as deep as possible before backtracking)."""
        visited = set()
        traversal_order = []

        def _explore(node):
            visited.add(node)
            traversal_order.append(node)
            for neighbor in self.adj_list.get(node, []):
                if neighbor not in visited:
                    _explore(neighbor)

        _explore(start_node)
        return traversal_order


if __name__ == "__main__":
    print("=" * 65)
    print("1. BINARY SEARCH TREE (BST) OPERATIONS")
    print("=" * 65)
    bst = BinarySearchTree()
    elements = [50, 30, 70, 20, 40, 60, 80]
    for el in elements:
        bst.insert(el)

    print(f"Inserted:         {elements}")
    print(f"In-order Traversal (Sorted): {bst.inorder_traversal()}")
    print(f"Search 40:        {bst.search(40)}")
    print(f"Search 95:        {bst.search(95)}")

    print("\n" + "=" * 65)
    print("2. GRAPH TRAVERSALS: BFS VS DFS")
    print("=" * 65)
    g = Graph()
    # Knowledge / neural network graph mockup
    g.add_edge("Input", "Dense_1")
    g.add_edge("Input", "Conv_1")
    g.add_edge("Dense_1", "Dropout")
    g.add_edge("Conv_1", "MaxPool")
    g.add_edge("Dropout", "Output")
    g.add_edge("MaxPool", "Output")

    print(f"BFS Order (Breadth-First): {g.bfs('Input')}")
    print(f"DFS Order (Depth-First):   {g.dfs('Input')}")