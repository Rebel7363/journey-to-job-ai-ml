"""
Module 02: Data Structures
Topic: Singly & Doubly Linked Lists, In-Place Reversal, and Floyd's Cycle Detection
"""


class Node:
    """Represents an individual node in a Singly Linked List."""

    def __init__(self, value, next_node=None):
        self.value = value
        self.next = next_node


class SinglyLinkedList:
    """Production-style Singly Linked List with common interview operations."""

    def __init__(self):
        self.head = None
        self._length = 0

    def append(self, value):
        """Appends element at the end: O(n) Time without tail pointer."""
        new_node = Node(value)
        if not self.head:
            self.head = new_node
        else:
            curr = self.head
            while curr.next:
                curr = curr.next
            curr.next = new_node
        self._length += 1

    def reverse(self):
        """In-place pointer reversal: O(n) Time | O(1) Auxiliary Space."""
        prev = None
        curr = self.head

        while curr:
            next_temp = curr.next
            curr.next = prev
            prev = curr
            curr = next_temp

        self.head = prev

    def has_cycle(self) -> bool:
        """
        Floyd's Tortoise and Hare Algorithm.
        Detects if a loop exists in O(n) Time and O(1) Space.
        """
        slow = self.head
        fast = self.head

        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next
            if slow == fast:
                return True

        return False

    def to_list(self) -> list:
        """Helper to serialize linked list into Python list for display."""
        elements = []
        curr = self.head
        visited = set()

        while curr:
            if id(curr) in visited:
                elements.append(f"...cycle back to ({curr.value})")
                break
            visited.add(id(curr))
            elements.append(curr.value)
            curr = curr.next

        return elements

    def __len__(self):
        return self._length


def demonstrate_doubly_linked_list():
    """Doubly Linked List node connection overview (used in LRU Cache / OS kernels)."""

    class DNode:
        def __init__(self, value):
            self.value = value
            self.prev = None
            self.next = None

    # Connect: [A] <-> [B] <-> [C]
    node_a = DNode("Input_Layer")
    node_b = DNode("Hidden_Layer")
    node_c = DNode("Output_Layer")

    node_a.next = node_b
    node_b.prev = node_a
    node_b.next = node_c
    node_c.prev = node_b

    # Forward traversal
    forward = []
    curr = node_a
    while curr:
        forward.append(curr.value)
        curr = curr.next

    # Backward traversal
    backward = []
    curr = node_c
    while curr:
        backward.append(curr.value)
        curr = curr.prev

    print("=" * 65)
    print("3. DOUBLY LINKED LIST (BIDIRECTIONAL TRAVERSAL)")
    print("=" * 65)
    print(f"Forward:  {' -> '.join(forward)}")
    print(f"Backward: {' -> '.join(backward)}")


if __name__ == "__main__":
    print("=" * 65)
    print("1. SINGLY LINKED LIST & IN-PLACE REVERSAL")
    print("=" * 65)
    ll = SinglyLinkedList()
    for item in [10, 20, 30, 40, 50]:
        ll.append(item)

    print(f"Original: {ll.to_list()}")
    ll.reverse()
    print(f"Reversed: {ll.to_list()}")

    print("\n" + "=" * 65)
    print("2. FLOYD'S CYCLE DETECTION (TORTOISE & HARE)")
    print("=" * 65)
    print(f"Cycle detected in linear list? {ll.has_cycle()}")

    # Inject deliberate cycle: last node -> second node
    curr = ll.head
    while curr.next:
        curr = curr.next
    curr.next = ll.head.next  # Loop created

    print(f"Cycle detected after injection? {ll.has_cycle()}")
    print(f"Representation: {ll.to_list()}")

    print()
    demonstrate_doubly_linked_list()