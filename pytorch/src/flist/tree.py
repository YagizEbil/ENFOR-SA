import math
from anytree import Node
from anytree import Node, NodeMixin, RenderTree
from anytree.iterators import LevelOrderIter
import src.utils.utils as u


class FaultNode(Node):
    def __init__(self, name, parent=None, level=0, **kwargs):
        super().__init__(name=name, parent=parent, **kwargs)
        self.visited = False
        self.is_critical = False
        self.input_indices = []
        self.fault_list = [] # the nodes fault list
        self.tag = -1 # for leaf-nodes only. keep the fault tag
        self.color = WHITE


def build_k_ary_tree(k, h, tree_id):
    root = FaultNode(f"root_{tree_id}")

    def _build(node, level):
        if level == h:
            return
        for i in range(k):
            child = FaultNode(f"{node.name}_{i}", parent=node)
            _build(child, level + 1)

    _build(root, 0)
    return root


def assign_item_intervals(root, start_index, k, h, num_items):
    """
    Assign global item intervals to all nodes in the tree.
    """

    leaf_capacity = k ** h

    def _assign(node, level, start):
        # Number of leaves under this node
        subtree_leaves = k ** (h - level)
        end = start + subtree_leaves - 1

        # Clip to actual item count
        clipped_end = min(end, num_items - 1)

        if start > num_items - 1:
            node.interval = None
        else:
            node.interval = (start, clipped_end)

        if not node.children:
            # Leaf: attach item if valid
            if start < num_items:
                node.item = start
            else:
                node.item = None
            return

        child_start = start
        for child in node.children:
            _assign(child, level + 1, child_start)
            child_start += subtree_leaves // k

    _assign(root, level=0, start=start_index)


def build_forest_with_intervals(items, k, h):
    N = len(items)
    C = k ** h
    num_trees = math.ceil(N / C)

    forest = []

    for t in range(num_trees):
        root = build_k_ary_tree(k, h, t)

        assign_item_intervals(
            root=root,
            start_index=t * C,
            k=k,
            h=h,
            num_items=N
        )
 
        forest.append(root)

    return forest

  
"""
tree color scheme after injection

non processed nodes (not visited): white
processed nodes without failures: yellow

processed nodes with failures:
    Non-leaf nodes: orange
    Leaf nodes: red
"""


WHITE = u.Co['fg'][231]
YELLOW = u.Co['fg'][226]
ORANGE = u.Co['fg'][208]
RED = u.Co['fg'][160]


def paint_tree(node):
    if node.is_leaf:
        if not node.visited:
            node.color = WHITE
        else:
            node.color = RED if node.is_critical else YELLOW
    else:
        if not node.visited:
            node.color = WHITE
        else:
            node.color = ORANGE if node.is_critical else YELLOW
    
        for node in node.children:
            paint_tree(node)


def print_forest(forest):
    print(f"Forest size: {len(forest)}")

    for root in forest:
        print("\nTree", root.name)

        paint_tree(root)

        for pre, _, node in RenderTree(root):
            if node.is_leaf:
                print(f"{node.color}{pre}{node.name}: itvl={node.interval} tag={node.tag}{u.R}")
            else:
                print(f"{node.color}{pre}{node.name}: itvl={node.interval}")


def build_forest(k, h, C):
    items = list(range(C))
    forest = build_forest_with_intervals(items, k, h)
    return forest