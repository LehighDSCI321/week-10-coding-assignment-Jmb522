
"""Week 10 coding assignment. We are implementing
a TraversableDigraph class which inherits from
a SortableDigraph class and DAG which inherits from
TraversableDigraph. This will augment SortableDigraph
with two additional methods, BFS and DFS."""

try:
    import graphviz
    GRAPHVIZ_AVAILABLE = True
except ImportError:
    GRAPHVIZ_AVAILABLE = False
try:
    from bokeh.plotting import figure, show, output_file
    BOKEH_AVAILABLE = True
except ImportError:
    BOKEH_AVAILABLE = False

class VersatileDigraph():
    """A versatile and directed graph implementation"""

    #first method initialize
    def __init__(self):
        """initialize an empty directed graph"""
        self.nodes = {} #dict w/node and value
        self.edges = [] #edges with source target and edge name-list of tuples
    def add_node(self, node, value=0):
        """this will add anode with a given value
        to this graph"""
        if not isinstance(node, str):
            raise TypeError(f"node must be a string, got {type(node).__name__}")
        if not isinstance(value, (int, float)):
            raise TypeError(f"node value must be numeric, got {type(value).__name__}")
        if node in self.nodes:
            raise ValueError(f"node '{node}' already exists in the graph")
        self.nodes[node] = value

    def get_nodes(self):
        """returns a dictionary of nodes in the graph.
        
        Returns:
            dict: dictionary with node names as keys and values as values
        """
        return self.nodes

    def add_edge(self, source, target, edge_name = None, edge_weight = 1):
        """adding an edge from the source to target
        with the option for an edge name and also weight"""
        if not isinstance(source, str):
            raise TypeError(f"source node must be a string, got {type(source).__name__}")
        if not isinstance(target, str):
            raise TypeError(f"Target node must be a string, got {type(target).__name__}")
        if edge_name is not None and not isinstance(edge_name, str):
            raise TypeError(f"edge name must be a string, got {type(edge_name).__name__}")
        if not isinstance(edge_weight, (int, float)):
            raise TypeError(f"edge weight must be numeric, got {type(edge_weight).__name__}")
        if edge_weight < 0:
            raise ValueError("Edge weight cannot be negative")
        if source not in self.nodes:
            raise KeyError(f"source node '{source}' does not exist in graph")
        if target not in self.nodes:
            raise KeyError(f"Target node '{target}' does not exist in the graph")
        #now we can check for duplicates
        if edge_name is not None:
            exist_names = [en for sn, tn, en, ew in self.edges if sn == source and en is not None]
            if edge_name in exist_names:
                raise ValueError(f"edgename '{edge_name}' exists for source node '{source}'")

        self.edges.append((source, target, edge_name, edge_weight))

    def predecessors(self, node):
        """Predeccessors will be for if we are
        given a node, return a list of the nodes
        that immediately come before that node.
        The args will be the node, which is the target
        for the predeccessors. The return will be a
        list, nodes that have edges that lead to
        the given node. We will essentially look at
        each edge and collect the source nodes that
        point to our target one"""
        if not isinstance(node, str):
            raise TypeError(f"node must be a string, got {type(node).__name__}")
        if node not in self.nodes:
            raise KeyError(f"node '{node}' does not exist in the graph")
        #had to use shorter vairable names. sn= source node
        #tn = target node. en= edge name.
        #find where the target matches our node
        result = [sn for sn, tn, en, ew in self.edges if tn == node]
        return result

    def successors(self, node):
        """finds all nodes that a given nodes points
        to. basically the reverse of predeccessors almost."""
        #look through all edges and collect target nodes
        #return where source matches our node
        if not isinstance(node, str):
            raise TypeError(f"node must be a string, got {type(node).__name__}")
        if node not in self.nodes:
            raise KeyError(f"Node '{node}' does not exist in the graph")
        result = [tn for sn, tn, en, ew in self.edges if sn == node]
        return result

    def successor_on_edge(self, node, edge_name):
        """finds the node that the node given points to using
        a specific edge name. It will look throguh all the edges
        to find the one with a matching source node and
        edge name."""
        if not isinstance(node, str):
            raise TypeError(f"node must be a string, got {type(node).__name__}")
        if not isinstance(edge_name, str):
            raise TypeError(f"edge name must be a string, got {type(edge_name).__name__}")
        if node not in self.nodes:
            raise KeyError(f"node '{node}' does not exist in graph")
        matching_targs = [tn for sn, tn, name, ew in self.edges if sn == node and name == edge_name]
        #we will return the first match or none if there is none
        #search for edges that start from our node
        #and also have the edge name we are looking for
        if len(matching_targs) > 0:
            return matching_targs[0]
        return None

    def in_degree(self, node):
        """We will use this to count how many edges
        point to the node."""
        if not isinstance(node, str):
            raise TypeError(f"node must be a string, got {type(node).__name__}")
        if node not in self.nodes:
            raise KeyError(f"node '{node}' does not exist in the graph")
        #count all the edges where out node is the target
        count = sum(1 for sn, tn, en, ew in self.edges if tn == node)
        return count

    def out_degree(self, node):
        """This will countr how many edges point away
        from the node. basicall the same concept as in_degree
        but using source instead of target."""
        if not isinstance(node, str):
            raise TypeError(f"node must be a string, got {type(node).__name__}")
        if node not in self.nodes:
            raise KeyError(f"node '{node}' does not exist in the graph")
        #count all the edges where our node is the source
        count = sum(1 for sn, tn, en, ew in self.edges if sn == node)
        return count

    def get_edge_weight(self, source, target):
        """gets the weight of the edge"""
        if not isinstance(source, str):
            raise TypeError("source node must be a string")
        if not isinstance(target, str):
            raise TypeError("target node must be a string")
        if source not in self.nodes:
            raise KeyError("source node does not exist in the graph")
        if target not in self.nodes:
            raise KeyError('target node does not exist in the graph')
        for sn, tn, _en, ew in self.edges:
            if sn == source and tn == target:
                return ew
        raise KeyError(f'edge from {source} to {target} does not exist')

    def get_node_value(self, node):
        """getter for the value of a node"""
        if not isinstance(node, str):
            raise TypeError("node must be a string")
        if node not in self.nodes:
            raise KeyError("node does not exist in the graph")
        return self.nodes[node]

    def plot_graph(self, filename="graph_output"):
        """making a plot of the object using graphviz"""
        #making sure we actually have nodes to plot
        if not self.nodes:
            raise ValueError("cannot plot empty graph if no nodes exist")

        if not GRAPHVIZ_AVAILABLE:
            raise ImportError("need to install graphviz package")
        #creating the graph here
        dot = graphviz.Digraph(comment = 'VersatileDigraph')
        dot.attr(rankdir='TB', size = '10,8') #top to bottom
        dot.attr('node', shape= 'ellipse', style='filled', fillcolor='lightblue')
        #makes all the nodes look like ovals like in the picture but
        #i filled them with blue to make it look nicer
        dot.attr('edge', fontsize='12')
        #makes the text bigger on edges

        #now we add the nodes with their values with a loop
        for node, value in self.nodes.items():
            dot.node(node, f"{node}:{value}")
            #for example it will add "Allentown:66"

        #add the edges now with a loop
        for sn, tn, en, ew in self.edges:
            if en is not None:
                #this draws an arrow from the source node
                #to the target node and labels it
                #showing the route and distance
                dot.edge(sn, tn, label = f"{en}({ew})")
            else:
                #if theres no name we just show the number
                dot.edge(sn, tn, label = str(ew))

        #now we can save and render it
        dot.render(filename, format='png', view=False, cleanup=True)
        print(f"graph saved as {filename}.png")
        return dot

    def plot_edge_weights(self):
        """makes a bar graph that shows the
        weight of each edge using bokeh"""
        if not self.edges:
            raise ValueError("Cannot plot edge weights if not edges exist")
        if not BOKEH_AVAILABLE:
            raise ImportError("must have bokeh installed")

        #setting up the data now for the chart
        edge_labels = []
        weights = []

        for sn, tn, _en, ew in self.edges:
            edge_label = f"{sn} to {tn}"
            edge_labels.append(edge_label)
            weights.append(ew)
        output_file("edge_weights.html")

        #creating the figure with labels and size
        p = figure(
            x_range= edge_labels,
            title="Edge Weights in Graph",
            x_axis_label= 'Edges',
            y_axis_label = 'Weight',
            width = 800,
            height = 400,
        )
        #adding bars now
        p.vbar(x=edge_labels, top = weights, width = 0.7,
               fill_color = 'red', line_color='blue', alpha = 0.8)
        #showing the chart
        show(p)
        print("edge weights chart saved as edge_weights.html")
        return p

class BinaryGraph(VersatileDigraph):
    """A binary tree implementation that
    will extend the versatile digraph"""

    def __init__(self):
        """initializes a binary tree
        with a automatic root node"""
        super().__init__()
        #this will automatically add the
        #root node with value 0
        self.add_node("Root", 0)

    def add_node_left(self, child_id, child_value, parent_id=None):
        """adds a left child to the parent node.
        arguments:
            child_id- id of the new child
            child_value- value of new child
            parent_id - id of the parent-
                deaults to root if none"""

        if parent_id is None:
            parent_id = "Root"

        #checks if parent exists
        if parent_id not in self.nodes:
            raise KeyError(f"Parent node '{parent_id}' does not exist")

        #check if left child already exists
        exist_left = self.successor_on_edge(parent_id, "left")
        if exist_left is not None:
            raise ValueError(f"Node '{parent_id}' already has the left child")

        #adds the child node and creats the edge
        self.add_node(child_id, child_value)
        self.add_edge(parent_id, child_id, edge_name="left", edge_weight=1)

    def add_node_right(self, child_id, child_value, parent_id=None):
        """adds a right child to the parent node.
        arguments:
            child_id- id of the new child
            child_value- value of new child
            parent_id - id of the parent-
                deaults to root if none"""

        if parent_id is None:
            parent_id = "Root"

        #checks if parent exists
        if parent_id not in self.nodes:
            raise KeyError(f"Parent node '{parent_id}' does not exist")

        #check if right child already exists
        exist_right = self.successor_on_edge(parent_id, "right")
        if exist_right is not None:
            raise ValueError(f"Node '{parent_id}' already has the right child")

        #adds the child node and creats the edge
        self.add_node(child_id, child_value)
        self.add_edge(parent_id, child_id, edge_name="right", edge_weight=1)

    def get_node_left(self, parent_id):
        """getter for the id of the left child
        arguments:
        parent id- id of the parent node
        returns the id of the left child or none(
        if it does not exist)"""

        #checks if parent node exists
        if parent_id not in self.nodes:
            raise KeyError(f"node '{parent_id}' does not exist")

        #we use the parent successor on edge method now
        return self.successor_on_edge(parent_id, "left")

    def get_node_right(self, parent_id):
        """getter for the id of the right child
        arguments:
        parent id- id of the parent node
        returns the id of the right child or none(
        if it does not exist)"""

        #checks if parent node exists
        if parent_id not in self.nodes:
            raise KeyError(f"node '{parent_id}' does not exist")

        #we use the parent successor on edge method now
        return self.successor_on_edge(parent_id, "right")

class SortingTree(BinaryGraph):
    """A binary search tree implementation
    that inherits from binary graph"""

    def __init__(self, root_value=None):
        """initializes a sorting tree with
        an optional root value.

        args:
            root_value- optional value for root node
        """
        super().__init__()
        if root_value is not None:
            if not isinstance(root_value, (int, float)):
                raise TypeError(f"value has to be numerical, got {type(root_value).__name__}")
            self.nodes["Root"] = root_value

    def insert(self, value, node=None):
        """inserts a numerical value into
        the binary search using recursion

        Args:
            value- numerical value to insert
            node- current node- defaults to root
        """
        if not isinstance(value, (int, float)):
            raise TypeError(f"value has to be numeric, got {type(value).__name__}")

        #default paremter
        if node is None:
            node = "Root"

        current_value = self.get_node_value(node)

        if value < current_value:
            left_child = self.get_node_left(node)
            if left_child is None:
                #this is the base case
                new_node_id = f"node_{value}_{id(value)}"
                self.add_node_left(new_node_id, value, node)
            else:
                #recursive case
                self.insert(value, left_child)
        else:
            right_child = self.get_node_right(node)
            if right_child is None:
                #this is the base case
                new_node_id = f"node_{value}_{id(value)}"
                self.add_node_right(new_node_id, value, node)
            else:
                #recursive case
                self.insert(value, right_child)

    def traverse(self, node=None):
        """traverses the tree and prints the
        values in sorted order.

        args:
            node- current node-defaults to root
        """

        #default paremeter
        if node is None:
            node = "Root"

        #recursive case
        left_child = self.get_node_left(node)
        if left_child is not None:
            self.traverse(left_child)
        print(self.get_node_value(node), end=" ")

        #recursive case for right
        right_child = self.get_node_right(node)
        if right_child is not None:
            self.traverse(right_child)

class SortableDigraph(VersatileDigraph):
    """A sortable directed graph that
    inherits from versatile digraph"""

    def __init__(self):
        """initializes an empty sortable
        directed graph"""
        super().__init__()
        self._sortable = True

    def top_sort(self):
        """performs topological sorting
        using Kahn's algorithm. 

        Returns:
        list: a list of nodes in topologically
        sorted order.Returns emppy list if
        graph is empty."""

        #we have to handle an empty graph case
        if not self.nodes:
            return []

        #count in degrees
        in_degree_map = {node: self.in_degree(node) for node in self.nodes}
        #find stating nodes with zero in degree
        zero_in_degree = [node for node in self.nodes if in_degree_map[node] == 0]
        #list of results
        sorted_nodes = []

        #process the nodes
        while zero_in_degree:
            current_node = zero_in_degree.pop()
            sorted_nodes.append(current_node)

            for successor in self.successors(current_node):
                in_degree_map[successor] -= 1
                if in_degree_map[successor] == 0:
                    zero_in_degree.append(successor)

        if len(sorted_nodes) != len(self.nodes):
            raise ValueError("Graph has at least one cycle; topological sort not possible")

        return sorted_nodes

from collections import deque

class TraversableDigraph(SortableDigraph):
    """A traversable directed graph that
    inherits from sortable digraph.
    Adds a DFS and a BFS capability."""

    def __init__(self):
        """initializes an empty traversable
        directed graph"""
        super().__init__()

    def dfs(self, start):
        """performs DFS traversal from the start node.

        Args
            start: the node to start traversal from
        Yields:
            nodes in depth first order
        """
        if not isinstance(start, str):
            raise TypeError(f"start node must be a string, got {type(start).__name__}")
        if start not in self.nodes:
            raise KeyError(f"start node '{start}' does not exist in the graph")

        visited = set([start]) #mark start as visited but do not yield

        def dfs_visit(node):
            """Recursive helper function for DFS taversal"""
            for successor in self.successors(node):
                if successor not in visited:
                    visited.add(successor)
                    yield successor
                    yield from dfs_visit(successor)

        yield from dfs_visit(start)

    def bfs(self, start):
        """performs BFS traversal from the start node.

        Args
            start: the node to start traversal from
        Yields:
            nodes in breadth first order
        """
        if not isinstance(start, str):
            raise TypeError(f"start node must be a string, got {type(start).__name__}")
        if start not in self.nodes:
            raise KeyError(f"start node '{start}' does not exist in the graph")

        visited = set([start]) #mark start as visited but do not yield
        queue = deque(self.successors(start)) #add all of start's successors to queue

        while queue:
            current_node = queue.popleft() #get first node from front of queue
            if current_node not in visited:
                visited.add(current_node)
                yield current_node
                #add all unvisited successors to the queue
                for successor in self.successors(current_node):
                    if successor not in visited:
                        queue.append(successor)

class DAG(TraversableDigraph):
    """A directed acyclic graph that
    inherits from traversable digraph.
    Overrides add_edge to prevent cycles."""

    def __init__(self):
        """initializes an empty DAG"""
        super().__init__()

if __name__ == "__main__":
    #testing the sortable digraph
    #Test 1: empty graph
    sortable_graph = SortableDigraph()
    print(sortable_graph.top_sort()) #should print []
    #test 2: simple DAG
    sortable_graph2 = SortableDigraph()
    #adding nodes
    sortable_graph2.add_node("A", 1)
    sortable_graph2.add_node("B", 2)
    sortable_graph2.add_node("C", 3)
    sortable_graph2.add_node("D", 4)
    sortable_graph2.add_node("E", 5)
    #adding edges
    sortable_graph2.add_edge("A", "B")
    sortable_graph2.add_edge("A", "C")
    sortable_graph2.add_edge("B", "C")
    sortable_graph2.add_edge("B", "D")
    sortable_graph2.add_edge("C", "D")
    print(sortable_graph2.top_sort())
    #should print a valid topological order like
    #['A', 'B', 'C', 'D', 'E']

    #Test 3: clothing graph
    sortable_graph3 = SortableDigraph()
    nodes = ['shirt', 'tie', 'jacket', 'belt', 'pants', 'socks', 'shoes', 'vest']
    for n in nodes:
        sortable_graph3.add_node(n)
    sortable_graph3.add_edge('shirt', 'pants')
    sortable_graph3.add_edge('shirt', 'vest')
    sortable_graph3.add_edge('shirt', 'tie')
    sortable_graph3.add_edge('shirt', 'jacket')
    sortable_graph3.add_edge('vest', 'jacket')
    sortable_graph3.add_edge('socks', 'shoes')
    sortable_graph3.add_edge('pants', 'belt')
    sortable_graph3.add_edge('tie', 'jacket')
    sortable_graph3.add_edge('pants', 'shoes')
    sortable_graph3.add_edge('belt', 'jacket')
    result1 = sortable_graph3.top_sort()
    print(result1)

if __name__ == "__main__":
    tree = BinaryGraph()
    #create a new tree with root value of 8
    tree.nodes["Root"] = 8
    #first level- add childs to root
    tree.add_node_left("71", 71)
    tree.add_node_right("41", 41)
    #second level
        #children for 71
    tree.add_node_left("31a", 31, "71")
    tree.add_node_right("10", 10, "71")
        #children for 41
    tree.add_node_left("11", 11, "41")
    tree.add_node_right("16", 16, "41")
    #third level
        #children for 31a
    tree.add_node_left("46", 46, "31a")
    tree.add_node_right("51", 51, "31a")
        #children for 10
    tree.add_node_left("31b", 31, "10")
    tree.add_node_right("21", 21, "10")
        #child for 11
    tree.add_node_left("13", 13, "11")

    #make the visualization
    tree.plot_graph("binary_tree")
