import graph_tool

class Node:
      def __init__(self,node,graph,atr):
	self.node = node
	self.graph = graph
	self.atr = atr

pnode_file = open("NCI109/NCI109_graph_indicator.txt","r")

pedge_file = open("NCI109/NCI109_A.txt",'r')

label_file = open("NCI109/NCI109_graph_labels.txt","r")

node_atr = open("NCI109/NCI109_node_labels.txt","r")

label = []
Graph = []

for lline in label_file:
    label.append(lline)
    g = graph_tool.Graph()
    g.set_directed(False)
    
    Graph.append(g)

l = len(label)

Nodes = {}
node_num = []
k = 1
n = 0
for i,node_line in enumerate(pnode_file):
     node_label = node_atr.readline().strip('\n')    
     if int(node_line) == k:
	Nodes[i] = Node(n,k-1,int(node_label)+1)
	n = n + 1
     else:
	Graph[k-1].add_vertex(n)
	vprop_value = Graph[k-1].new_vertex_property("int")
	Graph[k-1].vp.label = vprop_value
	k = k + 1
	n = 0
	Nodes[i] = Node(n,k-1,int(node_label)+1)

Graph[k-1].add_vertex(n)
vprop_value = Graph[k-1].new_vertex_property("int")
Graph[k-1].vp.label = vprop_value

for i in range(len(Nodes)):    
    No = Nodes[i]
    Graph[No.graph].vp.label[Graph[No.graph].vertex(No.node)] = No.atr

for i,edge_line in enumerate(pedge_file):
    node1,node2 = edge_line.split(',')
    Node1 = Nodes[int(node1)-1]
    Node2 = Nodes[int(node2)-1]

    Graph[Node1.graph].add_edge(Graph[Node1.graph].vertex(Node1.node),Graph[Node1.graph].vertex(Node2.node))


f_text = open("Nci109/text.txt","w")
for i in range(len(Graph)):
    file_name = "nci109_" + str(i)
    Graph[i].save("Nci109/"+ file_name + ".xml.gz")
    f_text.write(file_name + ".xml.gz" + " " + label[i])   
print(Graph[0])
print(Graph[len(Graph)-1])
	
