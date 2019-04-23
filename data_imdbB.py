import graph_tool

class Node:
      def __init__(self,node,graph,atr = 0):
	self.node = node
	self.graph = graph
	self.atr = atr

pnode_file = open("IMDB-MULTI/IMDB-MULTI_graph_indicator.txt","r")

pedge_file = open("REDDIT-BINARY/REDDIT-BINARY_A.txt",'r')

label_file = open("REDDIT-BINARY/REDDIT-BINARY_graph_labels.txt","r")

#node_atr = open("IMDB-BINARY/PTC_MR_node_labels.txt","r")

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
     #node_label = node_atr.readline().strip('\n')    
     if int(node_line) == k:
	Nodes[i] = Node(n,k-1)
	n = n + 1
     else:
	Graph[k-1].add_vertex(n)
	vprop_value = Graph[k-1].new_vertex_property("int")
	Graph[k-1].vp.label = vprop_value
	k = k + 1
	n = 0
	Nodes[i] = Node(n,k-1)

Graph[k-1].add_vertex(n)
vprop_value = Graph[k-1].new_vertex_property("int")
Graph[k-1].vp.label = vprop_value
print("hello")
for i in range(len(Nodes)):    
    No = Nodes[i]
    Graph[No.graph].vp.label[Graph[No.graph].vertex(No.node)] = No.atr

for i,edge_line in enumerate(pedge_file):
    node1,node2 = edge_line.split(', ')
    Node1 = Nodes[int(node1)-1]
    Node2 = Nodes[int(node2)-1]
    if Node1.node <= Node2.node:
       Node1.atr += 1
       Node2.atr += 1
       Graph[Node1.graph].add_edge(Graph[Node1.graph].vertex(Node1.node),Graph[Node1.graph].vertex(Node2.node))

for k in range(len(Graph)):
    vprop_value = Graph[k].new_vertex_property("int")
    Graph[k].vp.label = vprop_value

for i in range(len(Nodes)):    
    No = Nodes[i]
    Graph[No.graph].vp.label[Graph[No.graph].vertex(No.node)] = No.atr

f_text = open("Reddit-B/text.txt","w")
for i in range(len(Graph)):
    file_name = "reddit_b_" + str(i)
    Graph[i].save("Reddit-B/"+ file_name + ".xml.gz")
    f_text.write(file_name + ".xml.gz" + " " + label[i])   
print(Graph[0])
print(Graph[len(Graph)-1])
	
