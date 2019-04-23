import scipy.io as sio
import graph_tool

mutag = sio.loadmat('DD.mat')

data = mutag['DD']
label = mutag['ldd']
f_text = open("DD/text.txt","w")
for i in range(len(label)):
    g = graph_tool.Graph()
    g.set_directed(False)

    node = list(data['nl'][0,i].item(0))[0]
    
    edge = list(data['al'][0,i])

    g.add_vertex(len(node))

    vprop_name = g.new_vertex_property("string") 
    g.vp.name = vprop_name
    vprop_value = g.new_vertex_property("int")
    g.vp.label = vprop_value
    for j in range(len(node)):
        g.vp.name[g.vertex(j)] = "n"+str(j)
        g.vp.label[g.vertex(j)] = node[j].item(0)

    for j in range(int(len(edge))):
	node_edge = list(edge[j][0][0])
	#print(node_edge)
	for k in range(len(node_edge)):
	    if(j < node_edge[k]):
               g.add_edge(g.vertex(j),g.vertex(node_edge[k]-1))
 
  #  eprop = g.new_edge_property("int")
  #  g.edge_properties['weight'] = eprop
  #  for j in range(len(edge)):
  #      g.edge_index(j).weight = edge[j][2]


    file_name = "dd_" + str(i)
    g.save("DD/"+ file_name + ".xml.gz")
    f_text.write(file_name + ".xml.gz" + " " + str(label[i].item(0)) + '\n')
print(g)
#f_text.close()

