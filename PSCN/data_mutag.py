import scipy.io as sio
import graph_tool
import graph_tool.draw

mutag = sio.loadmat('MUTAG.mat')

data = mutag['MUTAG']
label = mutag['lmutag']
f_text = open("MUTAG/text.txt","w")
for i in range(len(label)):
    g = graph_tool.Graph()
    g.set_directed(False)

    node = list(data['nl'][0,i].item(0))[0]
    
    edge = list(data['el'][0,i])[0][0].item(0)[0]
 
#    print(edge)
#    print(edge[1])

    g.add_vertex(len(node))

    vprop_name = g.new_vertex_property("string") 
    g.vp.name = vprop_name
    vprop_value = g.new_vertex_property("int")
    g.vp.label = vprop_value
    for j in range(len(node)):
        g.vp.name[g.vertex(j)] = "n"+str(j)
        g.vp.label[g.vertex(j)] = node[j].item(0)
    
    eprop = g.new_edge_property("float")
    g.edge_properties['weight'] = eprop

    for j in range(int(len(edge))):
	node_edge = edge[j]
        g.add_edge(g.vertex(node_edge[0]-1),g.vertex(node_edge[1] -1))
        g.ep.weight[g.edge(node_edge[0]-1,node_edge[1] -1)] = node_edge[2]
	

 
  #  eprop = g.new_edge_property("int")
  #  g.edge_properties['weight'] = eprop
  #  for j in range(len(edge)):
  #      g.edge_index(j).weight = edge[j][2]


    file_name = "mutag_" + str(i)
    g.save("MUTAG/"+ file_name + ".xml.gz")
    f_text.write(file_name + ".xml.gz" + " " + str(label[i].item(0)) + '\n')
graph_tool.draw.graph_draw(g,size = 1)
#f_text.close()

