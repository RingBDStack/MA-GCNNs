import scipy.io as sio
import graph_tool

mutag = sio.loadmat('NCI1.mat')
data = mutag['NCI1']
label = mutag['lnci1']
f_text = open("NCI1/text.txt","w")
for i in range(len(label)):
    g = graph_tool.Graph()
    g.set_directed(False)

    node = list(data['nl'][0,i].item(0))[0]
    
    edge = list(data['el'][0,i])[0].item(0)[0]
   # print(len(edge))
  #  print(type(edge))

    g.add_vertex(len(node))

    vprop_name = g.new_vertex_property("string") 
    g.vp.name = vprop_name
    vprop_value = g.new_vertex_property("int")
    g.vp.label = vprop_value
    for j in range(len(node)):
        g.vp.name[g.vertex(j)] = "n"+str(j)
        g.vp.label[g.vertex(j)] = node[j].item(0)

    for j in range(int(len(edge))):
	if len(edge[j][0]) != 0:
	   node_edge = list(edge[j][0][0])
	   #print(node_edge)
	   for k in range(len(node_edge)):
               g.add_edge(g.vertex(j),g.vertex(node_edge[k]-1))
 
  #  eprop = g.new_edge_property("int")
  #  g.edge_properties['weight'] = eprop
  #  for j in range(len(edge)):
  #      g.edge_index(j).weight = edge[j][2]

  #  print(g)
    file_name = "nci1_" + str(i)
    g.save("NCI1/"+ file_name + ".xml.gz")
    f_text.write(file_name + ".xml.gz" + " " + str(label[i].item(0)) + '\n')
print(g)
#f_text.close()

