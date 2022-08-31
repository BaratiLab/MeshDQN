from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from dolfin import *
import h5py

#labels = pd.read_csv("./data/labels_large.csv")
#labels = pd.read_csv("./labels.csv")
#for label in labels['Airfoils']:
#    print(label)
#    vel = np.loadtxt("./velocity/" + label + "_vel", dtype=np.double)
#    edge = np.loadtxt("./edges/" + label + "_edges", dtype=np.double)
#    fig, ax = plt.subplots()
#    print(edge)
#    ax.scatter(edge[:,0], edge[:,1])
#    plt.show()
#    break

mesh = Mesh()
with XDMFFile(MPI.COMM_WORLD, "ag11.xdmf", "r") as xdmf:
    #mesh = xdmf.read_mesh(name="GRID")
    pass
#with XDMFFile("./test_geo.xdmf") as outfile:
#    print(outfile.write(mesh))
    #outfile.read(mesh, True)
    #raise
#mesh = Mesh("./geometry.xml")
    
#plot(mesh)
#plt.show()
