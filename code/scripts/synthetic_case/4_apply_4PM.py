import sys
import numpy as np
import pygimli as pg
from fpinv import FourPhaseModel

mesh = pg.load("mesh.bms")

if len(sys.argv) > 1:
    pd = pg.load("paraDomain_2.bms")
    resinv = np.loadtxt("res_conventional_2.dat")
    vest = np.loadtxt("vel_conventional_2.dat")
    scenario = "Fig2"
    pg.boxprint(scenario)
    phi = 0.3  # Porosity assumed to calculate fi, fa, fw with 4PM
else:
    pd = pg.load("paraDomain_1.bms")
    resinv = np.loadtxt("res_conventional_1.dat")
    vest = np.loadtxt("vel_conventional_1.dat")
    scenario = "Fig1"
    pg.boxprint(scenario)
    frtrue = np.load("true_model.npz")["fr"]

    phi = []
    for cell in pd.cells():
        idx = mesh.findCell(cell.center()).id()
        phi.append(1 - frtrue[idx])
    phi = np.array(phi)

# Save some stuff
fpm = FourPhaseModel(phi=phi)
fae, fie, fwe, maske = fpm.all(resinv, vest)
print(np.min(fwe), np.max(fwe))
np.savez("conventional_%s.npz" % scenario, vel=np.array(vest),
         rho=np.array(resinv), fa=fae, fi=fie, fw=fwe, mask=maske)
