import pygimli as pg
import numpy as np
from settings import fpm

pd = pg.load("paraDomain_1.bms")
resinv = np.loadtxt("res_conventional.dat")
vest = np.loadtxt("vel_conventional.dat")

fae, fie, fwe, maske = fpm.all(resinv, vest)
print(np.min(fwe), np.max(fwe))
np.savez("conventional.npz", vel=np.array(vest), rho=np.array(resinv), fa=fae,
         fi=fie, fw=fwe, mask=maske)
