#############################################
# to find "invlib" in the main folder
import sys, os
sys.path.insert(0, os.path.abspath("../.."))
#############################################

import numpy as np

import pygimli as pg
from invlib import FourPhaseModel, JointMod, JointInv
from pybert.manager import ERTManager
from pygimli.physics import Refraction


# Load meshes and data
mesh = pg.load("mesh.bms")
true = np.load("true_model.npz")
conventional = np.load("conventional.npz")
sensors = np.load("sensors.npy")

veltrue, rhotrue, fatrue, fitrue, fwtrue = true["vel"], true["rho"], true[
    "fa"], true["fi"], true["fw"]

ertScheme = pg.DataContainerERT("erttrue.dat")

meshRST = pg.load("paraDomain.bms")
meshERT = pg.load("meshERT.bms")

fpm = FourPhaseModel(phi=0.4)

# Setup managers and equip with meshes
ert = ERTManager()
ert.setMesh(meshERT)
ert.setData(ertScheme)
ert.fop.createRefinedForwardMesh()

rst = Refraction("tttrue.dat", verbose=True)
ttData = rst.dataContainer
rst.setMesh(meshRST)
rst.fop.createRefinedForwardMesh()

# Setup joint modeling and inverse operators
JM = JointMod(meshRST, ert, rst, fpm, fix_poro=False)

data = pg.cat(ttData("t"), ertScheme("rhoa"))
error = pg.cat(rst.relErrorVals(ttData), ertScheme("err"))
inv = JointInv(JM, data, error, frmin=0.5, frmax=0.7)

# Set gradient starting model of f_ice, f_water, f_air = phi/3
velstart = np.loadtxt("rst_startmodel.dat")
rhostart = np.ones_like(velstart) * np.mean(ertScheme("rhoa"))
fas, fis, fws, _ = fpm.all(rhostart, velstart)
startmodel = np.concatenate((fws, fis, fas, np.ones_like(fas) - fpm.phi))

# Set result of conventional inversion as starting model
# rockstart = np.ones_like(conventional["fi"]) - fpm.phi
# startmodel = np.concatenate((conventional["fw"], conventional["fi"], conventional["fa"], rockstart))

startmodel[startmodel <= 0] = np.min(startmodel[startmodel > 0])
inv.setModel(startmodel)

# Run inversion
model = inv.run()
pg.boxprint(("Chi squared fit:", inv.getChi2()), sym="+")

# Save results
fwe, fie, fae, fre = JM.fractions(model)
fsum = fwe + fie + fae + fre

print("Min/Max sum:", min(fsum), max(fsum))

rhoest = JM.fpm.rho(fwe, fie, fae, fre)
velest = 1. / JM.fpm.slowness(fwe, fie, fae, fre)

array_mask = np.array( ((fae < 0) | (fae > 1 - fre))
                     | ((fie < 0) | (fie > 1 - fre))
                     | ((fwe < 0) | (fwe > 1 - fre))
                     | ((fre < 0) | (fre > 1))
                     | (fsum > 1.01))

np.savez("joint_inversion.npz", vel=np.array(velest), rho=np.array(rhoest),
         fa=fae, fi=fie, fw=fwe, fr=fre, mask=array_mask)
