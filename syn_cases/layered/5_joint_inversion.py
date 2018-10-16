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

# Settings
if len(sys.argv) > 1:
    scenario = "Fig2"
    poro = 0.3 # startmodel if poro is estimated
    fix_poro = False
    poro_min = 0.2
    poro_max = 0.4
else:
    scenario = "Fig1"
    fix_poro = True
    poro_min = 0
    poro_max = 1

############
# Settings
maxIter = 15
############

# Poro to rock content (inversion parameter)
fr_min = 1 - poro_max
fr_max = 1 - poro_min

# Load meshes and data
mesh = pg.load("mesh.bms")
true = np.load("true_model.npz")
sensors = np.load("sensors.npy")

veltrue, rhotrue, fatrue, fitrue, fwtrue = true["vel"], true["rho"], true[
    "fa"], true["fi"], true["fw"]

ertScheme = pg.DataContainerERT("erttrue.dat")

meshRST = pg.load("paraDomain.bms")
meshERT = pg.load("meshERT.bms")

if fix_poro:
    frtrue = np.load("true_model.npz")["fr"]
    phi = 1 - pg.interpolate(mesh, frtrue, meshRST.cellCenters()).array()
    fr_min = 0
    fr_max = 1
else:
    phi = poro

fpm = FourPhaseModel(phi=phi)

# Setup managers and equip with meshes
ert = ERTManager()
ert.setMesh(meshERT)
ert.setData(ertScheme)
ert.fop.createRefinedForwardMesh()

rst = Refraction("tttrue.dat", verbose=True)
ttData = rst.dataContainer
rst.setMesh(meshRST, secNodes=5)
rst.fop.createRefinedForwardMesh()

# Setup joint modeling and inverse operators
JM = JointMod(meshRST, ert, rst, fpm, fix_poro=fix_poro)

data = pg.cat(ttData("t"), ertScheme("rhoa"))
error = pg.cat(rst.relErrorVals(ttData), ertScheme("err"))
inv = JointInv(JM, data, error, frmin=fr_min, frmax=fr_max, maxIter=maxIter)

# Set gradient starting model of f_ice, f_water, f_air = phi/3
velstart = np.loadtxt("rst_startmodel.dat")
rhostart = np.ones_like(velstart) * np.mean(ertScheme("rhoa"))
fas, fis, fws, _ = fpm.all(rhostart, velstart)
frs = np.ones_like(fas) - fpm.phi
if not fix_poro:
    frs[frs <= fr_min] = fr_min + 0.01
    frs[frs >= fr_max] = fr_max - 0.01
startmodel = np.concatenate((fws, fis, fas, frs))

# Fix small values to avoid problems in first iteration
startmodel[startmodel <= 0.01] = 0.01
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

np.savez("joint_inversion_%s.npz" % scenario, vel=np.array(velest), rho=np.array(rhoest),
         fa=fae, fi=fie, fw=fwe, fr=fre, mask=array_mask)
