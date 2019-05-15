#############################################
# to find "invlib" in the main folder
import sys, os
sys.path.insert(0, os.path.abspath("../.."))
#############################################
import numpy as np

import pygimli as pg
from invlib import FourPhaseModel, JointInv, JointMod
from pybert.manager import ERTManager
from pygimli.physics import Refraction

maxIter = 50

# Load meshes and data
ertScheme = pg.DataContainerERT("ert_filtered.data")

mesh = pg.load("mesh.bms")
paraDomain = pg.load("paraDomain.bms")

conventional = np.load("conventional.npz")
fwconventional = conventional["fw"]
ficonventional = conventional["fi"]
faconventional = conventional["fa"]

fix_poro = False
if fix_poro:
    # frtrue = np.load("true_model.npz")["fr"]
    # phi = 1 - pg.interpolate(mesh, frtrue, meshRST.cellCenters()).array()
    fr_min = 0
    fr_max = 1
else:
    poro = 0.5
    phi = np.ones(paraDomain.cellCount()) * poro

fpm = FourPhaseModel(phi=phi, va=300., vi=3500., vw=1500, m=1.4, n=2.4,
                     rhow=60, vr=6000)

# Setup managers and equip with meshes
ert = ERTManager()
ert.setMesh(mesh)
ert.setData(ertScheme)
ert.fop.createRefinedForwardMesh()

rst = Refraction("rst_filtered.data", verbose=True)
ttData = rst.dataContainer
rst.setMesh(paraDomain)
rst.fop.createRefinedForwardMesh()

# Setup joint modeling and inverse operators
JM = JointMod(paraDomain, ert, rst, fpm, fix_poro=False, zWeight=0.5)

data = pg.cat(ttData("t"), ertScheme("rhoa"))
error = pg.cat(rst.relErrorVals(ttData), ertScheme("err"))
inv = JointInv(JM, data, error, frmin=0.2, frmax=0.9, lam=50, maxIter=maxIter)

# Set gradient starting model of f_ice, f_water, f_air = phi/3
velstart = np.loadtxt("rst_startmodel.dat")
rhostart = np.ones_like(velstart) * np.mean(ertScheme("rhoa"))
fas, fis, fws, _ = fpm.all(rhostart, velstart)
frs = np.ones_like(fas) - fpm.phi
startmodel = np.concatenate((fws, fis, fas, frs))

# Fix small values to avoid problems in first iteration
startmodel[startmodel <= 0.01] = 0.01
inv.setModel(startmodel)
# # Set gradient starting model of f_ice, f_water, f_air = phi/3
# velstart = np.loadtxt("rst_startmodel.dat")
# rhostart = np.ones_like(velstart) * np.mean(ertScheme("rhoa"))
# fas, fis, fws, _ = fpm.all(rhostart, velstart)
# startmodel = np.concatenate((fws, fis, fas, np.ones_like(fas) - fpm.phi))
#
# # Set result of conventional inversion as starting model
# rockstart = np.ones_like(conventional["fi"]) - fpm.phi
# startmodel = np.concatenate((conventional["fw"], conventional["fi"], conventional["fa"], rockstart))
#
# startmodel[startmodel <= 0] = np.min(startmodel[startmodel > 0])
# inv.setModel(startmodel)

# Run inversion
model = inv.run()
#pg.boxprint(("Chi squared fit:", inv.getChi2()), sym="+")
print(("Chi squared fit:", inv.getChi2()))

# Save results
fwe, fie, fae, fre = JM.fractions(model)
fsum = fwe + fie + fae + fre

print("Min/Max sum:", min(fsum), max(fsum))

rhoest = JM.fpm.rho(fwe, fie, fae, fre)
velest = 1. / JM.fpm.slowness(fwe, fie, fae, fre)

array_mask = np.array(((fae < 0) | (fae > 1 - fre))
                      | ((fie < 0) | (fie > 1 - fre))
                      | ((fwe < 0) | (fwe > 1 - fre))
                      | ((fre < 0) | (fre > 1))
                      | (fsum > 1.01))

np.savez("joint_inversion.npz", vel=np.array(velest), rho=np.array(rhoest),
         fa=fae, fi=fie, fw=fwe, fr=fre, mask=array_mask)
