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

# erte rste lam weighting zWeight

erte=float(sys.argv[1])
rste=float(sys.argv[2])
lam=float(sys.argv[3])
weighting=bool(sys.argv[4])
zWeight=float(sys.argv[5])

maxIter = 30

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
    poro = 0.5
else:
    poro = 0.5
    fr_min = 0.1
    fr_max = 0.9
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

# Set errors
ttData.set("err", np.ones(ttData.size()) * rste)
ertScheme.set("err", np.ones(ertScheme.size()) * erte)

# Setup joint modeling and inverse operators
JM = JointMod(paraDomain, ert, rst, fpm, fix_poro=False, zWeight=zWeight)

data = pg.cat(ttData("t"), ertScheme("rhoa"))

if weighting:
    n_rst = ttData.size()
    n_ert = ertScheme.size()
    avg = (n_rst + n_ert)/2
    weight_rst = avg / n_rst
    weight_ert = avg / n_ert
else:
    weight_rst = 1
    weight_ert = 1

error = pg.cat(rst.relErrorVals(ttData) / weight_rst, ertScheme("err") / weight_ert)
inv = JointInv(JM, data, error, frmin=fr_min, frmax=fr_max, lam=lam, maxIter=maxIter)

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

print("#" * 80)
ertchi, _ = JM.ERTchi2(model, error)
rstchi, _ = JM.RSTchi2(model, error, ttData("t"))
print("ERT chi^2", ertchi)
print("RST chi^2", rstchi)
print("#" * 80)

fig = JM.showFit(model)
title = "Overall chi^2 = %.2f" % inv.getChi2()
title += "\nERT chi^2 = %.2f" % ertchi
title += "\nRST chi^2 = %.2f" % rstchi
fig.suptitle(title)
fig.savefig("datafit.png", dpi=150)
