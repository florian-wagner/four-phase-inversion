import matplotlib.pyplot as plt
import numpy as np

import pybert as pb
import pygimli as pg
import pygimli.meshtools as mt
from fpinv import FourPhaseModel
from pybert.manager import ERTManager
from pygimli.physics import Refraction
from pygimli.physics.traveltime import createRAData


def forward4PM(meshERT, meshRST, schemeERT, schemeSRT, Fx):
    """Forward response of fx."""
    ert = ERTManager()
    srt = Refraction()
    phi = 1 - Fx[-1]
    fpm = FourPhaseModel(phi=phi)
    rho = fpm.rho(*Fx)
    slo = fpm.slowness(*Fx)

    rho = np.append(rho, np.mean(rho))  # outer region
    rhoVec = rho[meshERT.cellMarkers()]  # needs to be copied!
    sloVec = slo[meshRST.cellMarkers()]

    dataERT = ert.simulate(meshERT, rhoVec, schemeERT)
    dataSRT = srt.simulate(meshRST, sloVec, schemeSRT)
    return dataERT, dataSRT


def jacobian4PM(meshERT, meshRST, schemeERT, schemeSRT, Fx, df=0.01,
                errorERT=0.03, errorSRT=0.0005):
    """Jacobian matrices by brute force."""
    dataERT, dataSRT = forward4PM(meshERT, meshRST, schemeERT, schemeSRT, Fx)
    npar = np.prod(Fsyn.shape)
    jacERT = np.zeros((dataERT.size(), npar))
    jacSRT = np.zeros((dataSRT.size(), npar))
    for i in range(npar):
        print("\n")
        pg.boxprint("%d / %d" % (i + 1, npar))
        print(Fx.flat[i], end=" ")
        Fx1 = np.copy(Fx)
        Fx1.flat[i] += df
        dataERT1, dataSRT1 = forward4PM(meshERT, meshRST, schemeERT, schemeSRT,
                                        Fx1)
        jacERT[:, i] = (np.log(dataERT1('rhoa')) -
                        np.log(dataERT('rhoa'))) / df / errorERT
        jacSRT[:, i] = (dataSRT1('t') - dataSRT('t')) / df / errorSRT

    print("ready.")
    return jacERT, jacSRT


# load synthetic mesh
mesh = pg.load("mesh.bms")
meshRST = pg.load("paraDomain_2.bms")
meshRST.createSecondaryNodes(3)
meshERT = pg.load("meshERT_2.bms")

for cell in meshRST.cells():
    NN = mesh.findCell(cell.center())
    cell.setMarker(mesh.cellMarkers()[NN.id()])

for cell in meshERT.cells():
    NN = mesh.findCell(cell.center())
    if NN:
        cell.setMarker(mesh.cellMarkers()[NN.id()])
    else:
        cell.setMarker(len(np.unique(mesh.cellMarkers())))  # triangle boundary

# create scheme files
sensors = np.load("sensors.npy", allow_pickle=True)
shmERT = pg.DataContainerERT("erttrue.dat")
shmSRT = createRAData(sensors)

Fsyn = np.loadtxt("syn_model.dat")

# %% compute forward response and jacobians
jacERT, jacSRT = jacobian4PM(meshERT, meshRST, shmERT, shmSRT, Fsyn)
jacJoint = np.vstack((jacSRT, jacERT))
print(jacERT.shape, jacSRT.shape, jacJoint.shape)
jacJoint.dump("jacJoint.npz")
pg.tic("Calculating JTJ")
JTJ = jacJoint.T.dot(jacJoint)
pg.toc()
MCM = np.linalg.inv(JTJ)
MCM.dump("MCM.npz")
# plt.matshow(MCM)
# %%
npar, nreg = Fsyn.shape
gMat = np.zeros((nreg, npar * nreg))
for i in range(nreg):
    for j in range(npar):
        gMat[i, j * nreg + i] = 1.0
# %%
pg.tic("Calculating JTJ")
jacJointConst = np.vstack((jacJoint, gMat * 10000))
JTJconst = jacJointConst.T.dot(jacJointConst)
pg.toc()

pg.tic("Matrix inversion")
MCMconst = np.linalg.inv(JTJconst)
pg.toc()

MCMconst.dump("MCMconst.npz")
plt.matshow(MCMconst)
# %% extract variances and scale MCM to diagonal
varVG = np.sqrt(np.diag(MCM))
print(varVG.reshape((Fsyn.shape)))
di = (1.0 / varVG)
MCMs = di.reshape(len(di), 1) * MCM * di
plt.matshow(MCMs, cmap=plt.cm.bwr, vmin=-1, vmax=1)
MCMs.dump("MCMs.npz")
plt.matshow(varVG.reshape((Fsyn.shape)))
