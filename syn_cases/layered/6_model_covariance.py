#############################################
# to find "invlib" in the main folder
import sys, os
sys.path.insert(0, os.path.abspath("../.."))
#############################################

import numpy as np
import matplotlib.pyplot as plt

import pybert as pb
import pygimli as pg
import pygimli.meshtools as mt

from pybert.manager import ERTManager
from pygimli.physics import Refraction
from pygimli.physics.traveltime import createRAData
from invlib import FourPhaseModel


def forward4PM(mesh, schemeERT, schemeSRT, Fx):
    """Forward response of fx."""
    ert = ERTManager()
    srt = Refraction()
    phi = 1 - Fx[-1]
    fpm = FourPhaseModel(phi=phi)
    rho = fpm.rho(*Fx)
    slo = fpm.slowness(*Fx)
    rhoVec = rho[mesh.cellMarkers()]  # needs to be copied!
    dataERT = ert.simulate(mesh, rhoVec, schemeERT)
    sloVec = slo[mesh.cellMarkers()]
    dataSRT = srt.simulate(mesh, sloVec, schemeSRT)
    return dataERT, dataSRT


def jacobian4PM(mesh, schemeERT, schemeSRT, Fx, df=0.01,
                errorERT=0.03, errorSRT=0.0005):
    """Jacobian matrices by brute force."""
    dataERT, dataSRT = forward4PM(mesh, schemeERT, schemeSRT, Fx)
    npar = np.prod(Fsyn.shape)
    jacERT = np.zeros((dataERT.size(), npar))
    jacSRT = np.zeros((dataSRT.size(), npar))
    for i in range(npar):
        print(Fx.flat[i], end=" ")
        Fx1 = np.copy(Fx)
        Fx1.flat[i] += df
        dataERT1, dataSRT1 = forward4PM(mesh, schemeERT, schemeSRT, Fx1)
        jacERT[:, i] = (np.log(dataERT1('rhoa')) -
                        np.log(dataERT('rhoa'))) / df / errorERT
        jacSRT[:, i] = (dataSRT1('t') - dataSRT('t')) / df / errorSRT

    print("ready.")
    return jacERT, jacSRT


# load synthetic mesh (no boundary!)
mesh = pg.load("mesh.bms")
# create scheme files
sensors = np.load("sensors.npy")
shmERT = pb.createData(sensors, "dd")
shmSRT = createRAData(sensors)
# create synthetic model starting with phi
phi = np.array([0.4, 0.3, 0.3, 0.2, 0.3])
fr = 1 - phi
fw = np.array([0.3, 0.18, 0.1, 0.02, 0.02])
fi = np.array([0.0, 0.1, 0.18, 0.18, 0.28])
fa = phi - fw - fi
fa[np.isclose(fa, 0.0)] = 0.0
Fsyn = np.vstack((fw, fi, fa, fr))
# %% compute forward response and jacobians
# dataERT, dataSRT = forward4PM(mesh, shmERT, shmSRT, Fsyn)
jacERT, jacSRT = jacobian4PM(mesh, shmERT, shmSRT, Fsyn)
jacJoint = np.vstack((jacERT, jacSRT))
jacJoint.dump("jacJoint.npy")
print(jacERT.shape, jacSRT.shape, jacJoint.shape)
JTJ = jacJoint.T.dot(jacJoint)
MCM = np.linalg.inv(JTJ)
MCM.dump("MCM.npz")
plt.matshow(MCM)
# %%
npar, nreg = Fsyn.shape
gMat = np.zeros((nreg, npar*nreg))
for i in range(nreg):
    for j in range(npar):
        gMat[i, j*nreg+i] = 1.0
# %%
jacJointConst = np.vstack((jacJoint, gMat*1000))
JTJconst = jacJointConst.T.dot(jacJointConst)
MCMconst = np.linalg.inv(JTJconst)
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
