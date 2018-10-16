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
from invlib import FourPhaseModel, JointMod, JointInv


def forward4PM(meshERT, meshRST, schemeERT, schemeSRT, Fx):
    """Forward response of fx."""
    ert = ERTManager()
    srt = Refraction()
    phi = 1 - Fx[-1]
    fpm = FourPhaseModel(phi=phi)
    rho = fpm.rho(*Fx)
    slo = fpm.slowness(*Fx)
    if len(rho < meshRST.cellCount()):
        rho = np.append(rho, np.mean(rho)) # outer region
        rhoVec = rho[meshERT.cellMarkers()]  # needs to be copied!
        sloVec = slo[meshRST.cellMarkers()]
    else:
        rhoVec = rho
        sloVec = slo

    dataERT = ert.simulate(meshERT, rhoVec, schemeERT)
    dataSRT = srt.simulate(meshRST.createSecondaryNodes(5), sloVec, schemeSRT)
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
        pg.boxprint(f"{i+1} / {npar}")
        print(Fx.flat[i], end=" ")
        Fx1 = np.copy(Fx)
        Fx1.flat[i] += df
        dataERT1, dataSRT1 = forward4PM(meshERT, meshRST, schemeERT, schemeSRT, Fx1)
        jacERT[:, i] = (np.log(dataERT1('rhoa')) -
                        np.log(dataERT('rhoa'))) / df / errorERT
        jacSRT[:, i] = (dataSRT1('t') - dataSRT('t')) / df / errorSRT

    print("ready.")
    return jacERT, jacSRT


# load synthetic mesh (no boundary!)
mesh = pg.load("mesh.bms")
meshRST = pg.load("paraDomain.bms")
meshERT = pg.load("meshERT.bms")

for cell in meshRST.cells():
    NN = mesh.findCell(cell.center())
    cell.setMarker(mesh.cellMarkers()[NN.id()])

for cell in meshERT.cells():
    NN = mesh.findCell(cell.center())
    if NN:
        cell.setMarker(mesh.cellMarkers()[NN.id()])
    else:
        cell.setMarker(len(np.unique(mesh.cellMarkers()))) # triangle boundary

# create scheme files
sensors = np.load("sensors.npy")
shmERT = pg.DataContainerERT("erttrue.dat")
shmSRT = createRAData(sensors)
# create synthetic model starting with phi
phi = np.array([0.4, 0.3, 0.3, 0.2, 0.3])
fr = 1 - phi
fw = np.array([0.3, 0.18, 0.1, 0.02, 0.02])
fi = np.array([0.0, 0.1, 0.18, 0.18, 0.28])
fa = phi - fw - fi
fa[np.isclose(fa, 0.0)] = 0.0

def jac(meshERT, meshRST, schemeERT, schemeSRT, Fx, df=0.01, errorERT=0.03,
        errorSRT=0.0005):
    """ Calculate jacobian as during inversion. """
    # Setup managers and equip with meshes
    ert = ERTManager()
    ert.setMesh(meshERT)
    ert.setData(shmERT)
    ert.fop.createRefinedForwardMesh()

    rst = Refraction("tttrue.dat", verbose=True)
    ttData = rst.dataContainer
    rst.setMesh(meshRST)
    rst.fop.createRefinedForwardMesh()

    # Setup joint modeling and inverse operators
    fpm = FourPhaseModel(phi=phi)
    JM = JointMod(meshRST, ert, rst, fpm, fix_poro=False)

    if len(Fx) < meshRST.cellCount():
        data = Fx[:,meshRST.cellMarkers()].flatten()
    else:
        data = Fx

    JM.createJacobian(data)

    def block2numpy(mat):
        pg.tic()
        A = np.zeros((mat.rows(), mat.cols()))
        for i in range(mat.rows()):
            A[i] = mat.row(i)
        pg.toc()
        return A

    J = block2numpy(JM.jac)
    error = pg.cat(rst.relErrorVals(ttData), shmERT("err"))

    return J / error.array()[:, np.newaxis]


# Load joint inversion result
# joint = np.load("joint_inversion.npz")
# fa, fi, fw, fr = joint["fa"], joint["fi"], joint["fw"], joint["fr"]
Fsyn = np.vstack((fw, fi, fa, fr))
# jacJoint = jac(meshERT, meshRST, shmERT, shmSRT, Fsyn)
# jacJoint.dump("jacJoint2.npy")

# %% compute forward response and jacobians
# dataERT, dataSRT = forward4PM(meshERT, meshRST, shmERT, shmSRT, Fsyn)
jacERT, jacSRT = jacobian4PM(meshERT, meshRST, shmERT, shmSRT, Fsyn)
jacJoint = np.vstack((jacSRT, jacERT))
print(jacERT.shape, jacSRT.shape, jacJoint.shape)
jacJoint.dump("jacJoint.npy")
pg.tic("Calculating JTJ")
JTJ = jacJoint.T.dot(jacJoint)
pg.toc()
MCM = np.linalg.inv(JTJ)
MCM.dump("MCM.npz")
# plt.matshow(MCM)
# %%
npar, nreg = Fsyn.shape
gMat = np.zeros((nreg, npar*nreg))
for i in range(nreg):
    for j in range(npar):
        gMat[i, j*nreg+i] = 1.0
# %%
pg.tic("Calculating JTJ")
jacJointConst = np.vstack((jacJoint, gMat*1000))
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
