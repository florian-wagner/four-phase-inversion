import os

import matplotlib.pyplot as plt
import numpy as np

import pybert as pb
import pygimli as pg
import pygimli.meshtools as mt
from lsqrinversion import LSQRInversion
from petro import FourPhaseModel
from pybert.manager import ERTManager
from pygimli.physics import Refraction

# from pygimli.physics.traveltime import createRAData

# Close open figures from previous run
plt.close("all")

# Use 8 CPUs
pg.setThreadCount(4)

mesh = pg.load("mesh.bms")
true = np.load("true_model.npz")
conventional = np.load("conventional.npz")
sensors = np.load("sensors.npy")

veltrue, rhotrue, fatrue, fitrue, fwtrue = true["vel"], true["rho"], true[
    "fa"], true["fi"], true["fw"]

ertScheme = pg.DataContainerERT("erttrue.dat")

meshRST = pg.load("paraDomain.bms")
# meshRST = mt.createParaMesh(ertScheme, paraDepth=15, paraDX=0.2,
#                              smooth=[1, 2], paraMaxCellSize=1, quality=33.5,
#                              boundary=0, paraBoundary=3)

# meshRST.save("paraDomain.bms")

meshERT = pg.load("meshERT.bms")

fpm = FourPhaseModel()

ert = ERTManager()
ert.setMesh(meshERT)
ert.setData(ertScheme)
ert.fop.createRefinedForwardMesh()

rst = Refraction("tttrue.dat", verbose=True)
ttData = rst.dataContainer
rst.setMesh(meshRST)
rst.fop.createRefinedForwardMesh()


class JointMod(pg.ModellingBase):
    def __init__(self, mesh, ertfop, rstfop, petromodel, verbose=True):
        pg.ModellingBase.__init__(self, verbose)
        self.mesh = pg.Mesh(mesh)
        self.ERT = ertfop
        self.RST = rstfop
        self.fops = [self.RST, self.ERT]
        self.fpm = petromodel
        self.cellCount = self.mesh.cellCount()
        self.createConstraints()

    def fractions(self, model):
        """Split model vector into individual distributions"""
        return np.reshape(model, (4, self.cellCount))

    def createJacobian(self, model):
        fw, fi, fa, fr = self.fractions(model)

        rho = self.fpm.rho(fw, fi, fa, fr)
        s = self.fpm.slowness(fw, fi, fa, fr)

        self.ERT.fop.createJacobian(rho)
        self.RST.fop.createJacobian(s)

        jacERT = self.ERT.fop.jacobian()
        jacRST = self.RST.fop.jacobian()

        # Setting inner derivatives
        self.jacRSTW = pg.MultRightMatrix(jacRST, r=1. / self.fpm.vw)
        self.jacRSTI = pg.MultRightMatrix(jacRST, r=1. / self.fpm.vi)
        self.jacRSTA = pg.MultRightMatrix(jacRST, r=1. / self.fpm.va)
        self.jacRSTR = pg.MultRightMatrix(jacRST, r=1. / self.fpm.vr)

        self.jacERTW = pg.MultRightMatrix(
            jacERT, r=self.fpm.rho_deriv_fw(fw, fi, fa, fr))
        self.jacERTR = pg.MultRightMatrix(
            jacERT, r=self.fpm.rho_deriv_fr(fw, fi, fa, fr))

        # Putting subjacobians together in block matrix
        self.jac = pg.BlockMatrix()
        nData = 0
        self.jac.addMatrix(self.jacRSTW, nData, 0)
        self.jac.addMatrix(self.jacRSTI, nData, self.cellCount)
        self.jac.addMatrix(self.jacRSTA, nData, self.cellCount * 2)
        self.jac.addMatrix(self.jacRSTR, nData, self.cellCount * 3)
        nData += self.RST.fop.data().size()  # update total vector length
        self.jac.addMatrix(self.jacERTW, nData, 0)
        self.jac.addMatrix(self.jacERTR, nData, self.cellCount * 3)
        self.setJacobian(self.jac)

    def createConstraints(self):
        # First order smoothness matrix
        self._Ctmp = pg.RSparseMapMatrix()
        self.RST.fop.regionManager().fillConstraints(self._Ctmp)

        # # Geostatistical constraints
        # CM = pg.utils.geostatistics.covarianceMatrix(self.mesh, I=[40, 3])
        # self._Ctmp = pg.matrix.Cm05Matrix(CM)

        # Identity matrix for interparameter regularization
        self._I = pg.IdentityMatrix(self.cellCount)

        # Putting together in block matrix
        self._C = pg.RBlockMatrix()
        cid = self._C.addMatrix(self._Ctmp)
        self._C.addMatrixEntry(cid, 0, 0)
        self._C.addMatrixEntry(cid, self._Ctmp.rows(), self.cellCount)
        self._C.addMatrixEntry(cid, self._Ctmp.rows() * 2, self.cellCount * 2)
        self._C.addMatrixEntry(cid, self._Ctmp.rows() * 3, self.cellCount * 3)
        self.setConstraints(self._C)

        self._G = pg.RBlockMatrix()
        iid = self._G.addMatrix(self._I)
        self._G.addMatrixEntry(iid, 0, 0)
        self._G.addMatrixEntry(iid, 0, self.cellCount)
        self._G.addMatrixEntry(iid, 0, self.cellCount * 2)
        self._G.addMatrixEntry(iid, 0, self.cellCount * 3)
        # Fix f_r
        self._G.addMatrixEntry(iid, self._G.rows(), self.cellCount * 3)

    def showModel(self, model):
        fw, fi, fa, fr = self.fractions(model)

        rho = self.fpm.rho(fw, fi, fa, fr)
        s = self.fpm.slowness(fw, fi, fa, fr)

        fig, axs = plt.subplots(3, 2)
        pg.show(self.mesh, fw, ax=axs[0, 0], label="Water content", hold=True,
                logScale=False, cmap="Blues")
        pg.show(self.mesh, fi, ax=axs[1, 0], label="Ice content", hold=True,
                logScale=False, cmap="Purples")
        pg.show(self.mesh, fa, ax=axs[2, 0], label="Air content", hold=True,
                logScale=False, cmap="Greens")
        pg.show(self.mesh, rho, ax=axs[0, 1], label="Rho", hold=True, cmap="Spectral_r")
        pg.show(self.mesh, 1 / s, ax=axs[1, 1], label="Velocity")

    def showFit(self, model):
        resp = self.response(model)

        fig, (ax1, ax2) = plt.subplots(1, 2)
        self.RST.showData(response=resp[:self.RST.dataContainer.size()],
                          ax=ax1)
        resprhoa = resp[self.RST.dataContainer.size():]

        fit = (ert.data("rhoa") - resprhoa) / resprhoa * 100
        lim = np.max(np.abs(fit))
        pb.show(ert.data, vals=fit, cMin=-lim, cMax=lim, label="Relative fit",
                cmap="RdBu_r", ax=ax2)
        fig.show()

    def response(self, model):
        return self.response_mt(model)

    def response_mt(self, model, i=0):
        os.environ["OMP_THREAD_LIMIT"] = "1"
        fw, fi, fa, fr = self.fractions(model)

        rho = self.fpm.rho(fw, fi, fa, fr)
        s = self.fpm.slowness(fw, fi, fa, fr)

        print("=" * 60)
        print("Water:", np.min(fw), np.max(fw))
        print("Ice:", np.min(fi), np.max(fi))
        print("Air:", np.min(fa), np.max(fa))
        print("Rock:", np.min(fr), np.max(fr))
        print("=" * 60)
        print("SUM", np.min(fa + fw + fi + fr), np.max(fa + fw + fi + fr))
        print("=" * 60)
        print("Rho:", np.min(rho), np.max(rho))
        print("Vel:", np.min(1 / s), np.max(1 / s))

        t = self.RST.fop.response(s)
        rhoa = self.ERT.fop.response(rho)

        return pg.cat(t, rhoa)


JM = JointMod(meshRST, ert, rst, fpm)
JM.setMultiThreadJacobian(8)
JM.setVerbose(True)

# pg.solver.showSparseMatrix(JM.constraints())
dtrue = pg.cat(ttData("t"), ertScheme("rhoa"))
# inv = pg.Inversion(dtrue, JM, verbose=True, dosave=True)
inv = LSQRInversion(dtrue, JM, verbose=True, dosave=True)

# Set data transformations
logtrans = pg.RTransLog()
trans = pg.RTrans()
cumtrans = pg.RTransCumulative()
cumtrans.add(trans, ttData.size())
cumtrans.add(logtrans, ertScheme.size())
inv.setTransData(cumtrans)
inv.setTransData(logtrans)

# Set model transformation
modtrans = pg.RTransLogLU(0, 1)
inv.setTransModel(modtrans)

# Set error
error = pg.cat(rst.relErrorVals(ttData), ertScheme("err"))
inv.setRelativeError(error)

# Set maximum number of iterations (default is 20)
inv.setMaxIter(50)

# Regularization strength
inv.setLambda(20)
inv.setDeltaPhiAbortPercent(1)
# inv.setLambdaFactor(0.8)
# inv.fop().regionManager().setZWeight(0.5)

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
inv.fop().createConstraints()
ones = pg.RVector(JM._I.rows(), 1.0)
phiVec = pg.cat(ones, ones - fpm.phi)
inv.setParameterConstraints(JM._G, phiVec, 10000)
#inv.setConstraintsH(cH)

# Some visualization and saving
JM.showModel(startmodel)
JM.showFit(startmodel)
# pg.wait()
# 1/0

model = inv.run()
print("Chi squared fit:", inv.getChi2())
JM.showModel(model)



np.savetxt("model_iter%d.dat" % inv.iter(), model)

# Save results
fwe, fie, fae, fre = JM.fractions(model)
fsum = fwe + fie + fae + fre

print("Min/Max sum:", min(fsum), max(fsum))

rhoest = JM.fpm.rho(fwe, fie, fae, fre)
velest = 1. / JM.fpm.slowness(fwe, fie, fae, fre)
np.savez("joint_inversion.npz", vel=np.array(velest), rho=np.array(rhoest),
         fa=fae, fi=fie, fw=fwe, fr=fre)
