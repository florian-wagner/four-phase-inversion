import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

import pybert as pb
import pygimli as pg
import pygimli.meshtools as mt
from petro import FourPhaseModel, testFourPhaseModel
from pybert.manager import ERTManager
from pygimli.physics import Refraction
from pygimli.physics.traveltime import createRAData

# Close open figures from previous run
plt.close("all")

# Use 8 CPUs
pg.setThreadCount(8)

mesh = pg.load("mesh.bms")
true = np.load("true_model.npz")
conventional = np.load("conventional.npz")
sensors = np.load("sensors.npy")

veltrue, rhotrue, fatrue, fitrue, fwtrue = true["vel"], true["rho"], true[
    "fa"], true["fi"], true["fw"]

ertScheme = pg.DataContainerERT("erttrue.dat")

meshRST = pg.load("paraDomain.bms")
# meshRST = mt.createParaMesh(ertScheme, paraDepth=15, paraDX=0.2, smooth=[1, 2],
#                             paraMaxCellSize=1, quality=33.5, boundary=0,
#                             paraBoundary=3)

# meshRST.save("paraDomain.bms")

meshERT = mt.appendTriangleBoundary(meshRST, xbound=1000, ybound=500,
                                    smooth=True, quality=33.5,
                                    isSubSurface=True)
pg.show(meshERT)

fpm = FourPhaseModel()
# fa, fi, fw, mask = fpm.all(rhotrue, veltrue)

res = pg.RVector()
pg.interpolate(mesh, rhotrue, meshRST.cellCenters(), res)

vel = pg.RVector()
pg.interpolate(mesh, veltrue, meshRST.cellCenters(), vel)

ert = ERTManager()
ert.setMesh(meshERT)
ert.setData(ertScheme)

ttData = pg.DataContainer("tttrue.dat")
rst = Refraction("tttrue.dat", verbose=True)
rst.setMesh(meshRST)


class JointMod(pg.ModellingBase):
    def __init__(self, mesh, ertfop, rstfop, petromodel, verbose=True):
        pg.ModellingBase.__init__(self, verbose)
        self.mesh = pg.Mesh(mesh)

        # for i in range(2):
        #     for cell in mesh.cells():
        #         cell.setMarker(i + 1)
        #
        #     self.regionManager().addRegion(i + 1, self.mesh)
        #     # Set first order smoothing constraints
        #     self.regionManager().region(i + 1).setConstraintType(1)

        self.ERT = ertfop
        self.RST = rstfop
        self.fops = [self.RST, self.ERT]
        self.fpm = petromodel
        self.cellCount = self.mesh.cellCount()
        self.createConstraints()

    def createJacobian(self, model):
        fw = np.array(model[:self.cellCount])
        fi = np.array(model[self.cellCount:])

        rho = self.fpm.rho(fw)
        s = self.fpm.slowness(fw, fi)

        self.ERT.fop.createJacobian(rho)
        self.RST.fop.createJacobian(s)

        jacERT = self.ERT.fop.jacobian()
        jacRST = self.RST.fop.jacobian()

        self.jac = pg.BlockMatrix()
        nData = 0

        # Setting inner derivatives
        # rUL = 1. / self.fpm.vw - 1./self.fpm.vr - 1. / self.fpm.va
        # rUR = 1. / self.fpm.vi - 1./self.fpm.vr - 1. / self.fpm.va
        rUL = 1. / self.fpm.vw - 1. / self.fpm.va
        rUR = 1. / self.fpm.vi - 1. / self.fpm.va
        rLL = self.fpm.rho_deriv(fw)

        self.jacUL = pg.MultRightMatrix(jacRST, r=rUL)
        self.jacUR = pg.MultRightMatrix(jacRST, r=rUR)
        self.jacLL = pg.MultRightMatrix(jacERT, r=rLL)

        self.jac.addMatrix(self.jacUL, nData, 0)
        self.jac.addMatrix(self.jacUR, nData, self.cellCount)
        nData += self.RST.fop.data().size()  # update total vector length
        self.jac.addMatrix(self.jacLL, nData, 0)
        self.setJacobian(self.jac)

    def createConstraints(self):
        # First order smoothness matrix
        self._Ctmp = pg.RSparseMapMatrix()
        self.RST.fop.regionManager().fillConstraints(self._Ctmp)

        # Identity matrix for interparameter regularization
        self._I = pg.IdentityMatrix(self.cellCount)

        # Putting together in block matrix
        self._C = pg.RBlockMatrix()
        cid = self._C.addMatrix(self._Ctmp)
        self._C.addMatrixEntry(cid, 0, 0)
        self._C.addMatrixEntry(cid, self._Ctmp.rows(), self._Ctmp.cols())

        iid = self._C.addMatrix(self._I)
        self._C.addMatrixEntry(iid, self._Ctmp.rows() * 2, 0)
        self._C.addMatrixEntry(iid, self._Ctmp.rows() * 2, self._I.cols())
        self.setConstraints(self._C)

    def showModel(self, model):
        fw = np.array(model[:self.cellCount])
        fi = np.array(model[self.cellCount:])

        rho = self.fpm.rho(fw)
        s = self.fpm.slowness(fw, fi)

        fig, axs = plt.subplots(2, 2)
        pg.show(self.mesh, fw, ax=axs[0, 0], label="Water content", hold=True,
                logScale=False)
        pg.show(self.mesh, fi, ax=axs[1, 0], label="Ice content", hold=True,
                logScale=False)
        pg.show(self.mesh, rho, ax=axs[0, 1], label="Rho", hold=True)
        pg.show(self.mesh, 1 / s, ax=axs[1, 1], label="Velocity", hold=True)

    def response(self, model):
        return self.response_mt(model)

    def response_mt(self, model, i=0):
        os.environ["OMP_THREAD_LIMIT"] = "1"
        fw = np.array(model[:self.cellCount])
        fi = np.array(model[self.cellCount:])

        rho = self.fpm.rho(fw)
        s = self.fpm.slowness(fw, fi)

        fa = 1 - fw - fi - self.fpm.fr
        print("Ice:", np.min(fi), np.max(fi))
        print("Water:", np.min(fw), np.max(fw))
        print("Air:", np.min(fa), np.max(fa))
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
inv = pg.Inversion(dtrue, JM, verbose=True, dosave=True)

# Set data transformations
logtrans = pg.RTransLog()
trans = pg.RTrans()
cumtrans = pg.RTransCumulative()
cumtrans.add(trans, ttData.size())
cumtrans.add(logtrans, ertScheme.size())
inv.setTransData(cumtrans)

# Set model transformation
modtrans = pg.RTransLogLU(0, fpm.phi)
inv.setTransModel(modtrans)

# Set error
error = pg.cat(rst.relErrorVals(ttData), ertScheme("err"))
inv.setRelativeError(error)

# Set maximum number of iterations (default is 20)
inv.setMaxIter(50)

# Regularization strength
inv.setLambda(80)
inv.setLambdaFactor(0.8)

# Set homogeneous starting model of f_ice, f_water, f_air = phi/3
n = JM.cellCount * 2
startmodel = pg.RVector(n, fpm.phi / 3.)

# Set result of conventional inversion as starting model
icestart = pg.RVector()
pg.interpolate(mesh, conventional["fi"], meshRST.cellCenters(), icestart)
waterstart = pg.RVector()
pg.interpolate(mesh, conventional["fw"], meshRST.cellCenters(), waterstart)
startmodel = pg.cat(waterstart, icestart).array()
startmodel[startmodel <= 0] = np.min(startmodel[startmodel > 0])

inv.setModel(startmodel)
# Run inversion
cH = pg.cat(pg.RVector(JM._Ctmp.rows() * 2, 0.0), pg.RVector(JM._I.rows(), fpm.phi))
cW = pg.cat(pg.RVector(JM._Ctmp.rows() * 2, 1.0), pg.RVector(JM._I.rows(), 0.0))
# cH = pg.RVector(JM._Ctmp.rows(), 0.0)
# cW = pg.RVector(n * 2, 1.0)
inv.fop().createConstraints()
inv.setCWeight(cW)
inv.setConstraintsH(cH)
model = inv.run()
print("Chi squared fit:", inv.getChi2())

# Some visualization and saving
JM.showModel(model)

np.savetxt("model_iter%d.dat" % inv.iter(), model)

resp = JM(model)
fit = (resp - dtrue) / dtrue
plt.figure()
plt.plot(fit, "r.")
plt.axvline(ttData.size())

# Save results
fwe = np.array(model[:JM.cellCount])
fie = np.array(model[JM.cellCount:])
fae = 1 - fwe - fie - JM.fpm.fr
rhoest = JM.fpm.rho(fwe)
velest = 1. / JM.fpm.slowness(fwe, fie)
np.savez("joint_inversion.npz", vel=np.array(velest), rho=np.array(rhoest),
         fa=np.array(fae), fi=np.array(fie), fw=np.array(fwe))
