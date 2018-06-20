import os

import matplotlib.pyplot as plt
import numpy as np

import pygimli as pg
import pygimli.meshtools as mt
from petro import FourPhaseModel
from pybert.manager import ERTManager
from pygimli.physics import Refraction

from lsqrinversion import LSQRInversion
# from pygimli.physics.traveltime import createRAData

# Close open figures from previous run
plt.close("all")

# Use 8 CPUs
pg.setThreadCount(4)


fileERT='SCH2017-08-29_rhoa_notopo_geom.dat'
fileRST='SCH2017-08-29_tt_notopo_geom.dat'
fileMesh='meshParaDomainSCH.bms'

#true = np.load("true_model.npz")
conventional = np.load("conventional.npz")
sensors = np.load("sensors.npy")

# mesh
mesh = pg.load('meshParaDomainSCH.bms')
pg.show(mesh, markers=False)

# ERT
ert = ERTManager()
ertData=ert.loadData(fileERT)
ertScheme = pg.DataContainerERT(fileERT)
##ert.setMesh(meshERT)
ert.setMesh(mesh)
ert.setData(ertScheme)
ert.fop.createRefinedForwardMesh()

# RST
rst = Refraction()
tt=rst.loadData(fileRST)
ttData = rst.dataContainer
#rst = Refraction("tttrue.dat", verbose=True)
#rst = Refraction(ttData,verbose=True)
##rst.setMesh(meshRST)
rst.setMesh(mesh)
rst.fop.createRefinedForwardMesh()

fpm = FourPhaseModel()

#veltrue, rhotrue, fatrue, fitrue, fwtrue = true["vel"], true["rho"], true["fa"], true["fi"], true["fw"]

#ertScheme = pg.DataContainerERT("erttrue.dat")

#meshRST = pg.load("paraDomain.bms")
#pg.show(meshRST, markers=False)

# meshRST = mt.createParaMesh(ertScheme, paraDepth=15, paraDX=0.2,
#                              smooth=[1, 2], paraMaxCellSize=1, quality=33.5,
#                              boundary=0, paraBoundary=3)

# meshRST.save("paraDomain.bms")

#meshERT = mt.appendTriangleBoundary(meshRST, xbound=1000, ybound=500,  smooth=True, quality=33.5,   isSubSurface=True)



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

    def createJacobian(self, model):
        fw = np.array(model[:self.cellCount])
        fi = np.array(model[self.cellCount:2 * self.cellCount])
        fa = np.array(model[2 * self.cellCount:])

        rho = self.fpm.rho(fw, fi, fa)
        s = self.fpm.slowness(fw, fi, fa)

        self.ERT.fop.createJacobian(rho)
        self.RST.fop.createJacobian(s)

        jacERT = self.ERT.fop.jacobian()
        jacRST = self.RST.fop.jacobian()

        # Setting inner derivatives
        self.jacUL = pg.MultRightMatrix(jacRST,
                                        r=1. / self.fpm.vw - 1. / self.fpm.vr)
        self.jacUM = pg.MultRightMatrix(jacRST,
                                        r=1. / self.fpm.vi - 1. / self.fpm.vr)
        self.jacUR = pg.MultRightMatrix(jacRST,
                                        r=1. / self.fpm.va - 1. / self.fpm.vr)
        self.jacLL = pg.MultRightMatrix(jacERT,
                                        r=self.fpm.rho_deriv_fw(fw, fi, fa))
        self.jacLM = pg.MultRightMatrix(jacERT,
                                        r=self.fpm.rho_deriv_fi_fa(fw, fi, fa))
        self.jacLR = pg.MultRightMatrix(jacERT,
                                        r=self.fpm.rho_deriv_fi_fa(fw, fi, fa))

        # Putting subjacobians together in block matrix
        self.jac = pg.BlockMatrix()
        nData = 0
        self.jac.addMatrix(self.jacUL, nData, 0)
        self.jac.addMatrix(self.jacUM, nData, self.cellCount)
        self.jac.addMatrix(self.jacUR, nData, self.cellCount * 2)
        nData += self.RST.fop.data().size()  # update total vector length
        self.jac.addMatrix(self.jacLL, nData, 0)
        self.jac.addMatrix(self.jacLM, nData, self.cellCount)
        self.jac.addMatrix(self.jacLR, nData, self.cellCount * 2)
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
        self.setConstraints(self._C)

        self._G = pg.RBlockMatrix()
        iid = self._G.addMatrix(self._I)
        self._G.addMatrixEntry(iid, 0, 0)
        self._G.addMatrixEntry(iid, 0, self.cellCount)
        self._G.addMatrixEntry(iid, 0, self.cellCount * 2)
#        self._G.addMatrixEntry(iid, self._Ctmp.rows() * 3, 0)
#        self._G.addMatrixEntry(iid, self._Ctmp.rows() * 3, self.cellCount)
#        self._G.addMatrixEntry(iid, self._Ctmp.rows() * 3, self.cellCount * 2)

    def showModel(self, model):
        fw = np.array(model[:self.cellCount])
        fi = np.array(model[self.cellCount:2 * self.cellCount])
        fa = np.array(model[2 * self.cellCount:])

        rho = self.fpm.rho(fw, fi, fa)
        s = self.fpm.slowness(fw, fi, fa)

        fig, axs = plt.subplots(3, 2)
        pg.show(self.mesh, fw, ax=axs[0, 0], label="Water content", hold=True,
                logScale=False)
        pg.show(self.mesh, fi, ax=axs[1, 0], label="Ice content", hold=True,
                logScale=False)
        pg.show(self.mesh, fa, ax=axs[2, 0], label="Air content", hold=True,
                logScale=False)
        pg.show(self.mesh, rho, ax=axs[0, 1], label="Rho", hold=True)
        pg.show(self.mesh, 1 / s, ax=axs[1, 1], label="Velocity", hold=True)

    def response(self, model):
        return self.response_mt(model)

    def response_mt(self, model, i=0):
        os.environ["OMP_THREAD_LIMIT"] = "1"
        fw = np.array(model[:self.cellCount])
        fi = np.array(model[self.cellCount:2 * self.cellCount])
        fa = np.array(model[2 * self.cellCount:])

        rho = self.fpm.rho(fw, fi, fa)
        s = self.fpm.slowness(fw, fi, fa)

        print("=" * 60)
        print("Ice:", np.min(fi), np.max(fi))
        print("Water:", np.min(fw), np.max(fw))
        print("Air:", np.min(fa), np.max(fa))
        print("=" * 60)
        print("Porosity", np.min(fa + fw + fi), np.max(fa + fw + fi))
        print("=" * 60)
        print("Rho:", np.min(rho), np.max(rho))
        print("Vel:", np.min(1 / s), np.max(1 / s))

        t = self.RST.fop.response(s)
        rhoa = self.ERT.fop.response(rho)

        return pg.cat(t, rhoa)

#JM = JointMod(meshRST, ert, rst, fpm)
JM = JointMod(mesh, ert, rst, fpm)
JM.setMultiThreadJacobian(8)
JM.setVerbose(True)

# pg.solver.showSparseMatrix(JM.constraints())
dtrue = pg.cat(ttData("t"), ertScheme("rhoa"))
# inv = pg.Inversion(dtrue, JM, verbose=True, dosave=True)
inv = LSQRInversion(dtrue, JM, verbose=True, dosave=True)

# Set data transformations
logtrans = pg.RTransLog()
# trans = pg.RTrans()
# cumtrans = pg.RTransCumulative()
# cumtrans.add(trans, ttData.size())
# cumtrans.add(logtrans, ertScheme.size())
# inv.setTransData(cumtrans)
inv.setTransData(logtrans)

# Set model transformation
modtrans = pg.RTransLogLU(0, fpm.phi)
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

# Set result of conventional inversion as starting model
#icestart = pg.RVector()
#pg.interpolate(mesh, conventional["fi"], meshRST.cellCenters(), icestart)
#waterstart = pg.RVector()
#pg.interpolate(mesh, conventional["fw"], meshRST.cellCenters(), waterstart)
#airstart = pg.RVector()
#pg.interpolate(mesh, conventional["fa"], meshRST.cellCenters(), airstart)
#startmodel = pg.cat(pg.cat(waterstart, icestart), airstart).array()
#startmodel[startmodel <= 0] = np.min(startmodel[startmodel > 0])

# Set homogeneous starting model of f_ice, f_water, f_air = phi/3
n = JM.cellCount * 3
startmodel = pg.RVector(n, fpm.phi / 3.)

inv.setModel(startmodel)
# Run inversion
inv.fop().createConstraints()
phiVec = pg.RVector(JM._I.rows(), fpm.phi)
inv.setParameterConstraints(JM._G, phiVec, 1000)
#inv.setConstraintsH(cH)

# min constraintsH = ndef. max constraintsH = ndef.
model = inv.run()
print("Chi squared fit:", inv.getChi2())

# Sensitivity comparison
# model = pg.load("with_fa/model_0.vector")
# JM.createJacobian(model)
# jac_num = pg.utils.gmat2numpy(pg.load("sens.bmat"))
#
# pg.show(meshRST, jac_num[400,JM.cellCount*2:], label="Sens", logScale=False)
# pg.show(meshRST, JM.jacUR.A.row(400) * 0, label="Sens", logScale=False)

# Some visualization and saving
JM.showModel(model)

np.savetxt("model_iter%d.dat" % inv.iter(), model)

resp = JM(model)
fit = (resp - dtrue) / dtrue
fig=plt.figure()
plt.plot(fit, "r.")
plt.axvline(ttData.size())

fig.savefig("joint_inversion_fit.png", dpi=120)


# Save results
fwe = np.array(model[:JM.cellCount])
fie = np.array(model[JM.cellCount:2 * JM.cellCount])
fae = np.array(model[2 * JM.cellCount:])
fsum = fwe + fie + fae
print("Min/Max sum:", min(fsum), max(fsum))

rhoest = JM.fpm.rho(fwe, fie, fae)
velest = 1. / JM.fpm.slowness(fwe, fie, fae)
np.savez("joint_inversion.npz", vel=np.array(velest), rho=np.array(rhoest),
         fa=np.array(fae), fi=np.array(fie), fw=np.array(fwe))
