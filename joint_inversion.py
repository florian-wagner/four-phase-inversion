# import matplotlib; matplotlib.use("Agg")
import os

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

pg.setThreadCount(1)


class WorkSpace():
    """ Empty class to store some data. """
    pass


mesh = pg.load("mesh.bms")
true = np.load("true_model.npz")
sensors = np.load("sensors.npy")

veltrue, rhotrue, fatrue, fitrue, fwtrue = true["vel"], true["rho"], true[
    "fa"], true["fi"], true["fw"]

ertScheme = pg.DataContainerERT("erttrue.dat")

meshRST = mt.createParaMesh(ertScheme, paraDepth=15, paraDX=0.33,
                            paraMaxCellSize=1.5, boundary=0, paraBoundary=3)

pg.show(meshRST)

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
        self.meshlist = []
        for i in range(2):
            for cell in mesh.cells():
                cell.setMarker(i + 1)

            self.meshlist.append(pg.Mesh(mesh))
            self.regionManager().addRegion(i + 1, self.meshlist[i])
            # Set first order smoothing constraints
            self.regionManager().region(i + 1).setConstraintType(1)

        self.mesh = self.meshlist[0]
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

    # def createJacobian(self, model):
    #     """Fill the individual jacobian matrices."""
    #     self.fop.createJacobian(self.trans(model))
    #     self.jac.r = self.trans.deriv(model)  # set inner derivative

    # def createConstraints(self):
    #     self._Ctmp = pg.RSparseMapMatrix()
    #     self.RST.fop.regionManager().fillConstraints(self._Ctmp)
    #     self._C = pg.RBlockMatrix()
    #     cid = self._C.addMatrix(self._Ctmp)
    #     self._C.addMatrixEntry(cid, 0, 0)
    #     self._C.addMatrixEntry(cid, self._Ctmp.rows(), self._Ctmp.cols())
    #     self.setConstraints(self._C)

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

        print("Water:", np.min(fw), np.mean(fw), np.max(fw))
        print("Ice:", np.min(fi), np.mean(fi), np.max(fi))
        print("Rho:", np.min(rho), np.mean(rho), np.max(rho))
        print("Vel:", np.min(1 / s), np.mean(1 / s), np.max(1 / s))

        t = self.RST.fop.response(s)
        rhoa = self.ERT.fop.response(rho)

        return pg.cat(t, rhoa)


JM = JointMod(meshRST, ert, rst, fpm)
# JM.setMultiThreadJacobian(1)
JM.setVerbose(True)

# Let parameter range between 0 and porosity
# for i in range(JM.regionManager().regionCount()):
#     JM.regionManager().region(i + 1).setLowerBound(0.0)
#     JM.regionManager().region(i + 1).setUpperBound(fpm.phi)
#     JM.regionManager().region(i + 1).setConstraintType(1)
# 1/0
# Inversion

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
inv.setRelativeError(0.01)

# Regularization strength
inv.setLambda(50)
inv.setLambdaFactor(0.9)

# Set homogeneous starting model of f_ice, f_water, f_air = phi/3
n = JM.regionManager().parameterCount()
startmodel = pg.RVector(n, fpm.phi / 3.)
inv.setModel(startmodel)

inv.fop().createConstraints()

# Test fop response
JM(startmodel)

# Save parameter domain for visualization later
# inv.fop().regionManager().paraDomain().save("paraDomain.bms")
inv.run()

#
# # XXX: Temporary for sens comparsion
#
# model = pg.load("./brute/model_0.vector")
# pd = pg.load("./brute/paraDomain.bms")
# pg.show(pd, model[:len(model) // 2], label="Fi")
#
# jac = pg.MultRightMatrix(pg.load("./brute/sens.bmat"))
# # sns.heatmap(jac[:ttData.size()])
#
# # inv.setModel(model)
# # inv.run()
#
# # inv.run()
# # print("Chi squared fit:", inv.getChi2())
# # print(jac[0])
# # print(JM.jacUL[0])
#
# JM.createJacobian(model)
# # print(JM.jacUL.mult(pg.RVector(JM.cellCount, 1.0))[0])
# # print(jac.mult(pg.RVector(JM.cellCount, 1.0))[0])
# plt.close("all")
#
#
# def transFwdWyllieS(phi, vm=4000, vw=1600, va=330):
#     """Wyllie transformation function slowness(saturation)."""
#     if va != 330.0:
#         print(va, "va is not 330.0")
#         raise BaseException('TODO')
#     return pg.RTransLin((1 / vw - 1. / va) * phi, (1 - phi) / vm + phi / va)
#
#
# def transFwdArchieS(rFluid=100, phi=0.4, m=1., n=1., a=2.):  # rho(S)
#     """Inverse Archie transformation function resistivity(saturation)."""
#     # rho = rFluid * phi^(-m) S^(-n)
#     return pg.RTransPower(-n, (a * rFluid * phi**(-m))**(1 / n))
#
# trans = transFwdArchieS()
# trans.fwd(0.5)
# JM.fpm.rho(np.ones(1) * 0.5)
# JM.fpm.phi
# JM.fpm.rho(np.ones(1) * 0.4)
# trans.fwd(1)
# trans.invTrans(500)
# JM.fpm.rho_deriv(500)
