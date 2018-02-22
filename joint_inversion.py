# import matplotlib; matplotlib.use("Agg")
import os

import matplotlib.pyplot as plt
import numpy as np

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

# Inititalize petrophysical four-phase model
fpm = FourPhaseModel()

# Initialize managers
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
            self.regionManager().region(i + 1).setConstraintType(1)

        self.mesh = self.meshlist[0]
        self.ERT = ertfop
        self.RST = rstfop
        self.fpm = petromodel
        self.cellCount = self.mesh.cellCount()
        self.createConstraints()

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

        fig, axs = plt.subplots(2,2)
        pg.show(self.mesh, fw, ax=axs[0,0], label="Water content", hold=True, logScale=False)
        pg.show(self.mesh, fi, ax=axs[1,0], label="Ice content", hold=True, logScale=False)
        pg.show(self.mesh, rho, ax=axs[0,1], label="Rho", hold=True)
        pg.show(self.mesh, 1/s, ax=axs[1,1], label="Velocity", hold=True)


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
        print("Vel:", np.min(1/s), np.mean(1/s), np.max(1/s))

        t = self.RST.fop.response(s)
        rhoa = self.ERT.fop.response(rho)
        # t = ttData("t") * 1.1
        # rhoa = ertScheme("rhoa") * 1.1

        return pg.cat(t, rhoa)

JM = JointMod(meshRST, ert, rst, fpm)
JM.setMultiThreadJacobian(1)
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

# Regularization
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
inv.fop().regionManager().paraDomain().save("paraDomain.bms")
# inv.run()
