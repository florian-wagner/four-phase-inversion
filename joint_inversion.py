import matplotlib; matplotlib.use("Agg")
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

veltrue, rhotrue, fatrue, fitrue, fwtrue = true["vel"], true["rho"], true["fa"], true["fi"], true["fw"]

ertScheme = pb.load("erttrue.dat")

meshRST = mt.createParaMesh(ertScheme, paraDepth=15, paraDX=0.33,
                            paraMaxCellSize=1.0, boundary=0, paraBoundary=3)

pg.show(meshRST)

meshERT = mt.appendTriangleBoundary(meshRST, xbound=1000, ybound=500,
                                    smooth=True, quality=33.5, isSubSurface=True)
pg.show(meshERT)

fpm = FourPhaseModel()
# fa, fi, fw, mask = fpm.all(rhotrue, veltrue)

res = pg.RVector()
pg.interpolate(mesh, rhotrue, meshRST.cellCenters(), res)

vel = pg.RVector()
pg.interpolate(mesh, veltrue, meshRST.cellCenters(), vel)

def solveERT(model):
    ert = ERTManager()
    ert.setMesh(pg.Mesh(meshERT))
    ertScheme = pb.load("erttrue.dat")
    ert.setData(ertScheme)
    ert.fop.setThreadCount(1)
    return ert.fop.response(model)

ert = ERTManager()
ert.setMesh(meshERT)
ert.setData(ertScheme)

def solveRST(model):
    ttData = pg.DataContainer("tttrue.dat")
    rst = Refraction("tttrue.dat", verbose=True)
    rst.setMesh(pg.Mesh(meshRST))
    rst.fop.setThreadCount(1)
    return rst.fop.response(model)

ttData = pg.DataContainer("tttrue.dat")
rst = Refraction("tttrue.dat", verbose=True)
rst.setMesh(meshRST)

class JointMod(pg.ModellingBase):
    def __init__(self, mesh, ertfop, rstfop, petromodel, verbose=True):
        pg.ModellingBase.__init__(self, verbose)
        self.setMesh(mesh)
        self.meshlist = []
        for i in range(2):
            for cell in mesh.cells():
                cell.setMarker(i + 1)
            self.meshlist.append(pg.Mesh(mesh))
            self.regionManager().addRegion(i + 1, self.meshlist[i])
            self.regionManager().recountParaMarker_()

        self.ERT = ertfop
        self.RST = rstfop
        self.fpm = petromodel

    def response(self, model):
        return self.response_mt(model)

    def response_mt(self, model, i=0):
        mesh = pg.Mesh(self.mesh())
        fw = model[:mesh.cellCount()]
        fi = model[mesh.cellCount():]
        rho = self.fpm.rho(fw)
        s = self.fpm.slowness(fw, fi)
        rhoa = self.ERT.fop.response(rho)
        t = self.RST.fop.response(s)
        # rhoa = solveERT(rho)
        # t = solveRST(s)

        return pg.cat(t, rhoa)

JM = JointMod(meshRST, ert, rst, fpm)
JM.setMultiThreadJacobian(2)
JM.setVerbose(True)

# Test forward simulation
model = pg.cat(res, vel)
response = JM.response(model)

# Inversion
dtrue = pg.cat(ttData("t"), ertScheme("rhoa"))
inv = pg.Inversion(dtrue, JM, verbose=True)
rm = JM.regionManager()

# Set homogeneous starting model of f_ice, f_water, f_air = phi/3
n = rm.parameterCount()
inv.setModel(pg.RVector(n, fpm.phi / 3.))
logtrans = pg.RTransLog()
inv.setTransData(logtrans)
for i in range(rm.regionCount()):
    # Let parameter range between 0 and porosity
    rm.region(i + 1).setLowerBound(0.0)
    rm.region(i + 1).setUpperBound(fpm.phi)


inv.start()
