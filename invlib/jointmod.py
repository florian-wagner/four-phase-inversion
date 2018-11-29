import matplotlib.pyplot as plt
import numpy as np

import pygimli as pg
import pybert as pb


class JointMod(pg.ModellingBase):

    def __init__(self, mesh, ertfop, rstfop, petromodel, fix_poro=True,
                 verbose=True):
        pg.ModellingBase.__init__(self, verbose)
        self.mesh = pg.Mesh(mesh)
        self.ERT = ertfop
        self.RST = rstfop
        self.fops = [self.RST, self.ERT]
        self.fpm = petromodel
        self.cellCount = self.mesh.cellCount()
        self.fix_poro = fix_poro
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

        # Putting together in block matrix
        self._C = pg.RBlockMatrix()
        cid = self._C.addMatrix(self._Ctmp)
        self._C.addMatrixEntry(cid, 0, 0)
        self._C.addMatrixEntry(cid, self._Ctmp.rows(), self.cellCount)
        self._C.addMatrixEntry(cid, self._Ctmp.rows() * 2, self.cellCount * 2)
        self._C.addMatrixEntry(cid, self._Ctmp.rows() * 3, self.cellCount * 3)
        self.setConstraints(self._C)

        # Identity matrix for interparameter regularization
        self._I = pg.IdentityMatrix(self.cellCount)

        self._G = pg.RBlockMatrix()
        iid = self._G.addMatrix(self._I)
        self._G.addMatrixEntry(iid, 0, 0)
        self._G.addMatrixEntry(iid, 0, self.cellCount)
        self._G.addMatrixEntry(iid, 0, self.cellCount * 2)
        self._G.addMatrixEntry(iid, 0, self.cellCount * 3)
        # Fix f_r
        if self.fix_poro:
            self._G.addMatrixEntry(iid, self._G.rows(), self.cellCount * 3)

    def showModel(self, model):
        fw, fi, fa, fr = self.fractions(model)

        rho = self.fpm.rho(fw, fi, fa, fr)
        s = self.fpm.slowness(fw, fi, fa, fr)

        fig, axs = plt.subplots(3, 2)
        pg.show(self.mesh, fw, ax=axs[0, 0], label="Water content", hold=True,
                logScale=False, cMap="Blues")
        pg.show(self.mesh, fi, ax=axs[1, 0], label="Ice content", hold=True,
                logScale=False, cMap="Purples")
        pg.show(self.mesh, fa, ax=axs[2, 0], label="Air content", hold=True,
                logScale=False, cMap="Greens")
        pg.show(self.mesh, fr, ax=axs[2, 1], label="Rock matrix content", hold=True,
                logScale=False, cMap="Oranges")
        pg.show(self.mesh, rho, ax=axs[0, 1], label="Rho", hold=True,
                cMap="Spectral_r")
        pg.show(self.mesh, 1 / s, ax=axs[1, 1], label="Velocity")

    def showFit(self, model):
        resp = self.response(model)

        fig, (ax1, ax2) = plt.subplots(1, 2)
        self.RST.showData(response=resp[:self.RST.dataContainer.size()],
                          ax=ax1)
        resprhoa = resp[self.RST.dataContainer.size():]

        fit = (self.ERT.data("rhoa") - resprhoa) / resprhoa * 100
        lim = np.max(np.abs(fit))
        pb.show(self.ERT.data, vals=fit, cMin=-lim, cMax=lim, label="Relative fit",
                cMap="RdBu_r", ax=ax2)
        fig.show()

    def response(self, model):
        return self.response_mt(model)

    def response_mt(self, model, i=0):
        model = np.nan_to_num(model)
        fw, fi, fa, fr = self.fractions(model)

        rho = self.fpm.rho(fw, fi, fa, fr)
        s = self.fpm.slowness(fw, fi, fa, fr)

        print("=" * 60)
        print("       Min. / Max.")
        print("Water: %.2f / %.2f" % (np.min(fw), np.max(fw)))
        print("Ice:   %.2f / %.2f" % (np.min(fi), np.max(fi)))
        print("Air:   %.2f / %.2f" % (np.min(fa), np.max(fa)))
        print("Rock:  %.2f / %.2f" % (np.min(fr), np.max(fr)))
        print("-" * 60)
        print("SUM:   %.2f / %.2f" % (np.min(fa + fw + fi + fr), np.max(fa + fw + fi + fr)))
        print("=" * 60)
        print("Rho:   %.2e / %.2e" % (np.min(rho), np.max(rho)))
        print("Vel:   %d / %d" % (np.min(1 / s), np.max(1 / s)))

        t = self.RST.fop.response(s)
        rhoa = self.ERT.fop.response(rho)

        return pg.cat(t, rhoa)
