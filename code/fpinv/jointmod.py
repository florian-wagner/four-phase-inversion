import matplotlib.pyplot as plt
import numpy as np

import pygimli as pg


class JointMod(pg.core.ModellingBase):
    def __init__(self, mesh, ertfop, rstfop, petromodel, fix_poro=True,
                 zWeight=1, verbose=True, corr_l=None, fix_water=False,
                 fix_ice=False, fix_air=False):
        """Joint petrophysical modeling operator.

        Parameters
        ----------
        mesh : pyGIMLi mesh
        ertfop : ERT forward operator
        rstfop : RST forward operator
        petromodel : Petrophysical four-phase model
        zWeight : zWeight for more or less layering
        verbose : Be more verbose
        corr_l : tuple
            Horizontal and vertical correlation lengths. If provided,
            geostatistical regularization will be used and classical smoothing
            with zWeight will be ignored.
        fix_poro|water|ice|air : boolean or vector
            Fix to starting model or provide weight vector for particular cells.
        """
        pg.core.ModellingBase.__init__(self, verbose)
        self.mesh = pg.Mesh(mesh)
        self.ERT = ertfop
        self.RST = rstfop
        self.fops = [self.RST, self.ERT]
        self.fpm = petromodel
        self.cellCount = self.mesh.cellCount()
        self.fix_water = fix_water
        self.fix_ice = fix_ice
        self.fix_air = fix_air
        self.fix_poro = fix_poro
        self.zWeight = zWeight
        # self.fix_cells = fix_cells
        self.corr_l = corr_l
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
        self.jacRSTW = pg.matrix.MultRightMatrix(jacRST, r=1. / self.fpm.vw)
        self.jacRSTI = pg.matrix.MultRightMatrix(jacRST, r=1. / self.fpm.vi)
        self.jacRSTA = pg.matrix.MultRightMatrix(jacRST, r=1. / self.fpm.va)
        self.jacRSTR = pg.matrix.MultRightMatrix(jacRST, r=1. / self.fpm.vr)

        self.jacERTW = pg.matrix.MultRightMatrix(
            jacERT, r=self.fpm.rho_deriv_fw(fw, fi, fa, fr))
        self.jacERTI = pg.matrix.MultRightMatrix(
            jacERT, r=self.fpm.rho_deriv_fi(fw, fi, fa, fr))
        self.jacERTA = pg.matrix.MultRightMatrix(
            jacERT, r=self.fpm.rho_deriv_fa(fw, fi, fa, fr))
        self.jacERTR = pg.matrix.MultRightMatrix(
            jacERT, r=self.fpm.rho_deriv_fr(fw, fi, fa, fr))

        # Putting subjacobians together in block matrix
        self.jac = pg.matrix.BlockMatrix()
        nData = 0

        jacsRST = [self.jacRSTW, self.jacRSTI, self.jacRSTA, self.jacRSTR]
        jacsERT = [self.jacERTW, self.jacERTI, self.jacERTA, self.jacERTR]

        for jaclist in jacsRST, jacsERT:
            for i in range(4):
                idx = self.jac.addMatrix(jaclist[i])
                self.jac.addMatrixEntry(idx, nData, self.cellCount * i)
            nData += self.RST.fop.data.size()  # update total vector length
        self.setJacobian(self.jac)

    def createConstraints(self):
        # First order smoothness matrix
        self._Ctmp = pg.matrix.SparseMapMatrix()

        if self.corr_l is None:
            pg.info("Using smoothing with zWeight = %.2f." % self.zWeight)
            rm = self.RST.fop.regionManager()
            rm.fillConstraints(self._Ctmp)

            # Set zWeight
            rm.setZWeight(self.zWeight)
            self.cWeight = pg.Vector()
            rm.fillConstraintWeights(self.cWeight)
            self._CW = pg.matrix.LMultRMatrix(self._Ctmp, self.cWeight)
        else:
            pg.info("Using geostatistical constraints with " + str(self.corr_l))
            # Geostatistical constraints by Jordi et al., GJI, 2018
            CM = pg.utils.geostatistics.covarianceMatrix(self.mesh, I=self.corr_l)
            self._Ctmp = pg.matrix.Cm05Matrix(CM)
            self._CW = self._Ctmp

        # Putting together in block matrix
        self._C = pg.matrix.BlockMatrix()
        cid = self._C.addMatrix(self._CW)
        self._C.addMatrixEntry(cid, 0, 0)
        self._C.addMatrixEntry(cid, self._Ctmp.rows(), self.cellCount)
        self._C.addMatrixEntry(cid, self._Ctmp.rows() * 2, self.cellCount * 2)
        self._C.addMatrixEntry(cid, self._Ctmp.rows() * 3, self.cellCount * 3)
        self.setConstraints(self._C)

        # Identity matrix for interparameter regularization
        self._I = pg.matrix.IdentityMatrix(self.cellCount)

        self._G = pg.matrix.BlockMatrix()
        iid = self._G.addMatrix(self._I)
        self._G.addMatrixEntry(iid, 0, 0)
        self._G.addMatrixEntry(iid, 0, self.cellCount)
        self._G.addMatrixEntry(iid, 0, self.cellCount * 2)
        self._G.addMatrixEntry(iid, 0, self.cellCount * 3)

        self.fix_val_matrices = {}
        # Optionally fix phases to starting model globally or in selected cells
        phases = ["water", "ice", "air", "rock matrix"]
        for i, phase in enumerate([self.fix_water, self.fix_ice, self.fix_air,
                                   self.fix_poro]):
            name = phases[i]
            vec = pg.Vector(self.cellCount)
            if phase is True:
                pg.info("Fixing %s content globally." % name)
                vec += 1.0
            elif hasattr(phase, "__len__"):
                pg.info("Fixing %s content at selected cells." % name)
                phase = np.asarray(phase, dtype="int")
                vec[phase] = 1.0
            self.fix_val_matrices[name] = pg.matrix.DiagonalMatrix(vec)
            mat = self._G.addMatrix(self.fix_val_matrices[name])
            self._G.addMatrixEntry(mat, self._G.rows(), self.cellCount * i)

    def showModel(self, model):
        fw, fi, fa, fr = self.fractions(model)

        rho = self.fpm.rho(fw, fi, fa, fr)
        s = self.fpm.slowness(fw, fi, fa, fr)

        _, axs = plt.subplots(3, 2)
        pg.show(self.mesh, fw, ax=axs[0, 0], label="Water content", hold=True,
                logScale=False, cMap="Blues")
        pg.show(self.mesh, fi, ax=axs[1, 0], label="Ice content", hold=True,
                logScale=False, cMap="Purples")
        pg.show(self.mesh, fa, ax=axs[2, 0], label="Air content", hold=True,
                logScale=False, cMap="Greens")
        pg.show(self.mesh, fr, ax=axs[2, 1], label="Rock matrix content",
                hold=True, logScale=False, cMap="Oranges")
        pg.show(self.mesh, rho, ax=axs[0, 1], label="Rho", hold=True,
                cMap="Spectral_r")
        pg.show(self.mesh, 1 / s, ax=axs[1, 1], label="Velocity")

    def showFit(self, model):
        resp = self.response(model)

        fig, axs = plt.subplots(2, 2, figsize=(10, 10))
        t_resp = resp[:self.RST.fop.data.size()]
        rhoa_resp = resp[self.RST.fop.data.size():]
        self.RST.showData(response=t_resp, ax=axs[0, 0])

        t_fit = t_resp - self.RST.fop.data("t")
        lim = np.max(np.abs(t_fit))
        axs[0, 0].set_title("Traveltime curves with fit")
        axs[1, 0].set_title("Deviation between traveltimes")
        self.RST.showVA(vals=t_fit, ax=axs[1, 0], cMin=-lim, cMax=lim,
                        cmap="RdBu_r")

        rhoa_fit = (self.ERT.fop.data("rhoa") - rhoa_resp) / rhoa_resp * 100
        lim = np.max(np.abs(rhoa_fit))
        # pb.show(self.ERT.data, ax=axs[0, 1], label=r"Measured data $\rho_a$")
        # pb.show(self.ERT.data, vals=rhoa_fit, cMin=-lim, cMax=lim,
        #         label="Relative fit (%%)", cMap="RdBu_r", ax=axs[1, 1])
        fig.show()
        return fig

    def ERTchi2(self, model, error):  # chi2 and relative rms for the rhoa data
        resp = self.response(model)
        resprhoa = resp[self.RST.fop.data.size():]
        rhoaerr = error[self.RST.fop.data.size():]
        chi2rhoa = pg.utils.chi2(self.ERT.fop.data("rhoa"), resprhoa, rhoaerr)
        rmsrhoa = pg.utils.rrms(self.ERT.fop.data("rhoa"), resprhoa)
        return chi2rhoa, rmsrhoa

    def RSTchi2(self, model, error,
                data):  # chi2 and relative rms for the travel time data
        resp = self.response(model)
        resptt = resp[:self.RST.fop.data.size()]
        tterr = error[:self.RST.fop.data.size()]
        chi2tt = pg.utils.chi2(data, resptt, tterr)
        rmstt = np.sqrt(np.mean((resptt - data)**2))
        return chi2tt, rmstt

    def response(self, model):
        print(self.RST.fop.mesh())
        print(self.ERT.fop.mesh())
        return self.response_mt(model)

    def response_mt(self, model, i=0):
        model = np.nan_to_num(model)
        fw, fi, fa, fr = self.fractions(model)

        rho = self.fpm.rho(fw, fi, fa, fr)
        s = self.fpm.slowness(fw, fi, fa, fr)

        print("=" * 30)
        print("        Min. | Max.")
        print("-" * 30)
        print(" Water: %.2f | %.2f" % (np.min(fw), np.max(fw)))
        print(" Ice:   %.2f | %.2f" % (np.min(fi), np.max(fi)))
        print(" Air:   %.2f | %.2f" % (np.min(fa), np.max(fa)))
        print(" Rock:  %.2f | %.2f" % (np.min(fr), np.max(fr)))
        print("-" * 30)
        print(" SUM:   %.2f | %.2f" % (np.min(fa + fw + fi + fr),
                                       np.max(fa + fw + fi + fr)))
        print("=" * 30)
        print(" Rho:   %.2e | %.2e" % (np.min(rho), np.max(rho)))
        print(" Vel:   %d | %d" % (np.min(1 / s), np.max(1 / s)))

        t = self.RST.fop.response(s)
        rhoa = self.ERT.fop.response(rho)

        return pg.cat(t, rhoa)
