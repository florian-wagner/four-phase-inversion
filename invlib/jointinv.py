import pygimli as pg

from .lsqrinversion import LSQRInversion


class JointInv(LSQRInversion):
    def __init__(self, fop, data, error, startmodel, lam=20, beta=10000,
                 maxIter=50, fwmin=0, fwmax=1, fimin=0, fimax=1, famin=0,
                 famax=1, frmin=0, frmax=1):
        LSQRInversion.__init__(self, data, fop, verbose=True, dosave=True)
        self._error = pg.RVector(error)

        # Set data transformations
        self.logtrans = pg.RTransLog()
        self.trans = pg.RTrans()
        self.dcumtrans = pg.RTransCumulative()
        self.dcumtrans.add(self.trans,
                           self.forwardOperator().RST.dataContainer.size())
        self.dcumtrans.add(self.logtrans,
                           self.forwardOperator().ERT.data.size())
        self.setTransData(self.dcumtrans)

        # Set model transformation
        n = self.forwardOperator().cellCount
        self.mcumtrans = pg.TransCumulative()
        self.transforms = []
        phase_limits = [[fwmin, fwmax], [fimin, fimax],
                        [famin, famax], [frmin, frmax]]
        for i, (lower, upper) in enumerate(phase_limits):
            if lower == 0:
                lower = 0.001
            self.transforms.append(pg.RTransLogLU(lower, upper))
            self.mcumtrans.add(self.transforms[i], n)

        self.setTransModel(self.mcumtrans)

        # Set error
        self.setRelativeError(self._error)

        # Set some defaults

        # Set maximum number of iterations (default is 20)
        self.setMaxIter(maxIter)

        # Regularization strength
        self.setLambda(lam)
        self.setDeltaPhiAbortPercent(0.25)

        fop = self.forwardOperator()
        fop.createConstraints()  # Important!
        ones = pg.RVector(fop._I.rows(), 1.0)
        phiVec = pg.cat(ones, startmodel)
        self.setParameterConstraints(fop._G, phiVec, beta)
        self.setModel(startmodel)
