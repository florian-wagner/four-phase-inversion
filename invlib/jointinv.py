import pygimli as pg
from .lsqrinversion import LSQRInversion

class JointInv(LSQRInversion):
    def __init__(self, fop, data, error, lam=20, maxIter=50, frmin=0, frmax=1):
        LSQRInversion.__init__(self, data, fop, verbose=True, dosave=True)
        self._error = pg.RVector(error)

        # Set data transformations
        self.logtrans = pg.RTransLog()
        self.trans = pg.RTrans()
        self.dcumtrans = pg.RTransCumulative()
        self.dcumtrans.add(self.trans, self.forwardOperator().RST.dataContainer.size())
        self.dcumtrans.add(self.logtrans, self.forwardOperator().ERT.data.size())
        self.setTransData(self.dcumtrans)

        # Set model transformation
        self.mcumtrans = pg.TransCumulative()
        self.modtrans = pg.RTransLogLU(0.0001, 1.0)

        n = self.forwardOperator().cellCount
        for i in range(3):
            self.mcumtrans.add(self.modtrans, n)

        self.phitrans = pg.RTransLogLU(frmin, frmax)
        self.mcumtrans.add(self.phitrans, n)
        self.setTransModel(self.mcumtrans)

        # Set error
        self.setRelativeError(self._error)

        # Set some defaults

        # Set maximum number of iterations (default is 20)
        self.setMaxIter(maxIter)

        # Regularization strength
        self.setLambda(lam)
        self.setDeltaPhiAbortPercent(0.25)

        self.forwardOperator().createConstraints() # Important!
        ones = pg.RVector(self.forwardOperator()._I.rows(), 1.0)
        if self.forwardOperator().fix_poro:
            phiVec = pg.cat(ones, ones - self.forwardOperator().fpm.phi)
        else:
            phiVec = ones
        self.setParameterConstraints(self.forwardOperator()._G, phiVec, 10000)
