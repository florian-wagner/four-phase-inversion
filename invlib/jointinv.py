import pygimli as pg
from .lsqrinversion import LSQRInversion

class JointInv(LSQRInversion):
    def __init__(self, fop, data, error):
        LSQRInversion.__init__(self, data, fop, verbose=True, dosave=True)
        self._error = pg.RVector(error)
        self.setDefaults()

    def setDefaults(self):
        # Set data transformations
        self.logtrans = pg.RTransLog()
        self.trans = pg.RTrans()
        self.cumtrans = pg.RTransCumulative()
        self.cumtrans.add(self.trans, self.forwardOperator().RST.dataContainer.size())
        self.cumtrans.add(self.logtrans, self.forwardOperator().ERT.data.size())
        self.setTransData(self.cumtrans)
        # self.setTransData(self.logtrans)

        # Set model transformation
        self.modtrans = pg.RTransLogLU(0, 1)
        self.setTransModel(self.modtrans)

        # Set error
        self.setRelativeError(self._error)

        # Set some defaults

        # Set maximum number of iterations (default is 20)
        self.setMaxIter(50)

        # Regularization strength
        self.setLambda(20)
        self.setDeltaPhiAbortPercent(1)

        self.forwardOperator().createConstraints() # Important!
        ones = pg.RVector(self.forwardOperator()._I.rows(), 1.0)
        phiVec = pg.cat(ones, ones - self.forwardOperator().fpm.phi)
        self.setParameterConstraints(self.forwardOperator()._G, phiVec, 10000)
