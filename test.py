import pygimli as pg

mesh = pg.createGrid(x=[0., 1., 2.], y=[0., 1., 2.])
pg.show(mesh)

class JointMod(pg.ModellingBase):
    def __init__(self, mesh, verbose=True):
        pg.ModellingBase.__init__(self, verbose)
        self.meshlist = []
        for i in range(2):
            for cell in mesh.cells():
                cell.setMarker(i + 1)
            self.meshlist.append(pg.Mesh(mesh))
            self.regionManager().addRegion(i + 1, self.meshlist[i])
            self.regionManager().region(i + 1).setConstraintType(1)

        self.mesh = self.meshlist[0]
        self.cellCount = self.mesh.cellCount()
        self.createConstraints()

JM = JointMod(mesh)

pg.solver.showSparseMatrix(JM.constraints(), full=True)
pg.utils.boxprint("Bis hier hin alles gut")

inv = pg.Inversion(JM)

# Set homogeneous starting model of f_ice, f_water, f_air = phi/3
n = JM.regionManager().parameterCount()
startmodel = pg.RVector(n, 0.5)
inv.setModel(startmodel)

inv.setMarquardtScheme(0.9)

inv.fop().createConstraints()
inv.checkConstraints()
pg.solver.showSparseMatrix(inv.fop().constraints(), full=True)
