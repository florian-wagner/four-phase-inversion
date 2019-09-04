import numpy as np

import pybert as pb
import pygimli as pg
import pygimli.meshtools as mt
from reproduce_pellet import depth_5000, depth_5198
from settings import erte, rste

ertData = pb.load("ert.data")

print("Number of electrodes:", ertData.sensorCount())
print(ertData)

rstData = pg.DataContainer("rst.data", "s g")
print("Number of shot/receivers:", rstData.sensorCount())
maxrst = pg.max(pg.x(rstData.sensors()))

idx = []
for i, sensor in enumerate(ertData.sensors()):
    if sensor[0] >= 50.0:
        idx.append(i)

ertData.removeSensorIdx(idx)
ertData.removeInvalid()
ertData.removeUnusedSensors()
ertData.set("err", pg.RVector(ertData.size(), erte))
ertData.save("ert_filtered.data")

rstData.set("err", pg.RVector(rstData.size(), rste))
#
# # Remove two data points with high v_a at zero-offset
# Calculate offset
px = pg.x(rstData.sensorPositions())
gx = np.array([px[int(g)] for g in rstData("g")])
sx = np.array([px[int(s)] for s in rstData("s")])
offset = np.absolute(gx - sx)
va = offset / rstData("t")
rstData.markInvalid((offset < 5) & (va > 800))
#
# # Remove shot 27, too high apparent velocities
rstData.markInvalid(np.isclose(rstData("s"), 27))
rstData.markInvalid(210) # outlier
rstData.removeInvalid()
rstData.save("rst_filtered.data")
rstData = pg.DataContainer("rst_filtered.data", "s g")
#########################
ertData = pb.load("ert_filtered.data")
print(ertData)

print(len(ertData.sensorPositions()))
for pos in ertData.sensorPositions():
    print(pos)
# %%


def is_close(pos, data, tolerance=0.1):
    for posi in data.sensorPositions():
        dist = pos.dist(posi)
        if dist <= tolerance:
            return True
    return False


combinedSensors = pg.DataContainer()
for pos in ertData.sensorPositions():
    combinedSensors.createSensor(pos)

for pos in rstData.sensorPositions():
    if is_close(pos, ertData):
        print("Not adding", pos)
    else:
        combinedSensors.createSensor(pos)

combinedSensors.sortSensorsX()
x = pg.x(combinedSensors.sensorPositions()).array()
z = pg.z(combinedSensors.sensorPositions()).array()

np.savetxt("sensors.npy", np.column_stack((x, z)))

print("Number of combined positions:", combinedSensors.sensorCount())
print(combinedSensors)
# %%

for case in 1, 2:
    if case == 2:
        p1 = [8., 0, -0.03]
        p2 = [12., 0, -0.23]
        p3 = [24., 0, -0.37]
        p4 = [28., 0, -0.69]
        for p in p1, p2, p3, p4:
            combinedSensors.createSensor(p)
        combinedSensors.sortSensorsX()

    plc = mt.createParaMeshPLC(combinedSensors, paraDX=0.15, boundary=4,
                               paraDepth=12, paraBoundary=3, paraMaxCellSize=0.3)

    if case == 2:
        box = pg.Mesh(2)
        radius = 2.
        points = [(p1, p2), (p3, p4)]
        for x, depth, pts in zip([10., 26.], [depth_5198, depth_5000], points):
            start = plc.createNode(x - radius, -depth, 0.0)
            ul = plc.createNode(pts[0][0], pts[0][2], 0.0)
            b = plc.createEdge(start, ul, marker=20)
            box.copyBoundary(b)

            end = plc.createNode(x + radius, -depth, 0.0)
            ur = plc.createNode(pts[1][0], pts[1][2], 0.0)
            b = plc.createEdge(end, ur, marker=20)
            plc.createEdge(start, end, marker=1)
            box.copyBoundary(b)
            plc.addRegionMarker([x, -1], 2, 0.5)

        box.save("box.bms")

    for x in [10., 26.]:
        plc.addRegionMarker([x, -12.0], 2, 0.5)

    mesh = mt.createMesh(plc, quality=33.8)

    # Set vertical boundaries of box to zero to allow lateral smoothing
    if case == 2:
        for bound in mesh.boundaries():
            if bound.marker() == 20:
                bound.setMarker(0)

    # mesh.save("mesh_%s.bms" % case)

    # Extract inner domain where parameters should be estimated.
    # Outer domain is only needed for ERT forward simulation,
    # not for seismic traveltime calculations.
    paraDomain = pg.Mesh(2)
    paraDomain.createMeshByMarker(mesh, 2)
    # paraDomain.save("paraDomain_%s.bms" % case)

    # if case == 2:
    #     pg.show(paraDomain)
    #     pg.wait()

# fig, ax = plt.subplots(figsize=(10, 6))
# pg.show(mesh, showMesh=True, markers=True, ax=ax, hold=True)
# ax.set_xlim(x[0] - 10, x[-1] + 10)
# ax.set_ylim(-45, max(z) + 5)
# ax.plot(pg.x(ertData.sensorPositions()), pg.z(ertData.sensorPositions()), "ro",
#         ms=3, label="Electrodes")
# ax.plot(pg.x(rstData.sensorPositions()), pg.z(rstData.sensorPositions()), "bv",
#         ms=3, label="Geophones")
# ax.legend()
# fig.savefig("mesh_with_sensors.png")
