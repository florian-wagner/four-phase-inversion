import matplotlib.pyplot as plt
import numpy as np

import pybert as pb
import pygimli as pg
import pygimli.meshtools as mt
from reproduce_pellet import depth_5000, depth_5198

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
ertData.set("err", pg.RVector(ertData.size(), 0.02))
ertData.save("ert_filtered.data")

rstData.set("err", pg.RVector(rstData.size(), 0.0005))
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

plc = mt.createParaMeshPLC(combinedSensors, paraDX=0.1, boundary=4,
                           paraDepth=12, paraBoundary=3, paraMaxCellSize=0.3)


radius = 2.
for x, depth in zip([10, 26], [depth_5198, depth_5000]):
    start = plc.createNode(x - radius, -depth, 0.0)
    end = plc.createNode(x + radius, -depth, 0.0)
    plc.createEdge(start, end, marker=1)
    plc.addRegionMarker([x, -12], 2, 0.8)

mesh = mt.createMesh(plc, quality=33.8)
mesh.save("mesh.bms")

# Extract inner domain where parameters should be estimated.
# Outer domain is only needed for ERT forward simulation,
# not for seismic traveltime calculations.
paraDomain = pg.Mesh(2)
paraDomain.createMeshByMarker(mesh, 2)
pg.show(paraDomain)
paraDomain.save("paraDomain.bms")

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
