#############################################
# to find "invlib" in the main folder
import sys, os
path = os.popen("git rev-parse --show-toplevel").read().strip("\n")
sys.path.insert(0, path)
#############################################

from invlib import FourPhaseModel

# Inversion settings
zWeight = 0.25 # four times more smoothing in lateral direction
erte = 0.03 # 3 %
rste = 0.0003 # 0.3 ms
maxIter = 50 # maximum number of iterations

# Petrophysical settings
poro = 0.53 # porosity
phi = poro
fpm = FourPhaseModel(phi=poro, va=300., vi=3500., vw=1500, m=1.4, n=2.4,
                     rhow=60, vr=6000)
