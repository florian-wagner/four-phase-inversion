import matplotlib.pyplot as plt
import numpy as np


class FourPhaseModel():
    def __init__(self, vw=1500., va=330., vi=3500., vr=6000, a=2., n=1., m=1.,
                 phi=0.4, rhow=100.):
        """Four phase model (4PM) after Hauck et al. (2011). Estimates fraction
        of ice, air and water from electrical bulk resistivity and seismic
        velocity.

        Parameters
        ----------
        vw : float or array type
            Velocity of water in m/s (the default is 1500.).
        va : float or array type
            Velocity of air in m/s (the default is 330.).
        vi : float or array type
            Velocity of ice in m/s (the default is 3500.).
        vr : float or array type
            Velocity of rock in m/s (the default is 6000).
        a : float or array type
            Archie parameter `a` (the default is 2).
        n : float or array type
            Archie parameter `n` (the default is 1).
        m : float or array type
            Archie parameter `m` (the default is 1).
        phi : float or array type
            Porosity `phi` (the default is 0.4).
        rhow : float or array type
            Water resistivity `rhow` (the default is 100).
        """

        # Velocities of water, air, ice and rock (m/s)
        self.vw = vw
        self.va = va
        self.vi = vi
        self.vr = vr

        # Archie parameter
        self.a = a
        self.m = m
        self.n = n
        self.phi = phi
        self.fr = 1 - phi  # fraction of rock
        self.rhow = rhow

    def water(self, rho):
        fw = ((self.a * self.rhow * self.phi**self.n) /
              (rho * self.phi**self.m))**1 / self.n
        return fw

    def ice(self, rho, v):
        fi = (self.vi * self.va / (self.va - self.vi) * (
            1. / v - self.fr / self.vr - self.phi / self.va - self.water(rho) *
            (1. / self.vw - 1. / self.va)))
        return fi

    def air(self, rho, v):
        # fa = ((self.vi * self.va / (self.vi - self.va) * (
        #     1. / v - self.fr / self.vr - self.phi / self.vi - self.water(rho) *
        #     (1. / self.vw - 1. / self.vi))))
        fa = 1 - self.fr - self.ice(rho, v) - self.water(rho)
        return fa

    def all(self, rho, v, mask=False):
        """ Syntatic sugar for all fractions including a mask for unrealistic values. """
        fa = self.air(rho, v)
        fi = self.ice(rho, v)
        fw = self.water(rho)

        # Check that fractions are between 0 and 1
        array_mask = np.array(
            ((fa < 0) | (fa > 1)) |
            ((fi < 0) | (fi > 1)) |
            ((fw < 0) | (fw > 1))
        )
        if array_mask.sum() > 1:
            print("WARNING: %d of %d fraction values outside 0-1 range." %
                  (int(array_mask.sum()), len(array_mask.ravel())))
        if mask:
            fa = np.ma.array(fa, mask=array_mask)
            fi = np.ma.array(fi, mask=array_mask)
            fw = np.ma.array(fw, mask=array_mask)

        return fa, fi, fw, array_mask


def testFourPhaseModel():
    # Parameters from Hauck et al. (2011)
    fpm = FourPhaseModel(vw=1500, vi=3500, va=300, vr=6000, phi=0.5, n=2.,
                         m=2., a=1., rhow=200.)

    assert fpm.water(10.0) == 10.0
    v = np.linspace(500, 6000, 1000)
    rho = np.logspace(2, 7, 1000)
    x, y = np.meshgrid(v, rho)

    fa, fi, fw, mask = fpm.all(y, x, mask=True)

    cmap = plt.cm.get_cmap('Spectral_r', 41)    # 11 discrete colors
    fig, axs = plt.subplots(3, figsize=(6,4.5), sharex=True)
    labels = ["Air content", "Ice content", "Water content"]
    for data, ax, label in zip([fa, fi, fw], axs, labels):
        im = ax.imshow(data[::-1], cmap=cmap, extent=[
            v.min(), v.max(), np.log10(rho.min()), np.log10(rho.max())
        ], aspect="auto", vmin=0, vmax=0.5)
        cb = plt.colorbar(im, ax=ax, label=label)

    axs[1].set_ylabel("Log resistivity ($\Omega$m)")
    axs[-1].set_xlabel("Velocity (m/s)")

    fig.tight_layout()

    plt.figure()
    im = plt.imshow(fa + fi + fw, vmin=0, vmax=0.5)
    plt.colorbar(im)

    return fig

if __name__ == '__main__':
    import seaborn
    seaborn.set(font="Fira Sans", style="ticks")
    plt.rcParams["image.cmap"] = "viridis"
    fig = testFourPhaseModel()
    fig.savefig("4PM_value_range.pdf")
