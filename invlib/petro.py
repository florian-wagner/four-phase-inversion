import matplotlib.pyplot as plt
import numpy as np

import pygimli as pg


class FourPhaseModel():

    def __init__(self, vw=1500., va=330., vi=3500., vr=5500, a=1., n=2., m=1.3,
                 phi=0.4, rhow=150.):
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
            Velocity of rock in m/s (the default is 5000).
        a : float or array type
            Archie parameter `a` (the default is 1).
        n : float or array type
            Archie parameter `n` (the default is 2).
        m : float or array type
            Archie parameter `m` (the default is 1.3).
        phi : float or array type
            Porosity `phi` (the default is 0.4).
        rhow : float or array type
            Water resistivity `rhow` (the default is 150).
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
        self.fr = 1 - self.phi  # fraction of rock
        self.rhow = rhow

    def water(self, rho):
        fw = ((self.a * self.rhow * self.phi**self.n) /
              (rho * self.phi**self.m))**(1 / self.n)
        fw[np.isclose(fw, 0)] = 0
        return fw

    def ice(self, rho, v):
        fi = (self.vi * self.va / (self.va - self.vi) * (
            1. / v - self.fr / self.vr - self.phi / self.va - self.water(rho) *
            (1. / self.vw - 1. / self.va)))
        fi[np.isclose(fi, 0)] = 0
        return fi

    def air(self, rho, v):
        fa = ((self.vi * self.va / (self.vi - self.va) * (
            1. / v - self.fr / self.vr - self.phi / self.vi - self.water(rho) *
            (1. / self.vw - 1. / self.vi))))
        fa[np.isclose(fa, 0)] = 0
        return fa

    def rho(self, fw, fi, fa, fr=None):
        """Return electrical resistivity based on fraction of water `fw`."""
        if fr is None:
            phi = fw + fi + fa
        else:
            phi = 1 - fr

        rho = self.a * self.rhow * phi**(-self.m) * (fw / phi)**(-self.n)
        if (rho <= 0).any():
            pg.warn(
                "Found negative resistivity, setting to nearest above zero.")
            rho[rho <= 0] = np.min(rho[rho > 0])
        return rho

    def rho_deriv_fw(self, fw, fi, fa, fr):
        return self.rho(fw, fi, fa, fr) * -self.n / fw

    def rho_deriv_fr(self, fw, fi, fa, fr):
        return self.rho(fw, fi, fa, fr) * (self.n - self.m) / (fr - 1)

    def rho_deriv_fi(self, fw, fi, fa, fr):
        return 0

    def rho_deriv_fa(self, fw, fi, fa, fr):
        return 0

    def slowness(self, fw, fi, fa, fr=None):
        """Return slowness based on fraction of water `fw` and ice `fi`."""
        if fr is None:
            fr = 1 - (fw + fi + fa)

        s = fw / self.vw + fr / self.vr + fi / self.vi + fa / self.va
        if (s <= 0).any():
            pg.warn("Found negative slowness, setting to nearest above zero.")
            s[s <= 0] = np.min(s[s > 0])
        return s

    def all(self, rho, v, mask=False):
        """Syntatic sugar for all fractions including a mask for unrealistic
        values."""

        # RVectors sometimes cause segfaults
        rho = np.array(rho)
        v = np.array(v)

        fa = self.air(rho, v)
        fi = self.ice(rho, v)
        fw = self.water(rho)

        # Check that fractions are between 0 and 1
        array_mask = np.array(((fa < 0) | (fa > 1 - self.fr))
                              | ((fi < 0) | (fi > 1 - self.fr))
                              | ((fw < 0) | (fw > 1 - self.fr))
                              | ((self.fr < 0) | (self.fr > 1)))
        if array_mask.sum() > 1:
            print("WARNING: %d of %d fraction values are unphysical." % (int(
                array_mask.sum()), len(array_mask.ravel())))

        if mask:
            fa = np.ma.array(fa, mask=(fa < 0) | (fa > 1 - self.fr))
            fi = np.ma.array(fi, mask=(fi < 0) | (fi > 1 - self.fr))
            fw = np.ma.array(fw, mask=(fw < 0) | (fw > 1 - self.fr))

        return fa, fi, fw, array_mask

    def show(self, mesh, rho, vel, mask=True, **kwargs):
        fa, fi, fw, mask = self.all(rho, vel, mask=mask)

        fig, axs = plt.subplots(3, 2, figsize=(16, 10))
        pg.show(mesh, fw, ax=axs[0, 0], label="Water content", hold=True,
                logScale=False, cMap="Blues", **kwargs)
        pg.show(mesh, fi, ax=axs[1, 0], label="Ice content", hold=True,
                logScale=False, cMap="Purples", **kwargs)
        pg.show(mesh, fa, ax=axs[2, 0], label="Air content", hold=True,
                logScale=False, cMap="Greens", **kwargs)
        pg.show(mesh, rho, ax=axs[0, 1], label="Rho", hold=True,
                cMap="Spectral_r", logScale=True, **kwargs)
        pg.show(mesh, vel, ax=axs[1, 1], label="Velocity", logScale=False,
                hold=True, **kwargs)
        pg.show(mesh, self.phi, ax=axs[2, 1], label="Porosity", logScale=False,
                hold=True, **kwargs)
        for ax in axs.flat:
            ax.set_facecolor("0.5")
        return fig, axs


def testFourPhaseModel():
    # Parameters from Hauck et al. (2011)
    fpm = FourPhaseModel(vw=1500, vi=3500, va=300, vr=6000, phi=0.5, n=2.,
                         m=2., a=1., rhow=200.)

    assert fpm.water(10.0) == 10.0
    v = np.linspace(500, 6000, 1000)
    rho = np.logspace(2, 7, 1000)
    x, y = np.meshgrid(v, rho)

    fa, fi, fw, mask = fpm.all(y, x, mask=True)

    cmap = plt.cm.get_cmap('Spectral_r', 41)
    fig, axs = plt.subplots(3, figsize=(6, 4.5), sharex=True)
    labels = ["Air content", "Ice content", "Water content"]
    for data, ax, label in zip([fa, fi, fw], axs, labels):
        im = ax.imshow(
            data[::-1], cmap=cmap, extent=[
                v.min(),
                v.max(),
                np.log10(rho.min()),
                np.log10(rho.max())
            ], aspect="auto", vmin=0, vmax=0.5)
        plt.colorbar(im, ax=ax, label=label)

    axs[1].set_ylabel(r"Log resistivity ($\Omega$m)")
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
