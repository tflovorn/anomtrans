from __future__ import division

def _get_coeffs(sizes):
    ncomp = len(sizes)
    coeffs = []
    for d in range(ncomp):
        coeff = 1
        for dc in range(d):
            coeff *= sizes[dc]
        
        coeffs.append(coeff)

    return coeffs

def _get_end_iall(sizes):
    end_iall = 1
    for size_i in sizes:
        end_iall *= size_i

    return end_iall

class GridBasis:
    def __init__(self, sizes):
        if len(sizes) <= 0:
            raise ValueError("must have at least one element of sizes")

        self.sizes = sizes
        self.coeffs = _get_coeffs(sizes)
        self.end_iall = _get_end_iall(sizes)

    def ncomp(self):
        return len(self.sizes)

    def decompose(self, iall):
        comps = [int(iall % self.sizes[0])]

        prev = iall
        for d in range(1, self.ncomp()):
            new_residual = (prev - comps[d-1]) / self.sizes[d-1]
            comps.append(int(new_residual % self.sizes[d]))
            prev = new_residual

        return comps

    def compose(self, components):
        total = 0
        for d in range(self.ncomp()):
            total += self.coeffs[d] * components[d]

        return total

    def add(self, iall, Delta):
        comps = self.decompose(iall)
        new_comps = []
        for d in range(self.ncomp()):
            new_comps.append((comps[d] + Delta[d]) % self.sizes[d])

        return self.compose(new_comps)

def _corresponding_GridBasis(Nk, Nbands):
    sizes = []
    for Nk_i in Nk:
        sizes.append(Nk_i)

    sizes.append(Nbands)
    return GridBasis(sizes)

class kmBasis:
    def __init__(self, Nk, Nbands):
        self.Nk = Nk
        self.Nbands = Nbands
        self.gb = _corresponding_GridBasis(Nk, Nbands)
        self.end_ikm = self.gb.end_iall

    def k_dim(self):
        return len(self.Nk)

    def decompose(self, ikm):
        all_comps = self.gb.decompose(ikm)
        iks = []
        for d in range(self.k_dim()):
            iks.append(all_comps[d])

        im = all_comps[self.k_dim()]
        return iks, im

    def compose(self, ikm_comps):
        all_comps = []
        for d in range(self.k_dim()):
            all_comps.append(ikm_comps[0][d])

        all_comps.append(ikm_comps[1])
        return self.gb.compose(all_comps)

    def add(self, ikm, Delta_k):
        Delta_km = []
        for d in range(self.k_dim()):
            Delta_km.append(Delta_k[d])

        Delta_km.append(0)
        return self.gb.add(ikm, Delta_km)

def km_at(Nk, ikm_comps):
    ks = []
    for d in range(len(Nk)):
        kd = ikm_comps[0][d] / Nk[d]
        ks.append(kd)

    km = (ks, ikm_comps[1])
    return km
