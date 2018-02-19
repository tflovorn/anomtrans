import unittest
from pyanomtrans.grid_basis import GridBasis, kmBasis

class TestGridBasis(unittest.TestCase):
    def test_GridBasis(self):
        sizes = [4, 4, 4]
        gb = GridBasis(sizes)
        self.assertEqual(gb.end_iall, 4*4*4)

        iall = 0
        for i2 in range(sizes[2]):
            for i1 in range(sizes[1]):
                for i0 in range(sizes[0]):
                    comps = [i0, i1, i2]
                    self.assertEqual(gb.decompose(iall), comps)
                    self.assertEqual(gb.compose(comps), iall)

                    for p2 in range(-sizes[2], sizes[2]+1):
                        for p1 in range(-sizes[1], sizes[1]+1):
                            for p0 in range(-sizes[0], sizes[0]+1):
                                p = [p0, p1, p2]
                                expect = [(i0 + p0) % sizes[0],
                                          (i1 + p1) % sizes[1],
                                          (i2 + p2) % sizes[2]]

                                self.assertEqual(gb.decompose(gb.add(gb.compose(comps), p)), expect)

                    iall += 1

class TestkmBasis(unittest.TestCase):
    def test_kmBasis(self):
        Nk = [8, 4]
        Nbands = 2
        kmb = kmBasis(Nk, Nbands)
        self.assertEqual(kmb.end_ikm, 8*4*2)

        iall = 0
        for m in range(Nbands):
            for ik1 in range(Nk[1]):
                for ik0 in range(Nk[0]):
                    ik_comps = [ik0, ik1]
                    ikm_comps = (ik_comps, m)
                    k_at_comps = [ik0 / Nk[0], ik1 / Nk[1]]
                    km_at_comps = (k_at_comps, m)

                    self.assertEqual(kmb.decompose(iall), ikm_comps)
                    self.assertEqual(kmb.compose(ikm_comps), iall)
                    self.assertEqual(kmb.km_at(ikm_comps), km_at_comps)

                    for p1 in range(-Nk[1], Nk[1]+1):
                        for p0 in range(-Nk[0], Nk[0]+1):
                            Delta_k = [p0, p1]
                            kp_expect = [(ik0 + p0) % Nk[0],
                                        (ik1 + p1) % Nk[1]]
                            kpm_expect = (kp_expect, m)

                            self.assertEqual(kmb.decompose(kmb.add(kmb.compose(ikm_comps), Delta_k)), kpm_expect)

                    iall += 1

if __name__ == '__main__':
    unittest.main()
