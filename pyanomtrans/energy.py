def get_energies(kmb, H):
    Ekm = []

    for ikm in range(kmb.end_ikm):
        ikm_comps = kmb.decompose(ikm)
        energy = H.energy(ikm_comps)
        Ekm.append(energy)

    return Ekm
