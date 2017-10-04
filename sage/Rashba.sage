qx, qy = var('qx qy', domain='real')
t0, tr = var('t0 tr', domain='real')

# <up|u_k^{+}> = <down|u_k^{+}>
def up_component(kx, ky):
    return (-I/sqrt(2)) * (I * sin(kx) + sin(ky)) / sqrt(sin(kx)**2 + sin(ky)**2)

# Eigenvector matrix [U_k]_{\sigma, m} = <sigma|km>
def evecs(kx, ky):
    return matrix([[up_component(kx, ky), up_component(kx, ky)],
                   [-I/sqrt(2), I/sqrt(2)]])

# Hamiltonian (up, down basis).
def H(kx, ky):
    H0 = 2 * t0 * (2 - cos(kx) - cos(ky))
    Hr = 2 * tr * (I*sin(kx) + sin(ky))

    return matrix([[H0, Hr],
                   [conjugate(Hr), H0]])

# Gradient of the Hamiltonian in (up, down) basis.
def grad_ud(kx, ky):
    return [derivative(H(kx=qx, ky=qy), qx)(qx=kx, qy=ky),
            derivative(H(kx=qx, ky=qy), qy)(qx=kx, qy=ky)]

# Gradient of the Hamiltonian in the eigenbasis given by evecs().
def grad_eig(kx, ky):
    U = evecs(kx, ky)
    dHx, dHy = grad_ud(kx, ky)
    return [U.conjugate().T * dHx * U,
            U.conjugate().T * dHy * U]

# E^{-}(k) - E^{+}(k) = -2|Hr|
def ediff(kx, ky):
    return -2*abs(H(kx, ky)[0, 1])

# Berry connection in the eigenbasis, {+, -} element.
def berry_pm(kx, ky):
    grad = grad_eig(kx, ky)
    coeff = I / ediff(kx, ky)
    return [coeff * grad[0], coeff * grad[1]]

print(grad_eig(qx, qy)[0][0, 1].full_simplify())
print(grad_eig(qx, qy)[1][0, 1].full_simplify())

print(berry_pm(qx, qy)[0][0, 1].full_simplify())
print(berry_pm(qx, qy)[1][0, 1].full_simplify())
