"""
3D fixed-source Monte Carlo transport using a "quantum state" bit-vector per particle.

- Particle state is a list of 0/1 bits.
- Encodes position, direction, and PRNG (64-bit) inside the bit-vector.
- Transport is in an axis-aligned box partitioned into a Cartesian mesh.
- Material is piecewise constant per cell: Sigma_t, Sigma_s (absorption is Sigma_a = Sigma_t - Sigma_s).
- Distance to collision: Exp(Sigma_t(cell)).
- Distance to boundary: computed to nearest cell face along direction.
- Advance either to collision or to boundary crossing.
- Tally track-length estimator of scalar flux per cell:
      flux[cell] += track_length_in_cell
  (You can normalize by volume and number of source particles as needed.)

No fission yet; fixed source only. This is intended as the base for a future k-eigenvalue extension.
"""

from __future__ import annotations
import math
from dataclasses import dataclass
from typing import List, Tuple

# ----------------------------
# Bit utilities
# ----------------------------

def bits_to_uint(bits: List[int]) -> int:
    v = 0
    for i, b in enumerate(bits):
        v |= (int(b) & 1) << i
    return v

def uint_to_bits(v: int, nbits: int) -> List[int]:
    return [(v >> i) & 1 for i in range(nbits)]

def bits_to_sint(bits: List[int]) -> int:
    n = len(bits)
    u = bits_to_uint(bits)
    sign = 1 << (n - 1)
    return u - (1 << n) if (u & sign) else u

def sint_to_bits(x: int, nbits: int) -> List[int]:
    mask = (1 << nbits) - 1
    return uint_to_bits(x & mask, nbits)

# ----------------------------
# PRNG (xorshift64*)
# ----------------------------

def xorshift64star(state: int) -> int:
    x = state & 0xFFFFFFFFFFFFFFFF
    x ^= (x >> 12) & 0xFFFFFFFFFFFFFFFF
    x ^= (x << 25) & 0xFFFFFFFFFFFFFFFF
    x ^= (x >> 27) & 0xFFFFFFFFFFFFFFFF
    x = (x * 2685821657736338717) & 0xFFFFFFFFFFFFFFFF
    return x

def u01_from_u64(x: int) -> float:
    top53 = (x >> 11) & ((1 << 53) - 1)
    return (top53 + 1) / (2**53 + 1)  # strictly in (0,1)

# ----------------------------
# State layout
# ----------------------------

@dataclass(frozen=True)
class StateLayout:
    pos_bits: int = 20   # per coordinate
    dir_bits: int = 18   # per direction component (signed)

    @property
    def total_bits(self) -> int:
        return 3*self.pos_bits + 3*self.dir_bits + 64

    def slices(self):
        i = 0
        x = (i, i+self.pos_bits); i += self.pos_bits
        y = (i, i+self.pos_bits); i += self.pos_bits
        z = (i, i+self.pos_bits); i += self.pos_bits

        ux = (i, i+self.dir_bits); i += self.dir_bits
        uy = (i, i+self.dir_bits); i += self.dir_bits
        uz = (i, i+self.dir_bits); i += self.dir_bits

        rng = (i, i+64); i += 64
        return x, y, z, ux, uy, uz, rng

# ----------------------------
# Geometry and mesh
# ----------------------------

@dataclass(frozen=True)
class BoxDomain:
    xmin: float; xmax: float
    ymin: float; ymax: float
    zmin: float; zmax: float

    def contains(self, r: Tuple[float, float, float]) -> bool:
        x, y, z = r
        return (self.xmin <= x <= self.xmax and
                self.ymin <= y <= self.ymax and
                self.zmin <= z <= self.zmax)

@dataclass(frozen=True)
class Mesh:
    nx: int; ny: int; nz: int
    dom: BoxDomain

    @property
    def dx(self) -> float: return (self.dom.xmax - self.dom.xmin) / self.nx
    @property
    def dy(self) -> float: return (self.dom.ymax - self.dom.ymin) / self.ny
    @property
    def dz(self) -> float: return (self.dom.zmax - self.dom.zmin) / self.nz

    def cell_index(self, r: Tuple[float, float, float]) -> Tuple[int, int, int]:
        x, y, z = r
        # Clamp to interior indexing; leakage handled separately
        i = min(self.nx-1, max(0, int((x - self.dom.xmin) / self.dx)))
        j = min(self.ny-1, max(0, int((y - self.dom.ymin) / self.dy)))
        k = min(self.nz-1, max(0, int((z - self.dom.zmin) / self.dz)))
        return i, j, k

    def flat(self, i: int, j: int, k: int) -> int:
        return i + self.nx*(j + self.ny*k)

    def cell_bounds(self, i: int, j: int, k: int) -> Tuple[float, float, float, float, float, float]:
        x0 = self.dom.xmin + i*self.dx
        x1 = x0 + self.dx
        y0 = self.dom.ymin + j*self.dy
        y1 = y0 + self.dy
        z0 = self.dom.zmin + k*self.dz
        z1 = z0 + self.dz
        return x0, x1, y0, y1, z0, z1

# ----------------------------
# Materials (piecewise constant per cell)
# ----------------------------

@dataclass(frozen=True)
class CellMaterial:
    Sigma_t: float
    Sigma_s: float

    @property
    def p_scatter(self) -> float:
        if self.Sigma_t <= 0:
            raise ValueError("Sigma_t must be positive.")
        if not (0.0 <= self.Sigma_s <= self.Sigma_t):
            raise ValueError("Require 0 <= Sigma_s <= Sigma_t.")
        return self.Sigma_s / self.Sigma_t

@dataclass
class MaterialField:
    mesh: Mesh
    mats: List[CellMaterial]  # length nx*ny*nz

    def mat_at(self, i: int, j: int, k: int) -> CellMaterial:
        return self.mats[self.mesh.flat(i, j, k)]

# ----------------------------
# Encode/decode position, direction, RNG inside bit-vector state
# ----------------------------

def decode_rng(bits: List[int], layout: StateLayout) -> int:
    (*_a, sr) = layout.slices()
    return bits_to_uint(bits[sr[0]:sr[1]])

def encode_rng(bits: List[int], layout: StateLayout, s: int) -> None:
    (*_a, sr) = layout.slices()
    bits[sr[0]:sr[1]] = uint_to_bits(s & 0xFFFFFFFFFFFFFFFF, 64)

def next_u01(bits: List[int], layout: StateLayout) -> float:
    s = decode_rng(bits, layout)
    s = xorshift64star(s)
    encode_rng(bits, layout, s)
    return u01_from_u64(s)

def decode_position(bits: List[int], layout: StateLayout, dom: BoxDomain) -> Tuple[float, float, float]:
    (sx, sy, sz, *_rest) = layout.slices()
    ix = bits_to_uint(bits[sx[0]:sx[1]])
    iy = bits_to_uint(bits[sy[0]:sy[1]])
    iz = bits_to_uint(bits[sz[0]:sz[1]])
    maxu = (1 << layout.pos_bits) - 1
    x = dom.xmin + (dom.xmax - dom.xmin) * (ix / maxu)
    y = dom.ymin + (dom.ymax - dom.ymin) * (iy / maxu)
    z = dom.zmin + (dom.zmax - dom.zmin) * (iz / maxu)
    return x, y, z

def encode_position(bits: List[int], layout: StateLayout, dom: BoxDomain, r: Tuple[float, float, float]) -> None:
    (sx, sy, sz, *_rest) = layout.slices()
    x, y, z = r
    maxu = (1 << layout.pos_bits) - 1

    def clamp(v, lo, hi): return max(lo, min(hi, v))

    tx = (x - dom.xmin) / (dom.xmax - dom.xmin)
    ty = (y - dom.ymin) / (dom.ymax - dom.ymin)
    tz = (z - dom.zmin) / (dom.zmax - dom.zmin)
    tx = clamp(tx, 0.0, 1.0); ty = clamp(ty, 0.0, 1.0); tz = clamp(tz, 0.0, 1.0)

    ix = int(round(tx * maxu))
    iy = int(round(ty * maxu))
    iz = int(round(tz * maxu))

    bits[sx[0]:sx[1]] = uint_to_bits(ix, layout.pos_bits)
    bits[sy[0]:sy[1]] = uint_to_bits(iy, layout.pos_bits)
    bits[sz[0]:sz[1]] = uint_to_bits(iz, layout.pos_bits)

def decode_direction(bits: List[int], layout: StateLayout) -> Tuple[float, float, float]:
    (_sx, _sy, _sz, sux, suy, suz, _sr) = layout.slices()
    ix = bits_to_sint(bits[sux[0]:sux[1]])
    iy = bits_to_sint(bits[suy[0]:suy[1]])
    iz = bits_to_sint(bits[suz[0]:suz[1]])
    scale = (1 << (layout.dir_bits - 1)) - 1
    ux, uy, uz = ix/scale, iy/scale, iz/scale
    n = math.sqrt(ux*ux + uy*uy + uz*uz)
    if n == 0.0:
        return 1.0, 0.0, 0.0
    return ux/n, uy/n, uz/n

def encode_direction(bits: List[int], layout: StateLayout, u: Tuple[float, float, float]) -> None:
    (_sx, _sy, _sz, sux, suy, suz, _sr) = layout.slices()
    ux, uy, uz = u
    n = math.sqrt(ux*ux + uy*uy + uz*uz)
    if n == 0.0:
        ux, uy, uz = 1.0, 0.0, 0.0
        n = 1.0
    ux, uy, uz = ux/n, uy/n, uz/n
    scale = (1 << (layout.dir_bits - 1)) - 1
    ix = int(round(ux * scale))
    iy = int(round(uy * scale))
    iz = int(round(uz * scale))
    bits[sux[0]:sux[1]] = sint_to_bits(ix, layout.dir_bits)
    bits[suy[0]:suy[1]] = sint_to_bits(iy, layout.dir_bits)
    bits[suz[0]:suz[1]] = sint_to_bits(iz, layout.dir_bits)

# ----------------------------
# Sampling
# ----------------------------

def sample_free_path(bits: List[int], layout: StateLayout, Sigma_t: float) -> float:
    u = next_u01(bits, layout)
    return -math.log(u) / Sigma_t

def sample_isotropic_direction(bits: List[int], layout: StateLayout) -> Tuple[float, float, float]:
    u1 = next_u01(bits, layout)
    u2 = next_u01(bits, layout)
    mu = 2.0*u1 - 1.0
    phi = 2.0*math.pi*u2
    st = math.sqrt(max(0.0, 1.0 - mu*mu))
    return st*math.cos(phi), st*math.sin(phi), mu

# ----------------------------
# Boundary distance within current cell
# ----------------------------

def distance_to_cell_face(mesh: Mesh, r: Tuple[float,float,float], u: Tuple[float,float,float],
                          ijk: Tuple[int,int,int]) -> float:
    """
    Distance along direction u from point r to the next cell face (x/y/z plane) within the current cell.
    Returns +inf if direction component is 0 for that axis.
    """
    i, j, k = ijk
    x, y, z = r
    ux, uy, uz = u
    x0, x1, y0, y1, z0, z1 = mesh.cell_bounds(i, j, k)

    eps = 1e-15

    def t_to_plane(pos, vel, lo, hi):
        if abs(vel) < eps:
            return math.inf
        if vel > 0:
            return (hi - pos) / vel
        else:
            return (lo - pos) / vel

    tx = t_to_plane(x, ux, x0, x1)
    ty = t_to_plane(y, uy, y0, y1)
    tz = t_to_plane(z, uz, z0, z1)
    tmin = min(tx, ty, tz)

    # Numerical guard: if we're exactly on a face, push forward slightly to avoid zero steps
    if tmin < 0:
        tmin = 0.0
    return tmin

# ----------------------------
# Tallies and driver
# ----------------------------

@dataclass
class Tallies:
    absorbed: int = 0
    leaked: int = 0
    scattered: int = 0
    collisions: int = 0

def make_initial_state(layout: StateLayout, dom: BoxDomain,
                       position: Tuple[float,float,float],
                       direction: Tuple[float,float,float],
                       seed64: int) -> List[int]:
    bits = [0]*layout.total_bits
    encode_position(bits, layout, dom, position)
    encode_direction(bits, layout, direction)
    if seed64 == 0:
        seed64 = 0x9E3779B97F4A7C15
    encode_rng(bits, layout, seed64)
    return bits

def simulate_fixed_source(
    n_particles: int,
    layout: StateLayout,
    mesh: Mesh,
    matfield: MaterialField,
    source_pos: Tuple[float,float,float],
    source_dir: Tuple[float,float,float],
    max_steps: int = 100000
) -> Tuple[Tallies, List[float]]:
    """
    Simulate n_particles from a mono-directional point source (source_pos, source_dir).
    Returns (event tallies, flux track-length tally per cell).
    """
    flux = [0.0]*(mesh.nx*mesh.ny*mesh.nz)
    events = Tallies()

    dom = mesh.dom
    tiny = 1e-12

    for p in range(n_particles):
        seed = 0xD1B54A32D192ED03 ^ (p * 0x9E3779B97F4A7C15)
        state = make_initial_state(layout, dom, source_pos, source_dir, seed)

        r = decode_position(state, layout, dom)
        u = decode_direction(state, layout)

        for _ in range(max_steps):
            # Leakage check
            if not dom.contains(r):
                events.leaked += 1
                break

            i, j, k = mesh.cell_index(r)
            mat = matfield.mat_at(i, j, k)

            # Sample collision distance in this cell's material
            s_col = sample_free_path(state, layout, mat.Sigma_t)

            # Distance to next cell face along u
            s_face = distance_to_cell_face(mesh, r, u, (i, j, k))

            # Determine event
            s = min(s_col, s_face)

            # Track-length tally in current cell
            flux[mesh.flat(i, j, k)] += s

            # Advance
            r = (r[0] + s*u[0], r[1] + s*u[1], r[2] + s*u[2])
            encode_position(state, layout, dom, r)  # keep state in sync

            # If boundary crossing happens first, continue into next cell
            if s_face <= s_col:
                # Nudge to avoid sticking on boundary due to floating error
                r = (r[0] + tiny*u[0], r[1] + tiny*u[1], r[2] + tiny*u[2])
                encode_position(state, layout, dom, r)
                continue

            # Otherwise collision
            events.collisions += 1
            xi = next_u01(state, layout)
            if xi < mat.p_scatter:
                events.scattered += 1
                u = sample_isotropic_direction(state, layout)
                encode_direction(state, layout, u)
                continue
            else:
                events.absorbed += 1
                break
        else:
            # If max_steps exceeded, treat as leaked (or absorb) depending on preference.
            events.leaked += 1

    return events, flux

def main():
    # Problem setup
    dom = BoxDomain(0.0, 10.0, 0.0, 10.0, 0.0, 10.0)
    mesh = Mesh(nx=10, ny=10, nz=10, dom=dom)
    layout = StateLayout(pos_bits=20, dir_bits=18)

    # Material field: homogeneous, but structured for future heterogeneity
    Sigma_t = 0.5
    Sigma_s = 0.45
    mats = [CellMaterial(Sigma_t=Sigma_t, Sigma_s=Sigma_s) for _ in range(mesh.nx*mesh.ny*mesh.nz)]
    matfield = MaterialField(mesh=mesh, mats=mats)

    # Fixed source: near x-min face, pointing +x
    source_pos = (0.1, 5.0, 5.0)
    source_dir = (1.0, 0.0, 0.0)

    # Run
    n_particles = 5000
    events, flux = simulate_fixed_source(
        n_particles=n_particles,
        layout=layout,
        mesh=mesh,
        matfield=matfield,
        source_pos=source_pos,
        source_dir=source_dir,
        max_steps=200000
    )

    # Report
    print("=== Fixed-source MC results ===")
    print(f"Particles:      {n_particles}")
    print(f"Collisions:     {events.collisions}")
    print(f"Scatters:       {events.scattered}")
    print(f"Absorbed:       {events.absorbed}")
    print(f"Leaked:         {events.leaked}")
    print(f"Leak fraction:  {events.leaked / n_particles:.6f}")
    print(f"Abs fraction:   {events.absorbed / n_particles:.6f}")

    # Basic flux normalization suggestion:
    # Track-length estimator for scalar flux: phi_i ≈ (1 / (V_i * N)) * sum track_length_in_i
    cell_vol = mesh.dx * mesh.dy * mesh.dz
    phi = [f / (cell_vol * n_particles) for f in flux]

    # Print a simple 1D cut: average flux over y,z for each x-cell index
    x_profile = []
    for i in range(mesh.nx):
        acc = 0.0
        for j in range(mesh.ny):
            for k in range(mesh.nz):
                acc += phi[mesh.flat(i, j, k)]
        x_profile.append(acc / (mesh.ny*mesh.nz))

    print("\n=== Mean flux vs x-cell (averaged over y,z) ===")
    for i, val in enumerate(x_profile):
        x_center = dom.xmin + (i + 0.5)*mesh.dx
        print(f"i={i:2d}  x~{x_center:6.3f}  phi~{val:.6e}")

if __name__ == "__main__":
    main()