import numpy as np
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.linalg import cg
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.collections import LineCollection
import seaborn as sns
import pandas as pd

# -------- geometry helpers --------
def tri_area(p, q, r):
    """Signed area of triangle (p,q,r) in 2D; return positive area."""
    return 0.5 * abs((q[0]-p[0])*(r[1]-p[1]) - (q[1]-p[1])*(r[0]-p[0]))

def cot_at_vertex(p, q, r):
    """
    cot(angle at 'p') in triangle (p,q,r).
    Using dot/cross to avoid trig: cot(theta) = ( (q-p)·(r-p) ) / |(q-p)x(r-p)|.
    """
    u = q - p
    v = r - p
    cross = abs(u[0]*v[1] - u[1]*v[0])
    dot   = u.dot(v)
    return dot / cross

# -------- main assembly --------
def assemble_fvem(
    nodes,             # array (N,2)
    tris,              # array (T,3) with vertex indices (int)
    kappa_elem,        # array (T,) scalar kappa per triangle, or callable K(x)
    f_elem=None,       # array (T,) value of f per triangle, or callable f(x)
    neumann_edges=None,# list of (i,j,gN) or (i,j) with callable gN(xmid,n)
    dirichlet=None     # dict {node_id: value}
):
    """
    Vertex-centered FVEM on triangles for -div(kappa grad u) = f.
    Unknowns at vertices. Dirichlet enforced by row/col modification.

    neumann_edges:
      - If provided as (i,j,gN_value), uses that constant value on edge.
      - If provided as (i,j) and you passed a callable gN(xmid, n), give gN as neumann_edges_gN.
    """
    N = len(nodes)
    T = len(tris)
    nodes = np.asarray(nodes, dtype=float)
    tris  = np.asarray(tris, dtype=int)

    # Normalize inputs
    if callable(kappa_elem):
        # evaluate at barycenter of each triangle
        kappa = np.empty(T)
        for t, (i,j,k) in enumerate(tris):
            xc = (nodes[i] + nodes[j] + nodes[k]) / 3.0
            kappa[t] = float(kappa_elem(xc))
    else:
        kappa = np.asarray(kappa_elem, dtype=float)
        assert len(kappa) == T

    if f_elem is None:
        f_values = np.zeros(T)
    elif callable(f_elem):
        f_values = np.empty(T)
        for t, (i,j,k) in enumerate(tris):
            xc = (nodes[i] + nodes[j] + nodes[k]) / 3.0
            f_values[t] = float(f_elem(xc))
    else:
        f_values = np.asarray(f_elem, dtype=float)
        assert len(f_values) == T

    # Triplet lists
    I = []
    J = []
    V = []
    b = np.zeros(N, dtype=float)

    # --- element (cotangent) contributions ---
    for t, (i, j, k) in enumerate(tris):
        xi, xj, xk = nodes[i], nodes[j], nodes[k]
        area = tri_area(xi, xj, xk)
        if area <= 0.0:
            raise ValueError("Degenerate or inverted triangle at index {}".format(t))

        # cot at vertex opposite each edge:
        # edge (i,j) opposite k -> use cot(angle at k)
        cot_k = cot_at_vertex(xk, xi, xj)
        cot_i = cot_at_vertex(xi, xj, xk)
        cot_j = cot_at_vertex(xj, xk, xi)

        w_ij = 0.5 * kappa[t] * cot_k
        w_jk = 0.5 * kappa[t] * cot_i
        w_ki = 0.5 * kappa[t] * cot_j

        # Assemble graph-Laplacian pattern (symmetric)
        # (i,j)
        I += [i, j, i, j]
        J += [i, j, j, i]
        V += [ w_ij, w_ij, -w_ij, -w_ij]
        # (j,k)
        I += [j, k, j, k]
        J += [j, k, k, j]
        V += [ w_jk, w_jk, -w_jk, -w_jk]
        # (k,i)
        I += [k, i, k, i]
        J += [k, i, i, k]
        V += [ w_ki, w_ki, -w_ki, -w_ki]

        # RHS: median-dual volume = area/3 per vertex
        fK = f_values[t]
        share = (area / 3.0) * fK
        b[i] += share
        b[j] += share
        b[k] += share

    # Build sparse matrix
    A = coo_matrix((V, (I, J)), shape=(N, N)).tocsr()
    # Ensure strict symmetry (numeric): A = (A + A.T)/2
    A = (A + A.T) * 0.5

    # --- Neumann edges (optional) ---
    if neumann_edges is not None:
        for item in neumann_edges:
            if len(item) == 3:   # (i,j,gN_const)
                i, j, gN = item
                xmid = 0.5 * (nodes[i] + nodes[j])
                nij = nodes[j] - nodes[i]
                L = np.linalg.norm(nij)
                # midpoint rule, split equally
                b[i] += 0.5 * L * gN
                b[j] += 0.5 * L * gN
            elif len(item) == 2: # (i,j) with callable gN provided as third arg in a tuple?
                raise ValueError("For callable Neumann data, pass triplets (i, j, gN_value). "
                                 "If you need (i,j,gN(xmid,n)), precompute gN on edges.")
            else:
                raise ValueError("Neumann edge entries must be (i,j,gN_const).")

    # --- Dirichlet constraints (row/col modification) ---
    if dirichlet:
        dir_nodes = sorted(dirichlet.keys())
        fixed = np.array(dir_nodes, dtype=int)
        uD    = np.array([dirichlet[i] for i in fixed], dtype=float)

        # Modify b := b - A[:,fixed] * uD, then zero rows/cols and set diag=1, b=uD
        # Take the contribution of known values to RHS:
        b -= A[:, fixed] @ uD

        # Zero rows and cols of fixed nodes
        for idx, val in zip(fixed, uD):
            # zero row idx
            A.data[A.indptr[idx]:A.indptr[idx+1]] = 0.0
        # zero columns by operating on A.T rows
        AT = A.transpose().tocsr()
        for idx in fixed:
            AT.data[AT.indptr[idx]:AT.indptr[idx+1]] = 0.0
        A = AT.transpose().tocsr()

        # Set diagonal to 1 and RHS to prescribed value
        A[fixed, fixed] = 1.0
        b[fixed] = uD

    return A.tocsr(), b

def plot_fvem_solution(
    nodes, tris, u,
    title="FVEM solution u",
    show_tri_edges=True,
    show_dual_edges=True,
    tri_edge_lw=0.8,
    dual_edge_lw=0.8
):
    """
    nodes: (N,2) array of vertex coords
    tris:  (T,3) int array of triangle vertex indices
    u:     (N,) array of nodal values

    Options:
      show_tri_edges  – draw the primal mesh (triangle) edges
      show_dual_edges – draw median-dual control-volume edges (FVEM “box scheme”)
      tri_edge_lw     – linewidth for triangle edges
      dual_edge_lw    – linewidth for dual edges
    """
    nodes = np.asarray(nodes, float)
    tris  = np.asarray(tris,  int)
    u     = np.asarray(u,     float)

    tri = mtri.Triangulation(nodes[:,0], nodes[:,1], triangles=tris)

    fig, ax = plt.subplots(figsize=(6,5))
    # Smooth color field; we’ll draw edges separately for clarity
    tpc = ax.tripcolor(tri, u, shading="flat")
    cbar = fig.colorbar(tpc, ax=ax)
    cbar.set_label("u")

    # --- Primal (triangle) edges ---
    if show_tri_edges:
        # Build unique undirected edges from triangles
        edges = set()
        for (i,j,k) in tris:
            for a,b in ((i,j),(j,k),(k,i)):
                if a > b: a,b = b,a
                edges.add((a,b))
        segs = [np.vstack((nodes[a], nodes[b])) for (a,b) in edges]
        lc_tri = LineCollection(segs, linewidths=tri_edge_lw)
        ax.add_collection(lc_tri)

    # --- Median-dual edges (control-volume boundaries) ---
    # We draw the “barycentric dual” edges:
    #   * For each interior primal edge (i,j) shared by triangles K1, K2:
    #       connect barycenter(K1) to barycenter(K2).
    #   * For each boundary edge (i,j) with single adjacent K:
    #       connect barycenter(K) to the midpoint of edge (i,j).
    if show_dual_edges:
        # Triangle barycenters
        bcs = (nodes[tris[:,0]] + nodes[tris[:,1]] + nodes[tris[:,2]]) / 3.0

        # Build edge -> adjacent triangle ids
        edge2tris = {}
        for t, (i,j,k) in enumerate(tris):
            for a,b in ((i,j),(j,k),(k,i)):
                e = (a,b) if a < b else (b,a)
                edge2tris.setdefault(e, []).append(t)

        segs_dual = []
        for (a,b), adj in edge2tris.items():
            mid = 0.5 * (nodes[a] + nodes[b])
            if len(adj) == 2:
                # interior edge: connect the two barycenters
                t1, t2 = adj
                segs_dual.append(np.vstack((bcs[t1], bcs[t2])))
            elif len(adj) == 1:
                # boundary edge: connect triangle barycenter to edge midpoint
                t1 = adj[0]
                segs_dual.append(np.vstack((bcs[t1], mid)))
            # (If mesh is consistent, len(adj) is 1 or 2.)

        lc_dual = LineCollection(segs_dual, linewidths=dual_edge_lw)
        ax.add_collection(lc_dual)

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(title)
    plt.tight_layout()
    plt.show()

def plot_fvem_solution_nodes(
    xs, ys, u,
    title="FVEM solution u",
    show_tri_edges=True,
    show_dual_edges=True,
    tri_edge_lw=0.8,
    dual_edge_lw=0.8
):
    """
    nodes: (N,2) array of vertex coords
    tris:  (T,3) int array of triangle vertex indices
    u:     (N,) array of nodal values

    Options:
      show_tri_edges  – draw the primal mesh (triangle) edges
      show_dual_edges – draw median-dual control-volume edges (FVEM “box scheme”)
      tri_edge_lw     – linewidth for triangle edges
      dual_edge_lw    – linewidth for dual edges
    """
    xs = np.asarray(xs, float)
    ys = np.asarray(ys, float)
    u     = np.asarray(u,     float)

    fig, ax = plt.subplots(figsize=(6,5))
    scat = ax.scatter(xs, ys, c=u)
    cbar = fig.colorbar(scat, ax=ax)
    plt.show()
    plt.show()

# --------- small usage example ----------
if __name__ == "__main__":
    # Unit square [0,1]^2, coarse 2x2 grid split into triangles
    #  (0,0)-(1,0)-(1,1)-(0,1) with one diagonal per cell
    nx = 4
    ny = 4
    xs = x = np.linspace(0, 1, nx+1)
    ys = x = np.linspace(0, 1, ny+1)
    X, Y = np.meshgrid(xs, ys, indexing="xy")
    nodes = np.c_[X.ravel(), Y.ravel()]  # 9 nodes

    # Helper to map (ix,iy) to node id
    def nid(ix, iy): return iy*(nx+1) + ix

    # 8 triangles (two per square)
    tris = []
    for iy in range(nx):
        for ix in range(ny):
            a = nid(ix,   iy)
            b = nid(ix+1, iy)
            c = nid(ix+1, iy+1)
            d = nid(ix,   iy+1)
            # split square along (a,c)
            tris += [(a,b,c), (a,c,d)]
    tris = np.array(tris, dtype=int)

    # Piecewise-constant kappa = 1
    kappa_elem = np.ones(len(tris))

    # Source f(x,y) = 1
    def f_callable(x): return 1.0

    # Dirichlet: u=0 on boundary nodes
    boundary_nodes = {i: 0.0 for i,(x,y) in enumerate(nodes)
                      if (abs(x-0.0)<1e-12 or abs(x-1.0)<1e-12 or
                          abs(y-0.0)<1e-12 or abs(y-1.0)<1e-12)}

    # Assemble
    A, b = assemble_fvem(nodes, tris, kappa_elem, f_elem=f_callable, neumann_edges=None,
                         dirichlet=boundary_nodes)

    # Solve (SPD): CG
    u, info = cg(A, b, tol=1e-10, maxiter=200)
    print("CG info =", info)
    print("u =", u.reshape(-1))

    # After you compute u = CG solution:
    plot_fvem_solution_nodes(
    X.ravel(), Y.ravel(), u)

    #plot_fvem_solution(
    #nodes, tris, u)