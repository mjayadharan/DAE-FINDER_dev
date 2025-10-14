from graphviz import Digraph
from IPython.display import display, SVG, Image
from shutil import which
import re
from typing import Optional, Sequence
import numpy as np

def _slug(x: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]+", "_", str(x))

# --- tiny color helpers ---
def _hex_to_rgb(h): h=h.lstrip("#"); return tuple(int(h[i:i+2],16) for i in (0,2,4))
def _rgb_to_hex(rgb): return "#{:02x}{:02x}{:02x}".format(*rgb)
def _tint(hex_color, f=0.22):
    r,g,b=_hex_to_rgb(hex_color); R=int(r+(255-r)*f); G=int(g+(255-g)*f); B=int(b+(255-b)*f)
    return _rgb_to_hex((R,G,B))
def _shade(hex_color, f=0.35):
    r,g,b=_hex_to_rgb(hex_color); R=int(r*(1-f)); G=int(g*(1-f)); B=int(b*(1-f))
    return _rgb_to_hex((R,G,B))
def _palette(n):
    base=["#4C78A8","#F58518","#54A24B","#E45756","#72B7B2","#EECA3B",
          "#B279A2","#FF9DA6","#9D755D","#8E6C8A","#59A14F","#BAB0AC"]
    if n<=len(base): return base[:n]
    import colorsys
    extra=[]
    for i in range(n-len(base)):
        h=(i/max(1,n-len(base)))%1.0; r,g,b=colorsys.hsv_to_rgb(h,0.55,0.92)
        extra.append("#{:02x}{:02x}{:02x}".format(int(r*255),int(g*255),int(b*255)))
    return base+extra

def draw_lr_mains_with_vertical_children(
    main_list,
    children_map,
    *,
    main_values: Optional[Sequence] = None,
    value_sep: str = " ",
    highlight_nodes: Optional[Sequence[str]] = None,  # list of names to highlight anywhere
    left_label="Groups",
    render_to_file=False, filename="graph", file_format="png", notebook_format="svg",
    fontname="Helvetica", node_shape_main="box", node_shape_child="ellipse"
):
    if which("dot") is None:
        raise RuntimeError("Graphviz 'dot' not found on PATH.")
    if main_values is not None and len(main_values) != len(main_list):
        raise ValueError("main_values must be the same length as main_list.")

    # yellow border highlight (fill stays original!)
    hl = set(map(str, (highlight_nodes or [])))
    HL_BORDER = "#C90000"  # warm/yellowish
    HL_PENW_M = "4.5"      # border width for mains
    HL_PENW_C = "4.0"      # border width for children

    colors = _palette(len(main_list))
    g = Digraph("G")
    g.attr(rankdir="TB", nodesep="0.1", ranksep="0.70", splines="ortho", fontname=fontname, fontsize="12")
    g.attr("node", style="rounded,filled", fontname=fontname, fontsize="11", margin="0.06,0.04")
    g.attr("edge", arrowsize="0.85", penwidth="1.1", color="#7d7d7d")

    g.node("_groups", left_label, shape="box", fillcolor="#f1f3f8", color="#b8bfd0", penwidth="1.4")

    max_kids = max((len(children_map.get(m, [])) for m in main_list), default=0)

    def spacer(i, r):
        sid = f"_sp_{i}_{r}"
        g.node(sid, label="", shape="point", width="0.01", height="0.01",
               color="#ffffff", style="invis", group=f"col{i}")
        return sid

    # ------- ROW 0: mains + "+" -------
    row0 = Digraph()
    row0.attr(rank="same")
    row0.node("_groups_proxy", label="", shape="point", width="0.01", height="0.01", color="#ffffff", style="invis")
    row0.edge("_groups", "_groups_proxy", style="invis", weight="100")

    main_row_nodes = []
    col_nodes = [[] for _ in main_list]
    row0_ids = []

    for i, m in enumerate(main_list):
        base = colors[i]
        label = f"{main_values[i]}{value_sep}{m}" if main_values is not None else str(m)

        # keep original fill; only border changes if highlighted
        is_hl = str(m) in hl
        fill   = _tint(base, 0.12)                 # original fill
        border = HL_BORDER if is_hl else _shade(base, 0.35)
        penw   = HL_PENW_M if is_hl else "1.6"

        row0.node(
            str(m),
            label=label,
            shape=node_shape_main,
            fillcolor=fill,
            color=border,
            penwidth=penw,
            peripheries="1",
            group=f"col{i}"
        )
        row0_ids.append(str(m))
        main_row_nodes.append(str(m))
        col_nodes[i].append(str(m))

        if i < len(main_list)-1:
            pid = f"_plus_{i}"
            row0.node(pid, label="<<B>+</B>>", shape="plaintext",
                      width="0.35", height="0.35",
                      fixedsize="true",
                      fontsize="16",
                      fontname="Helvetica-Bold",
                      group=f"col{i}")
            row0_ids.append(pid)
            main_row_nodes.append(pid)

    for a, b in zip(row0_ids, row0_ids[1:]):
        row0.edge(a, b, style="invis", weight="300")
    g.subgraph(row0)

    if main_list:
        g.edge("_groups", str(main_list[0]), arrowhead="none", color="#b2b2b2", penwidth="1.1")

    # ------- CHILD ROWS -------
    col_child_ids = [[] for _ in main_list]
    for r in range(max_kids):
        row = Digraph()
        row.attr(rank="same")
        row_ids = []
        for i, m in enumerate(main_list):
            base = colors[i]
            kids = children_map.get(m, [])
            if r < len(kids):
                child_name = str(kids[r])
                kid_id = f"_kid_{i}_{r}_{_slug(child_name)}"

                # keep original fill; only border changes if highlighted
                is_hl = child_name in hl
                c_fill   = _tint(base, 0.26)                 # original child fill
                c_border = HL_BORDER if is_hl else _shade(base, 0.25)
                c_penw   = HL_PENW_C if is_hl else "1.25"

                g.node(
                    kid_id,
                    label=child_name,
                    shape=node_shape_child,
                    fillcolor=c_fill,
                    color=c_border,
                    penwidth=c_penw,
                    peripheries="1",
                    group=f"col{i}"
                )
                row_ids.append(kid_id)
                col_nodes[i].append(kid_id)
                col_child_ids[i].append(kid_id)

                if r == 0:
                    g.edge(str(m), kid_id, color=_shade(base,0.20), penwidth="1.15",
                           arrowhead="none", weight="3", constraint="true")
                else:
                    prev_id = col_child_ids[i][r-1]
                    g.edge(prev_id, kid_id, color=_shade(base,0.20), penwidth="1.0",
                           arrowhead="none", weight="2", constraint="true")
            else:
                row_ids.append(spacer(i, r))
        for a, b in zip(row_ids, row_ids[1:]):
            row.edge(a, b, style="invis", weight="200")
        g.subgraph(row)

    # ------- CLUSTERS (unchanged) -------
    if main_row_nodes:
        cl_main = Digraph(name="cluster_main_row")
        cl_main.attr(style="dashed,rounded", color="#9aa3b2", penwidth="1.2", margin="0", label="")
        for nid in main_row_nodes:
            cl_main.node(nid)
        g.subgraph(cl_main)

    for i, nodes in enumerate(col_nodes):
        if not nodes: continue
        base = colors[i]
        cl = Digraph(name=f"cluster_col_{i}")
        cl.attr(style="dashed,rounded", color=_shade(base, 0.35), penwidth="1.2", label="")
        for nid in nodes:
            cl.node(nid)
        g.subgraph(cl)

    if render_to_file:
        return g.render(filename=filename, format=file_format, cleanup=True)
    out = g.pipe(format=notebook_format.lower())
    display(SVG(out) if notebook_format.lower()=="svg" else Image(data=out))


# # ---------- Example ----------
# if __name__ == "__main__":
#     mains = ["A","B","C","D"]
#     values = [10, 20, 30, 40]
#     children = {
#         "A": ["A1","A2"],
#         "B": ["A1", "B1"],
#         "C": ["C1","C2","C3"],
#         "D": ["D1","D2"]
#     }
#     draw_lr_mains_with_vertical_children(
#         mains, children,
#         main_values=values,
#         highlight_nodes=["A", "B", "C2", "D1"],  # yellow *border* only
#         notebook_format="svg"
#     )




def QR_with_threshold(A, rank_threshold=1e-3, stable_right=True, verbose=False):
    A = np.asarray(A)
    N, J = A.shape
    A_work = A.copy()
    P = np.arange(J)
    Q_cols = []
    L = 0
    R_full = np.zeros((J, J), dtype=A.dtype)

    k, r = 0, J - 1
    while k <= r:
        a = A_work[:, k].copy()
        for i in range(L):
            rij = np.dot(Q_cols[i], a)
            R_full[i, k] = rij
            a -= rij * Q_cols[i]
        rkk = np.linalg.norm(a)

        if rkk <= rank_threshold:
            if verbose:
                print(f"pos {k} (orig {P[k]}) deemed dependent (rkk={rkk:.3e} <= {rank_threshold:.1e}); pivot right.")
            if k != r:
                A_work[:, [k, r]] = A_work[:, [r, k]]
                if L > 0:
                    R_full[:L, [k, r]] = R_full[:L, [r, k]]
                P[k], P[r] = P[r], P[k]
            r -= 1
            continue
        else:
            qk = a / (rkk if rkk != 0 else 1.0)
            Q_cols.append(qk)
            R_full[L, k] = rkk
            L += 1
            k += 1

    Q = np.column_stack(Q_cols) if L > 0 else np.zeros((N, 0), dtype=A.dtype)
    R = R_full[:L, :]

    # Accuracy pass + optional stable ordering for the right block
    if L > 0 and L < J:
        if stable_right:
            order = np.argsort(P[L:])          # sort by original indices
            P[L:] = P[L:][order]
            R[:, L:] = R[:, L:][:, order]
        # Recompute right-block with the final permutation for accuracy
        R[:, L:] = Q.T @ A[:, P[L:]]

    # Clean up strictly upper in left block
    for i in range(L):
        R[i+1:, i] = 0.0

    return Q, R, P, L

def draw_relation_map(model_df, redundancy_df, column_name = "Coefficient", eq_lhs_label='Y', true_terms=[], model_threshold=1, redundancy_threshold=0.0001):
    assert column_name in model_df.columns, "Column name \"{}\" not in model_df".format(column_name)
    mains = model_df.index[model_df[column_name].abs() > model_threshold]
    mains = mains[np.argsort(model_df.loc[mains, column_name].abs())].tolist()[::-1]
    main_values = [f"{v:.2f}" for v in model_df.loc[mains, column_name]]

    children = {col: redundancy_df.index[redundancy_df[col] > redundancy_threshold].tolist() for col in redundancy_df.columns}
   
    return draw_lr_mains_with_vertical_children(mains, children, main_values=main_values, highlight_nodes=true_terms, left_label=eq_lhs_label, notebook_format="svg")

# # --------------------------- Test / demo for QR_with_threshold---------------------------
# if __name__ == "__main__":
#     rng = np.random.default_rng(0)

#     # Construct a test matrix with clear dependencies.
#     N, J = 8, 6
#     # Start with 3 independent columns
#     B = rng.normal(size=(N, 3))
#     # Create 3 more columns as linear combos (+ small noise)
#     C1 = 1.5 * B[:, 0] - 0.2 * B[:, 1]
#     C2 = -0.7 * B[:, 1] + 0.3 * B[:, 2]
#     C3 = 0.9 * B[:, 0] + 0.9 * B[:, 1] - 0.2 * B[:, 2]
#     # Add small noise to make them "nearly" dependent (tune below threshold)
#     noise_level = 1e-4
#     C1 += noise_level * rng.normal(size=N)
#     C2 += noise_level * rng.normal(size=N)
#     C3 += noise_level * rng.normal(size=N)

#     A = np.column_stack([ C1,B, C2, C3])
#     A = np.column_stack([B, C1, C2, C3])
    
#     # Run our CPQR with threshold
#     Q, R, P, L = QR_with_threshold(A, rank_threshold=1e-3)

#     # Reconstruct and measure error
#     A_pivoted = A[:, P]                   # permuted columns
#     A_recon = Q @ R                       # reconstruction
#     err_fro = np.linalg.norm(A_pivoted - A_recon, ord='fro')
#     rel_err = err_fro / (np.linalg.norm(A_pivoted, ord='fro') + 1e-16)

#     # Report
#     print("Permutation P:", P)
#     print("Detected independent columns (L):", L)
#     print("Q shape:", Q.shape, "R shape:", R.shape)
#     print("Frobenius reconstruction error:", err_fro)
#     print("Relative Frobenius error:", rel_err)

#     # Sanity checks
#     # 1) Q has orthonormal columns
#     if Q.shape[1] > 0:
#         orth_err = np.linalg.norm(Q.T @ Q - np.eye(Q.shape[1]))
#         print("Orthonormality error ||Q^T Q - I||_F:", orth_err)
#     # 2) The right block columns are small after projecting out Q
#     if Q.shape[1] < J:
#         residual_right = np.linalg.norm(
#             A_pivoted[:, Q.shape[1]:] - Q @ (Q.T @ A_pivoted[:, Q.shape[1]:]),
#             ord='fro'
#         )
#         print("Residual norm of right (dependent) block:", residual_right)

#         print(A.shape)