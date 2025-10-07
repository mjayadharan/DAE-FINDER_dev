from graphviz import Digraph
from IPython.display import display, SVG, Image
from shutil import which

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
    main_list, children_map, *,
    left_label="Groups",
    render_to_file=False, filename="graph", file_format="png", notebook_format="svg",
    fontname="Helvetica", node_shape_main="box", node_shape_child="ellipse"
):
    """Main nodes appear left->right in the top row; each parent's children are stacked directly below it.
       Adds: dotted main-row box (includes '+' nodes) and per-column dotted boxes (parent+children)."""
    if which("dot") is None:
        raise RuntimeError("Graphviz 'dot' not found on PATH.")

    colors = _palette(len(main_list))
    g = Digraph("G")
    # g.attr(rankdir="TB", nodesep="0.10", ranksep="0.70", splines="ortho")
    g.attr(rankdir="TB", nodesep="0.1", ranksep="0.70", splines="ortho", fontname=fontname, fontsize="12")
    g.attr("node", style="rounded,filled", fontname=fontname, fontsize="11", margin="0.06,0.04")
    g.attr("edge", arrowsize="0.85", penwidth="1.1", color="#7d7d7d")

    # --- Groups node (far left in the top row)
    g.node("_groups", left_label, shape="box", fillcolor="#f1f3f8", color="#b8bfd0", penwidth="1.4")

    # Determine grid height (max #children across parents)
    max_kids = 0
    for m in main_list:
        max_kids = max(max_kids, len(children_map.get(m, [])))

    # Helper to create invisible spacer (keeps the grid column intact)
    def spacer(i, r):
        sid = f"_sp_{i}_{r}"
        g.node(sid, label="", shape="point", width="0.01", height="0.01",
               color="#ffffff", style="invis", group=f"col{i}")
        return sid

    # ------- ROW 0: mains + "+" between them (rank = same) -------
    row0 = Digraph()
    row0.attr(rank="same")
    # put Groups in the same rank so it sits to the left
    row0.node("_groups_proxy", label="", shape="point", width="0.01", height="0.01", color="#ffffff", style="invis")
    row0.edge("_groups", "_groups_proxy", style="invis", weight="100")

    # Track nodes for: main-row cluster and per-column clusters
    main_row_nodes = []
    col_nodes = [[] for _ in main_list]  # will hold parent + children per column

    row0_ids = []
    for i, m in enumerate(main_list):
        base = colors[i]
        # Parent node
        row0.node(str(m), shape=node_shape_main, fillcolor=_tint(base,0.12),
                  color=_shade(base,0.35), penwidth="1.6", group=f"col{i}")
        row0_ids.append(str(m))
        main_row_nodes.append(str(m))
        col_nodes[i].append(str(m))

        # Plus node (stays visually with left neighbor)
        if i < len(main_list)-1:
            pid = f"_plus_{i}"
            row0.node(pid, label="<<B>+</B>>", shape="plaintext",
                      width="0.35", height="0.35",
                      fixedsize="true",
                      fontsize="16", 
                      fontname="Helvetica-Bold",
                      # fillcolor=_tint(base,0.22), color=_shade(base,0.35),
                      # penwidth="1.2",
                      group=f"col{i}")
           # row0.node(
           #      pid,
           #      label="",                  # node itself is invisible
           #      shape="point",
           #      width="0.01", height="0.01",
           #      style="invis",             # no box, no fill
           #      group=f"col{i}",
           #      xlabel="<<B>+</B>>",       # visible + without a box
           #      fontname="Helvetica-Bold",
           #      fontsize="12",
           #      fontcolor=_shade(base, 0.35))
            row0_ids.append(pid)
            main_row_nodes.append(pid)

    # Freeze the left->right ordering with invis edges
    for a, b in zip(row0_ids, row0_ids[1:]):
        row0.edge(a, b, style="invis", weight="300")
    g.subgraph(row0)

    # Connect Groups ONLY to the first main (visual line)
    if main_list:
        first_main = str(main_list[0])
        g.edge("_groups", first_main, arrowhead="none", color="#b2b2b2", penwidth="1.1")

    # ------- CHILD ROWS (rank=same per row) -------
    for r in range(max_kids):
        row = Digraph()
        row.attr(rank="same")
        row_ids = []
        for i, m in enumerate(main_list):
            base = colors[i]
            kids = children_map.get(m, [])
            if r < len(kids):
                k = str(kids[r])
                g.node(k, shape=node_shape_child, fillcolor=_tint(base,0.26),
                       color=_shade(base,0.25), penwidth="1.25", group=f"col{i}")
                row_ids.append(k)
                col_nodes[i].append(k)

                # Edge from parent or previous child downward to this child (visible)
                if r == 0:
                    g.edge(str(m), k, color=_shade(base,0.20), penwidth="1.15",
                           arrowhead="none", weight="3", constraint="true")
                else:
                    prev = str(children_map[m][r-1])
                    g.edge(prev, k, color=_shade(base,0.20), penwidth="1.0",
                           arrowhead="none", weight="2", constraint="true")
            else:
                row_ids.append(spacer(i, r))
        # lock the left->right order of this row as well
        for a, b in zip(row_ids, row_ids[1:]):
            row.edge(a, b, style="invis", weight="200")
        g.subgraph(row)

    # ------- CLUSTERS (dotted boxes) -------

    # (A) Main-row dotted box enclosing mains + '+' signs
    if main_row_nodes:
        cl_main = Digraph(name="cluster_main_row")
        cl_main.attr(style="dashed,rounded", color="#9aa3b2", penwidth="1.2", margin="0", label="")
        for nid in main_row_nodes:
            cl_main.node(nid)  # referencing existing nodes puts them in the cluster
        g.subgraph(cl_main)

    # (B) Per-column vertical dotted boxes (parent + its children), tinted to parent color
    for i, nodes in enumerate(col_nodes):
        if not nodes:  # safety
            continue
        base = colors[i]
        cl = Digraph(name=f"cluster_col_{i}")
        cl.attr(style="dashed,rounded", color=_shade(base, 0.35), penwidth="1.2", label="")
        for nid in nodes:
            cl.node(nid)
        g.subgraph(cl)

    # --- Output (inline by default)
    if render_to_file:
        return g.render(filename=filename, format=file_format, cleanup=True)
    out = g.pipe(format=notebook_format.lower())
    display(SVG(out) if notebook_format.lower()=="svg" else Image(data=out))


# ---------- Example ----------
if __name__ == "__main__":
    mains = ["A","B","C","D"]  # left -> right (top row)
    children = {
        "A": ["A1","A2"],
        "B": ["B1"],
        "C": ["C1","C2","C3"],
        "D": ["D1","D2"]
    }
    draw_lr_mains_with_vertical_children(mains, children, left_label="dx/dt", notebook_format="svg")
