import sys
import math
import time as _time
import pandas as pd
import matplotlib.pyplot as plt

from mip import Model, xsum, BINARY, CONTINUOUS, minimize
from scipy.spatial import cKDTree


def horiz_dist(p, q):
    dx = p[0] - q[0]
    dy = p[1] - q[1]
    return math.sqrt(dx * dx + dy * dy)

def euclid_dist(p, q):
    dx = p[0] - q[0]
    dy = p[1] - q[1]
    dz = p[2] - q[2]
    return math.sqrt(dx * dx + dy * dy + dz * dz)

def edge_time_energy(p, q):
    """
    Time:
      - horizontal speed 1.5 m/s
      - ascending speed 1 m/s
      - descending speed 2 m/s
      - oblique time = max(horizontal_time, vertical_time)

    Energy:
      - horizontal 10 J/m
      - ascending 50 J/m
      - descending 5 J/m
      - oblique energy = horizontal + vertical
    """
    dxy = horiz_dist(p, q)
    dz = q[2] - p[2]

    th = dxy / 1.5 if dxy > 0 else 0.0
    if dz > 0:
        tv = dz / 1.0
        ev = 50.0 * dz
    elif dz < 0:
        tv = abs(dz) / 2.0
        ev = 5.0 * abs(dz)
    else:
        tv = 0.0
        ev = 0.0

    t = max(th, tv)
    e = 10.0 * dxy + ev
    return t, e


# ------------------------------------
# Instance parameters (Edificio1 / 2)
# ------------------------------------
def instance_params(num_points):
    if num_points <= 200:
        entry_y_threshold = -12.5
        battery_wh = 1.0
        base_x = range(-8, 6)
        base_y = range(-17, -14)
    else:
        entry_y_threshold = -20.0
        battery_wh = 6.0
        base_x = range(-10, 11)
        base_y = range(-31, -29)

    battery_joule = battery_wh * 3600.0
    base_candidates = [(x, y, 0.0) for x in base_x for y in base_y]
    return entry_y_threshold, battery_joule, base_candidates


# ------------------------------------
# Build allowed edges among CSV points
# ------------------------------------
def build_point_edges(points_xyz):
    """
    Two points i, j are connected if:
      - dist <= 4
        OR
      - dist <= 11 AND at least two coordinate diffs <= 0.5
    Returns directed edges over CSV indices (i,j).
    """
    pts = points_xyz
    tree = cKDTree(pts)
    pairs = tree.query_pairs(r=11.0)

    allowed_directed = set()
    for (i, j) in pairs:
        p = pts[i]
        q = pts[j]
        d = euclid_dist(p, q)

        ok = False
        if d <= 4.0:
            ok = True
        else:
            dx = abs(p[0] - q[0])
            dy = abs(p[1] - q[1])
            dz = abs(p[2] - q[2])
            small = 0
            if dx <= 0.5: small += 1
            if dy <= 0.5: small += 1
            if dz <= 0.5: small += 1
            if d <= 11.0 and small >= 2:
                ok = True

        if ok:
            allowed_directed.add((i, j))
            allowed_directed.add((j, i))

    return allowed_directed


# ------------------------------------
# Extract ONE base-connected cycle for plotting
# ------------------------------------
def extract_route_for_trip(x_sol, k, n_nodes):
    """
    Build a walk 0-...-0 using the selected arcs of trip k.
    Works even if nodes are revisited / have multiple outgoing arcs.
    Produces an Euler-style traversal over the chosen directed edges.
    """
    # adjacency list of chosen arcs for this trip
    adj = {i: [] for i in range(n_nodes)}
    m_edges = 0
    for (i, j, kk), val in x_sol.items():
        if kk == k and val > 0.5:
            adj[i].append(j)
            m_edges += 1

    if m_edges == 0:
        return None

    # Hierholzer traversal from base (0)
    stack = [0]
    path = []
    while stack:
        v = stack[-1]
        if adj[v]:
            nxt = adj[v].pop()  # remove one unused arc v->nxt
            stack.append(nxt)
        else:
            path.append(stack.pop())

    path.reverse()

    # Must start at 0; if it doesn't, no valid base-connected walk was formed
    if not path or path[0] != 0:
        return None

    # Ideally ends at 0; if not, still return the walk we got (should be rare if model has base in/out)
    return path


# ------------------------------------
# Subtour detection + cut generation (Edificio2)
# ------------------------------------
def find_subtours_for_trip(
    k,
    n_nodes,
    t_used_var,
    y_vars,
    x_vars,
    xb_out_vars,
    xb_in_vars,
    allowed_out_by_i,
    entry_model_nodes,
    B
):
    """
    Detect directed cycles that do NOT contain base (node 0).
    Returns list of sets S ⊆ {1..N} representing subtours for trip k.
    """
    if t_used_var.x is None or t_used_var.x <= 0.5:
        return []

    active_nodes = [j for j in range(1, n_nodes) if y_vars[(j, k)].x is not None and y_vars[(j, k)].x > 0.5]

    succ = {}
    entry_set = set(entry_model_nodes)

    # base successor (0 -> some entry) if exists
    base_out = None
    for b in range(B):
        for j in entry_model_nodes:
            if xb_out_vars[(b, j, k)].x is not None and xb_out_vars[(b, j, k)].x > 0.5:
                base_out = j
                break
        if base_out is not None:
            break
    if base_out is not None:
        succ[0] = base_out

    # successor for each active node
    for i in active_nodes:
        out_j = None

        # entry node could go to base
        if i in entry_set:
            for b in range(B):
                if xb_in_vars[(b, i, k)].x is not None and xb_in_vars[(b, i, k)].x > 0.5:
                    out_j = 0
                    break

        # otherwise go to another node via x
        if out_j is None:
            for j in allowed_out_by_i.get(i, []):
                var = x_vars.get((i, j, k))
                if var is not None and var.x is not None and var.x > 0.5:
                    out_j = j
                    break

        if out_j is not None:
            succ[i] = out_j

    # find cycles
    visited_global = set()
    subtours = []
    candidates = ([0] if 0 in succ else []) + active_nodes

    for start in candidates:
        if start in visited_global:
            continue

        path_index = {}
        cur = start
        step = 0

        while True:
            if cur in visited_global:
                break
            visited_global.add(cur)

            path_index[cur] = step
            step += 1

            if cur not in succ:
                break

            nxt = succ[cur]
            if nxt in path_index:
                idx0 = path_index[nxt]
                cycle = [node for node, pos in path_index.items() if pos >= idx0]
                cycle_set = set(cycle)
                if 0 not in cycle_set:
                    cycle_set.discard(0)
                    if len(cycle_set) >= 2:
                        subtours.append(cycle_set)
                break

            cur = nxt

    # dedup
    uniq = []
    seen = set()
    for S in subtours:
        key = tuple(sorted(S))
        if key not in seen:
            seen.add(key)
            uniq.append(S)
    return uniq

# warm-start CBC
def greedy_warm_start(
    pts, N, n_nodes, K, battery_J,
    entry_model_nodes, entry_set,
    allowed_out_by_i, time_edge, energy_edge,
    time_b, energy_b
):
    """
    Build a quick feasible-ish warm start:
    - Assign nodes to trips greedily
    - Build a path per trip using nearest neighbor among allowed edges
    - Ensure battery constraint is respected
    Returns:
      start_t_used[k] in {0,1}
      start_y[(j,k)] in {0,1}
      start_x[(i,j,k)] in {0,1}
      start_xb_out[(b,j,k)], start_xb_in[(b,j,k)] in {0,1}
    Notes:
    - Assumes B=1 for Edificio2 (preselected base).
    - Uses only arcs that exist in allowed_out_by_i.
    """
    b = 0  # B=1
    base = None  # not needed explicitly because base arcs handled via time_b/energy_b

    # quick nearest entry for starting a trip: just pick entry with min time_b
    entry_sorted = sorted(entry_model_nodes, key=lambda j: time_b[(b, j)])

    unassigned = set(range(1, n_nodes))  # all nodes
    start_t_used = {k: 0 for k in range(K)}
    start_y = {}
    start_x = {}
    start_xb_out = {}
    start_xb_in = {}

    # init all starts = 0
    for k in range(K):
        for j in range(1, n_nodes):
            start_y[(j, k)] = 0

    # helper: choose next neighbor among allowed edges, preferring smaller time
    def best_next(i, candidates):
        # candidates are nodes j such that arc i->j is allowed
        # choose min time edge if exists
        best = None
        best_t = 1e100
        for j in candidates:
            t = time_edge.get((i, j), None)
            if t is None:
                continue
            if t < best_t:
                best_t = t
                best = j
        return best

    for k in range(K):
        if not unassigned:
            break

        # Start this trip from base to best entry
        # (we will later ensure we end at an entry too)
        start_entry = entry_sorted[0]
        # battery cost of base->entry and entry->base reserved (we'll add return later)
        e_base_to = energy_b[(b, start_entry)]
        e_back = energy_b[(b, start_entry)]
        budget_left = battery_J - (e_base_to + e_back)

        if budget_left <= 0:
            # battery too small (should not happen for Edificio2)
            continue

        start_t_used[k] = 1
        start_xb_out[(b, start_entry, k)] = 1
        start_xb_in[(b, start_entry, k)] = 1  # placeholder; we’ll possibly change final entry later

        # Build a path beginning at start_entry
        route_nodes = [start_entry]
        start_y[(start_entry, k)] = 1
        if start_entry in unassigned:
            unassigned.remove(start_entry)

        cur = start_entry

        # Greedily add nodes until battery runs out
        for _ in range(N):  # hard cap
            if not unassigned:
                break

            # try to go to an unassigned neighbor
            cand = [j for j in allowed_out_by_i.get(cur, []) if j in unassigned]
            if not cand:
                break

            nxt = best_next(cur, cand)
            if nxt is None:
                break

            e_move = energy_edge[(cur, nxt)]
            # Also need to ensure we can still go from nxt to SOME entry and back to base.
            # We'll enforce "end at entry" by only ending at entry node (so last->base is xb_in).
            # For safety, require nxt can reach some entry (maybe itself) via allowed edges (weak check):
            # We'll just reserve a conservative return via start_entry (same as already reserved).
            if e_move <= budget_left:
                start_x[(cur, nxt, k)] = 1
                start_y[(nxt, k)] = 1
                unassigned.remove(nxt)
                budget_left -= e_move
                route_nodes.append(nxt)
                cur = nxt
            else:
                break

        # Ensure we end at an entry node
        # If current is not entry, try to hop to a nearby entry if possible within battery
        if cur not in entry_set:
            # try to find an entry reachable from cur
            cand_entries = [j for j in allowed_out_by_i.get(cur, []) if j in entry_set]
            if cand_entries:
                end_entry = best_next(cur, cand_entries)
                if end_entry is not None:
                    e_move = energy_edge[(cur, end_entry)]
                    if e_move <= budget_left:
                        start_x[(cur, end_entry, k)] = 1
                        start_y[(end_entry, k)] = 1
                        if end_entry in unassigned:
                            unassigned.remove(end_entry)
                        cur = end_entry

        # Now set base return to the actual last entry we ended at (must be entry)
        # If still not entry, we keep the original start_entry as end (not perfect but gives warm start).
        end_entry = cur if cur in entry_set else start_entry
        # Update xb_in choice: zero out old and set new
        start_xb_in[(b, start_entry, k)] = 0
        start_xb_in[(b, end_entry, k)] = 1

    # Fill missing base vars with zeros
    # (caller will use .get(...,0))
    return start_t_used, start_y, start_x, start_xb_out, start_xb_in

# -----------------------------
# Main solve routine
# -----------------------------
def solve_instance(csv_path):
    df = pd.read_csv(csv_path)
    pts = df[["x", "y", "z"]].to_numpy().tolist()
    N = len(pts)
    n_nodes = N + 1  # base is 0

    entry_y_thr, battery_J, base_candidates = instance_params(N)

    entry_points = [i for i, p in enumerate(pts) if p[1] <= entry_y_thr]
    if not entry_points:
        raise ValueError("No entry points found. Check entry threshold logic / dataset.")
    entry_model_nodes = [i + 1 for i in entry_points]
    entry_set = set(entry_model_nodes)

    # --------- Edificio2: preselect base to shrink model massively ----------
    if N > 200:
        entry_coords = [pts[i] for i in entry_points]

        def base_score(bcoord):
            step = max(1, len(entry_coords) // 60)
            sample = entry_coords[::step]
            return sum(edge_time_energy(bcoord, p)[0] for p in sample) / len(sample)

        best_base = min(base_candidates, key=base_score)
        base_candidates = [best_base]
        print(f"[DEBUG] Preselected base for Edificio2: {best_base}")

    # --------- Build allowed edges ----------
    allowed_pairs = build_point_edges(pts)
    allowed_edges = set((a + 1, b + 1) for (a, b) in allowed_pairs)

    # --------- Edificio2: prune outgoing edges ONCE ----------
    if N > 200:
        MAX_OUT = 25
        out = {i: [] for i in range(1, n_nodes)}
        for (i, j) in allowed_edges:
            out[i].append(j)

        pruned = set()
        for i in range(1, n_nodes):
            nbrs = out[i]
            if len(nbrs) <= MAX_OUT:
                for j in nbrs:
                    pruned.add((i, j))
            else:
                pi = pts[i - 1]
                nbrs_sorted = sorted(nbrs, key=lambda j: edge_time_energy(pi, pts[j - 1])[0])
                for j in nbrs_sorted[:MAX_OUT]:
                    pruned.add((i, j))
        allowed_edges = pruned

    E = len(allowed_edges)
    avg_deg = E / max(1, N)
    print(f"[DEBUG] N(points)={N}, directed_edges={E}, avg_out_degree≈{avg_deg:.2f}")

    # adjacency for faster subtour cuts
    allowed_out_by_i = {i: [] for i in range(1, n_nodes)}
    for (i, j) in allowed_edges:
        allowed_out_by_i[i].append(j)

    # Precompute time/energy for non-base edges
    time_edge = {}
    energy_edge = {}
    for (i, j) in allowed_edges:
        t, e = edge_time_energy(pts[i - 1], pts[j - 1])
        time_edge[(i, j)] = t
        energy_edge[(i, j)] = e

    # base<->entry edge costs
    time_b = {}
    energy_b = {}
    for b_idx, bcoord in enumerate(base_candidates):
        for j in entry_model_nodes:
            t, e = edge_time_energy(bcoord, pts[j - 1])
            time_b[(b_idx, j)] = t
            energy_b[(b_idx, j)] = e

    # Choose K
    K = 8 if N <= 200 else 20

    # -----------------------------
    # Build MIP Model
    # -----------------------------
    m = Model(sense=minimize)
    m.verbose = 1

    # Base selection
    B = len(base_candidates)
    bsel = [m.add_var(var_type=BINARY, name=f"bsel[{b}]") for b in range(B)]
    m += xsum(bsel[b] for b in range(B)) == 1

    # Trip used or not
    t_used = [m.add_var(var_type=BINARY, name=f"t[{k}]") for k in range(K)]

    # Symmetry breaking: use trips in order
    for k in range(K - 1):
        m += t_used[k] >= t_used[k + 1]

    # Visit indicator y[j,k]
    y = {}
    for j in range(1, n_nodes):
        for k in range(K):
            y[(j, k)] = m.add_var(var_type=BINARY, name=f"y[{j},{k}]")

    # Strong symmetry breaking: non-increasing visited count
    for k in range(K - 1):
        m += xsum(y[(j, k)] for j in range(1, n_nodes)) >= xsum(y[(j, k + 1)] for j in range(1, n_nodes))

    # Edge decision variables x[i,j,k]
    x = {}
    for (i, j) in allowed_edges:
        for k in range(K):
            x[(i, j, k)] = m.add_var(var_type=BINARY, name=f"x[{i},{j},{k}]")

    # Base-edge variables xb_out/xb_in
    xb_out = {}
    xb_in = {}
    for b in range(B):
        for j in entry_model_nodes:
            for k in range(K):
                xb_out[(b, j, k)] = m.add_var(var_type=BINARY, name=f"xb_out[{b},{j},{k}]")
                xb_in[(b, j, k)] = m.add_var(var_type=BINARY, name=f"xb_in[{b},{j},{k}]")
                m += xb_out[(b, j, k)] <= bsel[b]
                m += xb_in[(b, j, k)] <= bsel[b]

    # Trip start/end: if used, exactly one out of base and one into base
    for k in range(K):
        m += xsum(xb_out[(b, j, k)] for b in range(B) for j in entry_model_nodes) == t_used[k]
        m += xsum(xb_in[(b, j, k)] for b in range(B) for j in entry_model_nodes) == t_used[k]

    # Degree linking (RELAXED):
    # If y[j,k]=0 -> no edges touch j in trip k
    # If y[j,k]=1 -> at least one "enter" and at least one "leave" (but can be >1 to allow backtracking)
    for k in range(K):
        for j in range(1, n_nodes):

            in_base = xsum(xb_out[(b, j, k)] for b in range(B)) if j in entry_set else 0
            out_base = xsum(xb_in[(b, j, k)] for b in range(B)) if j in entry_set else 0

            in_edges = xsum(x[(i, j, k)] for (i, jj) in allowed_edges if jj == j)
            out_edges = xsum(x[(j, i, k)] for (jj, i) in allowed_edges if jj == j)

            indeg = in_base + in_edges
            outdeg = out_base + out_edges

            # link to y
            m += indeg >= y[(j, k)]
            m += outdeg >= y[(j, k)]

            # upper bounds: if y=0 then indeg=outdeg=0; if y=1 then can be multiple
            # use MAX_OUT as a safe-ish cap for non-entry; for entry allow +1 for base arc
            cap_out = (len(allowed_out_by_i.get(j, [])) + (1 if j in entry_set else 0))
            cap_in  = cap_out  # crude but ok; graph is directed but similar order

            m += indeg <= cap_in * y[(j, k)]
            m += outdeg <= cap_out * y[(j, k)]

            # can't visit a node if trip not used
            m += y[(j, k)] <= t_used[k]


    # Coverage: every point visited at least once
    for j in range(1, n_nodes):
        m += xsum(y[(j, k)] for k in range(K)) >= 1

    # Battery per trip
    for k in range(K):
        expr = []
        expr.append(xsum(energy_edge[(i, j)] * x[(i, j, k)] for (i, j) in allowed_edges))
        expr.append(xsum(energy_b[(b, j)] * xb_out[(b, j, k)] for b in range(B) for j in entry_model_nodes))
        expr.append(xsum(energy_b[(b, j)] * xb_in[(b, j, k)] for b in range(B) for j in entry_model_nodes))
        m += xsum(expr) <= battery_J * t_used[k]

    # Objective: minimize total time
    obj_terms = []
    for k in range(K):
        obj_terms.append(xsum(time_edge[(i, j)] * x[(i, j, k)] for (i, j) in allowed_edges))
        obj_terms.append(xsum(time_b[(b, j)] * xb_out[(b, j, k)] for b in range(B) for j in entry_model_nodes))
        obj_terms.append(xsum(time_b[(b, j)] * xb_in[(b, j, k)] for b in range(B) for j in entry_model_nodes))
    m.objective = xsum(obj_terms)

    # --------- Connectivity (FLOW) for ALL instances ----------
    # This kills subtours without needing cut loops.
    BIGF = n_nodes  # safe upper bound

    f = {}
    for (i, j) in allowed_edges:
        for k in range(K):
            f[(i, j, k)] = m.add_var(var_type=CONTINUOUS, lb=0.0, name=f"f[{i},{j},{k}]")
            m += f[(i, j, k)] <= BIGF * x[(i, j, k)]

    fb_out = {}
    fb_in = {}
    for b in range(B):
        for j in entry_model_nodes:
            for k in range(K):
                fb_out[(b, j, k)] = m.add_var(var_type=CONTINUOUS, lb=0.0, name=f"fb_out[{b},{j},{k}]")
                fb_in[(b, j, k)] = m.add_var(var_type=CONTINUOUS, lb=0.0, name=f"fb_in[{b},{j},{k}]")
                m += fb_out[(b, j, k)] <= BIGF * xb_out[(b, j, k)]
                m += fb_in[(b, j, k)] <= BIGF * xb_in[(b, j, k)]

    # Flow conservation: each visited node consumes 1 unit of flow
    for k in range(K):
        for j in range(1, n_nodes):
            inflow = xsum(f[(i, j, k)] for (i, jj) in allowed_edges if jj == j)
            outflow = xsum(f[(j, i, k)] for (jj, i) in allowed_edges if jj == j)

            if j in entry_set:
                inflow += xsum(fb_out[(b, j, k)] for b in range(B))
                outflow += xsum(fb_in[(b, j, k)] for b in range(B))

            m += inflow - outflow == y[(j, k)]

        # base sends exactly total number of visited nodes units
        base_out = xsum(fb_out[(b, j, k)] for b in range(B) for j in entry_model_nodes)
        base_in  = xsum(fb_in[(b, j, k)] for b in range(B) for j in entry_model_nodes)
        m += base_out - base_in == xsum(y[(j, k)] for j in range(1, n_nodes))
    
    # -----------------------------
    # WARM START (python-mip style) — apply once (no cut rounds anymore)
    # -----------------------------
    if N > 200:
        ws_t, ws_y, ws_x, ws_xbout, ws_xbin = greedy_warm_start(
            pts=pts, N=N, n_nodes=n_nodes, K=K, battery_J=battery_J,
            entry_model_nodes=entry_model_nodes, entry_set=entry_set,
            allowed_out_by_i=allowed_out_by_i,
            time_edge=time_edge, energy_edge=energy_edge,
            time_b=time_b, energy_b=energy_b
        )

        mipstart = []

        # Trip used (ONLY the 1's)
        for k in range(K):
            if ws_t.get(k, 0) == 1:
                mipstart.append((t_used[k], 1))

        # Visits (ONLY the 1's)
        for (j, k), val in ws_y.items():
            if val == 1:
                mipstart.append((y[(j, k)], 1))

        # Non-base arcs (ONLY the 1's)
        for (i, j, k), val in ws_x.items():
            if val == 1:
                var = x.get((i, j, k), None)
                if var is not None:
                    mipstart.append((var, 1))

        # Base arcs (ONLY the 1's)
        b = 0  # Edificio2 has B=1 after base preselect
        for (bb, j, k), val in ws_xbout.items():
            if val == 1 and bb == b:
                mipstart.append((xb_out[(b, j, k)], 1))
        for (bb, j, k), val in ws_xbin.items():
            if val == 1 and bb == b:
                mipstart.append((xb_in[(b, j, k)], 1))

        # APPLY IT DIRECTLY
        m.start = mipstart
        print(f"[DEBUG] Warm start applied. #start_assignments={len(mipstart)}")
    # -----------------------------
    # Solve (single run — flow already prevents subtours)
    # -----------------------------
    if N <= 200:
        m.max_seconds = 900
    else:
        m.max_seconds = 3600  # you can increase if needed

    status = m.optimize()

    print(f"Solve status: {status}")
    print(f"Objective (total time): {m.objective_value}")
    # --------- STOP CLEANLY if no feasible solution ----------
    if status in (None, ) or m.objective_value is None:
        print("\n[INFO] CBC did not find a feasible integer solution within the time limit.")
        print("[INFO] Increase time, reduce K, prune more edges, or add a warm-start heuristic.\n")
        return

    # Recover chosen base
    chosen_b = None
    for b in range(B):
        if bsel[b].x is not None and bsel[b].x > 0.5:
            chosen_b = b
            break
    if chosen_b is None:
        print("[INFO] No base selected because no feasible solution exists yet.")
        return
    
    base_coord = base_candidates[chosen_b]
    print(f"Chosen base candidate index: {chosen_b}, coord={base_coord}")

    # Coverage check
    visited = set()
    for j in range(1, n_nodes):
        if sum(y[(j, k)].x for k in range(K) if y[(j, k)].x is not None) > 0.5:
            visited.add(j)
    missing = sorted(set(range(1, n_nodes)) - visited)
    print("Visited points:", len(visited))
    print("Missing points:", missing)

    # Trip time/energy reporting
    trip_stats = []
    for k in range(K):
        if t_used[k].x is None or t_used[k].x < 0.5:
            continue

        trip_time = 0.0
        trip_energy = 0.0

        for (i, j) in allowed_edges:
            if x[(i, j, k)].x is not None and x[(i, j, k)].x > 0.5:
                trip_time += time_edge[(i, j)]
                trip_energy += energy_edge[(i, j)]

        for b in range(B):
            for j in entry_model_nodes:
                if xb_out[(b, j, k)].x is not None and xb_out[(b, j, k)].x > 0.5:
                    trip_time += time_b[(b, j)]
                    trip_energy += energy_b[(b, j)]
                if xb_in[(b, j, k)].x is not None and xb_in[(b, j, k)].x > 0.5:
                    trip_time += time_b[(b, j)]
                    trip_energy += energy_b[(b, j)]

        ok = (trip_energy <= battery_J + 1e-6)
        trip_stats.append((k, trip_time, trip_energy, ok))

    for idx, (k, tt, ee, ok) in enumerate(trip_stats, start=1):
        print(f"Trip {idx} (k={k}): time={tt:.2f}s, energy={ee:.2f} J, battery_ok={ok}")

    # Collect x solution for plotting / route extraction
    x_sol = {}
    for (i, j, k), var in x.items():
        if var.x is not None and var.x > 0.5:
            x_sol[(i, j, k)] = 1.0
    for b in range(B):
        for j in entry_model_nodes:
            for k in range(K):
                if xb_out[(b, j, k)].x is not None and xb_out[(b, j, k)].x > 0.5:
                    x_sol[(0, j, k)] = 1.0
                if xb_in[(b, j, k)].x is not None and xb_in[(b, j, k)].x > 0.5:
                    x_sol[(j, 0, k)] = 1.0

    # Plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    zs = [p[2] for p in pts]
    ax.scatter(xs, ys, zs, s=10)
    ax.scatter([base_coord[0]], [base_coord[1]], [base_coord[2]], s=60, marker="^")

    trip_num = 0
    for k in range(K):
        if t_used[k].x is None or t_used[k].x < 0.5:
            continue

        route = extract_route_for_trip(x_sol, k, n_nodes)
        if not route:
            continue

        trip_num += 1
        print(f"Viaggio {trip_num}: " + "-".join(map(str, route)))

        def node_coord(node_id):
            return base_coord if node_id == 0 else pts[node_id - 1]

        # plot the walk as a polyline
        rx, ry, rz = [], [], []
        for nid in route:
            c = node_coord(nid)
            rx.append(c[0]); ry.append(c[1]); rz.append(c[2])
        ax.plot(rx, ry, rz, color=colors[(trip_num-1) % len(colors)])



    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    plt.show()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python main.py <input_csv>")
        sys.exit(1)
    solve_instance(sys.argv[1])
