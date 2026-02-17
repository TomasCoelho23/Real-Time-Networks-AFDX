################################################################@
"""
AFDX Network Calculus - Step 2: System Design & Stability Verification

Refactored from __base.py into an Object-Oriented architecture with:
  - Node hierarchy (Station / Switch)
  - Edge with full-duplex directional load tracking
  - Flow with AFDX bandwidth calculation
  - Network container (replaces global lists)
  - Strategy Pattern for routing (stub)
"""
################################################################@

import xml.etree.ElementTree as ET
import os.path
import sys
from abc import ABC, abstractmethod

################################################################@
#  Phase 1 — Core Data Model
################################################################@

# ---------- Node hierarchy ---------- #

class Node(ABC):
    """Abstract base class for every network node."""

    def __init__(self, name: str, service_policy: str = "FIRST_IN_FIRST_OUT"):
        self.name = name
        self.service_policy = service_policy

    @abstractmethod
    def is_switch(self) -> bool:
        ...

    def __repr__(self):
        return f"{self.__class__.__name__}({self.name!r})"


class Station(Node):
    """End-system station."""

    def __init__(self, name: str, service_policy: str = "FIRST_IN_FIRST_OUT"):
        super().__init__(name, service_policy)

    def is_switch(self) -> bool:
        return False


class Switch(Node):
    """AFDX switch with a technological latency."""

    def __init__(self, name: str, latency: float = 0.0,
                 service_policy: str = "FIRST_IN_FIRST_OUT"):
        super().__init__(name, service_policy)
        self.latency = latency          # seconds

    def is_switch(self) -> bool:
        return True


# ---------- Edge (physical link) ---------- #

class Edge:
    """
    A physical full-duplex link between two nodes.

    Tracks load in BOTH directions independently:
      - load_direct  : traffic flowing  source  →  dest
      - load_reverse : traffic flowing  dest    →  source
    """

    def __init__(self, name: str, source: Node, dest: Node,
                 src_port: int = 0, dest_port: int = 0,
                 capacity: float = 100_000_000):
        self.name = name
        self.source = source            # Node object
        self.dest = dest                # Node object
        self.src_port = src_port
        self.dest_port = dest_port
        self.capacity = capacity        # bits per second

        self.load_direct = 0.0          # bps,  source → dest
        self.load_reverse = 0.0         # bps,  dest   → source

    def add_load(self, amount: float, direction_source_node: Node) -> None:
        """Add *amount* (bps) in the direction that starts from
        *direction_source_node*."""
        if direction_source_node is self.source:
            self.load_direct += amount
        else:
            self.load_reverse += amount

    def __repr__(self):
        return (f"Edge({self.name!r}, {self.source.name}:{self.src_port} "
                f"→ {self.dest.name}:{self.dest_port}, "
                f"C={self.capacity:.0f} bps)")


# ---------- Target ---------- #

class Target:
    """One destination inside a (possibly multicast) flow."""

    def __init__(self, flow: "Flow", to: str):
        self.flow = flow
        self.to = to
        self.path: list[str] = []       # ordered node names from source → dest

    def __repr__(self):
        return f"Target({self.to!r}, path={self.path})"


# ---------- Flow (Virtual Link) ---------- #

class ArrivalCurve:
    """Affine arrival curve  α(t) = σ + ρ·t.

    σ (sigma) — burst, in **bits**
    ρ (rho)   — long-term rate, in **bits per second**

    Supports aggregation via  +  operator:
        (σ_A + σ_B,  ρ_A + ρ_B)
    """

    def __init__(self, sigma: float, rho: float):
        self.sigma = sigma      # bits
        self.rho = rho          # bps

    def __add__(self, other: "ArrivalCurve") -> "ArrivalCurve":
        return ArrivalCurve(self.sigma + other.sigma,
                            self.rho + other.rho)

    def __repr__(self):
        return f"ArrivalCurve(σ={self.sigma:.2f} bits, ρ={self.rho:.2f} bps)"


class Flow:
    """
    An AFDX Virtual Link.

    Bandwidth formula:
        BW = (payload + overhead) * 8 / period      [bps]

    where payload & overhead are in bytes, period in seconds.
    """

    def __init__(self, name: str, source: str,
                 payload: float, overhead: float, period: float):
        self.name = name
        self.source = source            # source node name
        self.payload = payload          # bytes
        self.overhead = overhead        # bytes
        self.period = period            # seconds  (already converted from ms)
        self.targets: list[Target] = [] # ← instance-level list (fixes class-var bug)

        # --- Arrival curve initial state (Step 3) ---
        self.initial_sigma: float = (payload + overhead) * 8.0   # bits
        self.initial_rho: float = self.get_bandwidth()           # bps

        # End-to-end delay results: target_name → delay in seconds
        self.delays_per_target: dict[str, float] = {}

    def max_frame_bits(self) -> float:
        """Maximum frame size in bits (payload + overhead)."""
        return (self.payload + self.overhead) * 8.0

    def get_bandwidth(self) -> float:
        """Return the bandwidth consumed by this VL in bits per second."""
        return (self.payload + self.overhead) * 8.0 / self.period

    def __repr__(self):
        return (f"Flow({self.name!r}, src={self.source}, "
                f"BW={self.get_bandwidth():.2f} bps)")


################################################################@
#  Phase 3 — Strategy Pattern for Routing (stub)
################################################################@

class IRoutingStrategy(ABC):
    """Interface for path-computation strategies."""

    @abstractmethod
    def compute_path(self, network: "Network",
                     source: str, dest: str) -> list[str]:
        """Return an ordered list of node names from *source* to *dest*
        (excluding the source, including the destination)."""
        ...


class DijkstraStrategy(IRoutingStrategy):
    """Stub — prints a notice and returns an empty path."""

    def compute_path(self, network: "Network",
                     source: str, dest: str) -> list[str]:
        print(f"  [DijkstraStrategy] Using Dijkstra to route "
              f"{source} → {dest}")
        return []


################################################################@
#  Phase 2 — Network Container
################################################################@

class Network:
    """
    Central container that owns the full graph state.

    Replaces the old global `nodes`, `edges`, `flows` lists.
    """

    def __init__(self):
        # --- network-level attributes (from XML <network> tag) ---
        self.name: str = ""
        self.overhead: int = 67                          # bytes
        self.default_capacity: float = 100_000_000       # bps
        self.shortest_path_policy: str = "DIJKSTRA"

        # --- graph data ---
        self.nodes: dict[str, Node] = {}                 # name → Node
        self.edges: list[Edge] = []
        self.flows: list[Flow] = []

        # --- routing strategy (Strategy Pattern) ---
        self.routing_strategy: IRoutingStrategy = DijkstraStrategy()

    # ---- helpers ---- #

    def add_node(self, node: Node) -> None:
        self.nodes[node.name] = node

    def get_node(self, name: str) -> Node:
        """Lookup a node by name. Raises KeyError if not found."""
        return self.nodes[name]

    def get_edge(self, node_a_name: str, node_b_name: str) -> Edge | None:
        """Find the edge connecting *node_a_name* and *node_b_name*
        regardless of which end is 'source' vs 'dest' in the Edge object."""
        for e in self.edges:
            if ((e.source.name == node_a_name and e.dest.name == node_b_name) or
                    (e.source.name == node_b_name and e.dest.name == node_a_name)):
                return e
        return None

    # ---- pretty-print (replaces old traceNetwork) ---- #

    def trace(self) -> None:
        print("Stations:")
        for n in self.nodes.values():
            if not n.is_switch():
                print(f"\t{n.name}  (policy={n.service_policy})")

        print("\nSwitches:")
        for n in self.nodes.values():
            if n.is_switch():
                print(f"\t{n.name}  (latency={n.latency} s, "
                      f"policy={n.service_policy})")

        print("\nEdges:")
        for e in self.edges:
            print(f"\t{e.name}: {e.source.name}:{e.src_port} "
                  f"→ {e.dest.name}:{e.dest_port}  "
                  f"(C={e.capacity:.0f} bps)")

        print("\nFlows:")
        for f in self.flows:
            print(f"\t{f.name}: src={f.source}  "
                  f"(L={f.payload}, OH={f.overhead}, "
                  f"p={f.period} s, BW={f.get_bandwidth():.2f} bps)")
            for t in f.targets:
                print(f"\t\tTarget={t.to}  path={t.path}")

    # ---- Phase 5: load calculation & stability ---- #

    def calculate_loads(self) -> None:
        """Walk every flow/target path and accumulate bandwidth on edges.

        **MULTICAST HANDLING:** In AFDX, a frame is sent ONCE from the source,
        and switches replicate it. So if multiple targets share a common link,
        we only count the bandwidth ONCE for that (edge, direction) pair.

        For each consecutive pair (A → B) in a target's path, locate the
        physical Edge and call ``edge.add_load(bw, node_A)`` — but only if
        this (edge, direction) hasn't been visited for this flow yet.
        """
        for flow in self.flows:
            bw = flow.get_bandwidth()
            
            # Track which (edge, direction) pairs have been visited for THIS flow
            visited_links_for_this_flow = set()

            for target in flow.targets:
                path = target.path          # e.g. ['Source', 'Switch', 'Dest1']
                for i in range(len(path) - 1):
                    node_a_name = path[i]
                    node_b_name = path[i + 1]

                    edge = self.get_edge(node_a_name, node_b_name)
                    if edge is None:
                        print(f"  [WARNING] No edge between "
                              f"{node_a_name} and {node_b_name} "
                              f"(flow={flow.name}, target={target.to})")
                        continue

                    # Determine the direction: node_a is the sender
                    node_a = self.get_node(node_a_name)
                    
                    # Create a unique key for this (edge, direction) pair
                    # Direction is determined by comparing node_a to edge.source
                    is_direct = (node_a is edge.source)
                    link_key = (edge, is_direct)
                    
                    # Only add load if we haven't visited this link in this direction
                    # for this flow yet (multicast tree aggregation)
                    if link_key not in visited_links_for_this_flow:
                        edge.add_load(bw, node_a)
                        visited_links_for_this_flow.add(link_key)

    def check_stability(self) -> bool:
        """Verify ρ < 1  (load < capacity) on every edge direction.

        Returns True if the network is stable, False otherwise.
        Prints a warning for each overloaded direction.
        """
        stable = True
        for e in self.edges:
            # direct direction
            if e.load_direct >= e.capacity:
                print(f"  [UNSTABLE] {e.name} direct "
                      f"({e.source.name}→{e.dest.name}): "
                      f"load={e.load_direct:.2f} bps >= "
                      f"capacity={e.capacity:.0f} bps  "
                      f"(ρ={e.load_direct / e.capacity:.4f})")
                stable = False
            # reverse direction
            if e.load_reverse >= e.capacity:
                print(f"  [UNSTABLE] {e.name} reverse "
                      f"({e.dest.name}→{e.source.name}): "
                      f"load={e.load_reverse:.2f} bps >= "
                      f"capacity={e.capacity:.0f} bps  "
                      f"(ρ={e.load_reverse / e.capacity:.4f})")
                stable = False

        if stable:
            print("  [OK] Network is stable (ρ < 1 on all links).")
        return stable

    # ---- Step 3: end-to-end delay analysis ---- #

    def calculate_delays(self) -> None:
        """Compute worst-case end-to-end path delay bounds using Network Calculus.

        Algorithm — Hop-by-hop feed-forward iteration:

        For each hop index (0 .. max_hops-1):
          1. Identify every (edge, direction) that is the hop-th link
             for at least one (flow, target).
          2. On each such (edge, direction):
             a. AGGREGATE the arrival curves of all *distinct flows*
                (multicast dedup — a flow counts once even if multiple
                targets share the edge at this hop).
             b. Compute service curve parameters:
                  R = edge capacity
                  T = L_max/R if upstream is a Station (source serialization)
                      OR switch technological latency if upstream is a Switch.
                      This policy is kept compatible with Open-Timaeus
                      path-delay references for this course project.
             c. Delay bound:  D = σ_agg / R + T
             d. Update every (flow, target) state on this edge:
                  - accumulated_delay += D
                  - σ_new = σ_old + ρ_old × D   (output burstiness theorem)

        Export note:
            ``<target value="...">`` in the output XML is this worst-case
            end-to-end path delay expressed in microseconds (µs).
        """
        # Open-Timaeus compatibility toggles for this project profile.
        SERIALIZATION_AT_SOURCE_ONLY = True
        USE_FULL_FRAME_SIZE = True
        

        # --- Build per-(flow, target) state ---
        # state key = (flow object, target object)
        # state val = { "curve": ArrivalCurve, "delay": float }
        states: dict[tuple[Flow, Target], dict] = {}
        for flow in self.flows:
            for target in flow.targets:
                states[(flow, target)] = {
                    "curve": ArrivalCurve(flow.initial_sigma, flow.initial_rho),
                    "delay": 0.0,
                }

        # --- Determine max hops ---
        max_hops = 0
        for flow in self.flows:
            for target in flow.targets:
                # number of edges = len(path) - 1
                hops = len(target.path) - 1
                if hops > max_hops:
                    max_hops = hops

        # --- Iterate hop by hop ---
        for hop in range(max_hops):

            # 1. Collect (edge, direction, upstream_node_name) for this hop,
            #    along with all (flow, target) pairs active on it.
            #    Key: (edge, is_direct)
            #    Value: list of (flow, target) pairs  +  upstream node name
            edge_groups: dict[tuple[Edge, bool], list[tuple[Flow, Target, str]]] = {}

            for flow in self.flows:
                for target in flow.targets:
                    path = target.path
                    if hop >= len(path) - 1:
                        continue  # this target has fewer hops

                    node_a_name = path[hop]
                    node_b_name = path[hop + 1]

                    edge = self.get_edge(node_a_name, node_b_name)
                    if edge is None:
                        continue

                    node_a = self.get_node(node_a_name)
                    is_direct = (node_a is edge.source)
                    key = (edge, is_direct)

                    if key not in edge_groups:
                        edge_groups[key] = []
                    edge_groups[key].append((flow, target, node_a_name))

            # 2. Process each (edge, direction) group
            for (edge, is_direct), ft_list in edge_groups.items():

                R = edge.capacity  # bps

                # --- Aggregate arrival curves (dedup by flow) ---
                seen_flows: set[Flow] = set()
                agg_curve = ArrivalCurve(0.0, 0.0)
                l_max = 0.0  # max frame size in bits among flows on this edge

                for flow, target, upstream_name in ft_list:
                    if flow not in seen_flows:
                        seen_flows.add(flow)
                        state = states[(flow, target)]
                        agg_curve = agg_curve + state["curve"]
                        if USE_FULL_FRAME_SIZE:
                            frame_bits = flow.max_frame_bits()
                        else:
                            frame_bits = flow.payload * 8.0
                        if frame_bits > l_max:
                            l_max = frame_bits

                # --- Service curve parameters ---
                # Upstream node's switch latency (0 if station)
                # Use the first entry's upstream name (same for all in group)
                upstream_name = ft_list[0][2]
                upstream_node = self.get_node(upstream_name)

                # 1. Switches: Only pay technological latency (Serialization is in sigma/R)
                # 2. Stations: Pay L_max/R (Initial serialization onto the wire)
                if upstream_node.is_switch():
                    T = upstream_node.latency
                else:
                    T = l_max / R
                    
                # --- Delay bound ---
                D = agg_curve.sigma / R + T

                # --- Update every (flow, target) state ---
                for flow, target, _ in ft_list:
                    s = states[(flow, target)]
                    s["delay"] += D
                    # Output burstiness theorem: σ_new = σ_old + ρ_old × D
                    s["curve"] = ArrivalCurve(
                        s["curve"].sigma + s["curve"].rho * D,
                        s["curve"].rho,  # rho stays the same
                    )

        # --- Write results back to Flow objects ---
        for (flow, target), state in states.items():
            flow.delays_per_target[target.to] = max(
                0.0, state["delay"] 
            )

    # ---- Phase 6: XML output ---- #

    def export_results(self, filename: str) -> None:
        """Generate an XML results file.
        
        Structure:
        <results>
            <delays>
                <flow name="…">
                    <target name="…" value="[µs]" />
                </flow>
            </delays>
            <load>
                <edge name="…">
                    <usage type="direct"  value="…" percent="…" />
                    <usage type="reverse" value="…" percent="…" />
                </edge>
            </load>
        </results>
        """
        with open(filename, "w", encoding="utf-8") as f:
            f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
            f.write('<results>\n')

            # --- Section 1: DELAYS (Step 3) ---
            f.write('\t<delays>\n')
            for flow in self.flows:
                if flow.delays_per_target:
                    f.write(f'\t\t<flow name="{flow.name}">\n')
                    for target in flow.targets:
                        delay_us = flow.delays_per_target.get(target.to, 0.0) * 1e6
                        f.write(f'\t\t\t<target name="{target.to}" '
                                f'value="{delay_us:.2f}" />\n')
                    f.write(f'\t\t</flow>\n')
            f.write('\t</delays>\n')

            # --- Section 2: LOAD (Step 2) ---
            f.write('\t<load>\n')
            
            for e in self.edges:
                # Avoid division by zero
                if e.capacity > 0:
                    pct_direct = (e.load_direct / e.capacity) * 100.0
                    pct_reverse = (e.load_reverse / e.capacity) * 100.0
                else:
                    pct_direct = 0.0
                    pct_reverse = 0.0

                f.write(f'\t\t<edge name="{e.name}">\n')
                # Direct Direction
                f.write(f'\t\t\t<usage type="direct" '
                        f'value="{e.load_direct:.0f}" '
                        f'percent="{pct_direct:.2f}%" />\n')
                # Reverse Direction
                f.write(f'\t\t\t<usage type="reverse" '
                        f'value="{e.load_reverse:.0f}" '
                        f'percent="{pct_reverse:.2f}%" />\n')
                f.write(f'\t\t</edge>\n')

            f.write('\t</load>\n')
            f.write('</results>\n')

################################################################@
#  Phase 4 — XML Parsing
################################################################@

def _parse_capacity(raw: str | None, default: float = 100_000_000) -> float:
    """Convert a transmission-capacity string to bits-per-second.

    Accepted formats:
        • Pure numeric:  "100000000"
        • With suffix:   "100Mbps", "10Mbps", "1Gbps"
    Falls back to *default* when the attribute is missing.
    """
    if raw is None:
        return default
    raw = raw.strip()
    # Try pure number first
    try:
        return float(raw)
    except ValueError:
        pass
    # Handle common suffixed forms
    lower = raw.lower()
    if lower.endswith("gbps"):
        return float(raw[:-4]) * 1e9
    if lower.endswith("mbps"):
        return float(raw[:-4]) * 1e6
    if lower.endswith("kbps"):
        return float(raw[:-4]) * 1e3
    # Last resort
    return default


def _parse_network_attributes(root: ET.Element, net: Network) -> None:
    """Read the <network> tag and populate *net*'s global attributes."""
    elem = root.find("network")
    if elem is None:
        return
    net.name = elem.get("name", "")
    net.overhead = int(elem.get("overhead", "67"))
    net.shortest_path_policy = elem.get("shortest-path-policy", "DIJKSTRA")
    net.default_capacity = _parse_capacity(
        elem.get("transmission-capacity"), 100_000_000
    )


def _parse_stations(root: ET.Element, net: Network) -> None:
    """Parse every <station> tag and add Station nodes to the network."""
    for el in root.findall("station"):
        name = el.get("name")
        policy = el.get("service-policy", "FIRST_IN_FIRST_OUT")
        net.add_node(Station(name, service_policy=policy))


def _parse_switches(root: ET.Element, net: Network) -> None:
    """Parse every <switch> tag and add Switch nodes to the network."""
    for el in root.findall("switch"):
        name = el.get("name")
        latency = float(el.get("tech-latency", "0")) * 1e-6   # µs → s
        policy = el.get("service-policy", "FIRST_IN_FIRST_OUT")
        net.add_node(Switch(name, latency=latency, service_policy=policy))


def _parse_edges(root: ET.Element, net: Network) -> None:
    """Parse every <link> tag and create Edge objects with Node references."""
    for el in root.findall("link"):
        name = el.get("name")
        src_node = net.get_node(el.get("from"))
        dst_node = net.get_node(el.get("to"))
        src_port = int(el.get("fromPort", "0"))
        dst_port = int(el.get("toPort", "0"))
        capacity = _parse_capacity(
            el.get("transmission-capacity"), net.default_capacity
        )
        net.edges.append(
            Edge(name, src_node, dst_node,
                 src_port=src_port, dest_port=dst_port,
                 capacity=capacity)
        )


def _parse_flows(root: ET.Element, net: Network) -> None:
    """Parse every <flow> tag, including multicast targets and paths.

    Path convention kept from __base.py:
        target.path = [source, node1, …, destination]
    If no <path> tags are present, the routing strategy is invoked.
    """
    for el in root.findall("flow"):
        name = el.get("name")
        source = el.get("source")
        payload = float(el.get("max-payload", "0"))
        period = float(el.get("period", "1")) * 1e-3   # ms → s

        flow = Flow(name, source,
                    payload=payload,
                    overhead=net.overhead,
                    period=period)
        net.flows.append(flow)

        for tg in el.findall("target"):
            target = Target(flow, tg.get("name"))
            flow.targets.append(target)

            # Collect explicit <path> nodes (if any)
            path_nodes = [pt.get("node") for pt in tg.findall("path")]

            if path_nodes:
                # Explicit path: prepend the source
                target.path = [source] + path_nodes
            else:
                # No explicit path → ask the routing strategy
                computed = net.routing_strategy.compute_path(
                    net, source, target.to
                )
                target.path = [source] + computed


# ---------- top-level entry point ---------- #

def parse_network(xml_file: str) -> Network:
    """Parse an AFDX XML file and return a fully populated Network."""
    net = Network()

    if not os.path.isfile(xml_file):
        print(f"File not found: {xml_file}")
        return net

    tree = ET.parse(xml_file)
    root = tree.getroot()

    _parse_network_attributes(root, net)
    _parse_stations(root, net)
    _parse_switches(root, net)
    _parse_edges(root, net)
    _parse_flows(root, net)

    return net


################################################################@
#  Phase 7 — Main
################################################################@
################################################################@
#  Main Execution
################################################################@

def file_to_stdout(filename: str):
    """Reads the generated XML file and prints it to standard output."""
    if os.path.isfile(filename):
        with open(filename, "r") as f:
            print(f.read())

if len(sys.argv) >= 2:
    xml_file = sys.argv[1]
else:
    xml_file = "./Samples/ES2E_M.xml"  # Default for testing

# 1. Parse
network = parse_network(xml_file)

# 2. Calculate
# (Do NOT print anything to console here, or Timaeus will break)
network.calculate_loads()
network.check_stability()
network.calculate_delays()

# 3. Export
dot = xml_file.rfind(".")
if dot != -1:
    out_file = xml_file[:dot] + "_res.xml"
else:
    out_file = xml_file + "_res.xml"

network.export_results(out_file)

# 4. Final Handoff
# This is the ONLY thing that should appear in the console output
file_to_stdout(out_file)
