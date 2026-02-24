################################################################@
"""
AFDX Network Calculus - Performance Analysis Tool
==================================================

This program implements a performance analysis tool for AFDX (Avionics Full-Duplex
Switched Ethernet) networks using Network Calculus theory.

Given an input XML file describing the network topology (stations, switches, links)
and traffic (flows/Virtual Links with their paths), it computes:
  1. The load (in bps) on every physical link in both directions, and verifies
     the stability condition (ρ < 1, i.e., load < capacity on every link).
  2. The worst-case end-to-end delay bound for each flow to reach each of its
     destinations (multicast targets), using the Network Calculus framework
     (leaky-bucket arrival curves + rate-latency service curves).

The program outputs an XML results file containing delay bounds (in µs) and
link utilisation percentages, compatible with the Open-Timaeus-Net tool for
comparison and validation.

Architecture Overview (Object-Oriented refactoring of __base.py):
  - Node (ABC) ──► Station | Switch        — network elements
  - Edge                                    — full-duplex physical link
  - Flow                                    — AFDX Virtual Link (VL)
  - Target                                  — one destination of a (multicast) flow
  - ArrivalCurve                            — affine curve σ + ρt for Network Calculus
  - Network                                 — central container replacing global lists
  - IRoutingStrategy (ABC) ──► DijkstraStrategy  — Strategy Pattern for path computation

Reference:
  - Appendix A: XML input/output format specification
  - Appendix B: Integration procedure with Open-Timaeus-Net
  - Network Calculus Theorems 1 & 2 (delay bound and output burstiness)
"""
################################################################@

import xml.etree.ElementTree as ET
import os.path
import sys
from abc import ABC, abstractmethod

################################################################@
#  Phase 1 — Core Data Model
#
#  These classes model the physical and logical elements of an
#  AFDX network. They correspond to what the project specification
#  calls "Station", "Switch", "Edge", "Flow", and "Target".
################################################################@

# ─────────────── Node hierarchy ─────────────── #

class Node(ABC):
    """Abstract base class for every network node (station or switch).

    Each node has:
      - name:           unique identifier (e.g. "ES1", "SWA")
      - service_policy: scheduling discipline at the output ports.
                        Typically "FIRST_IN_FIRST_OUT" for AFDX.

    The ABC enforces that subclasses implement `is_switch()` so we
    can dispatch behaviour without isinstance() checks.
    """

    def __init__(self, name: str, service_policy: str = "FIRST_IN_FIRST_OUT"):
        self.name = name
        self.service_policy = service_policy

    @abstractmethod
    def is_switch(self) -> bool:
        ...

    def __repr__(self):
        return f"{self.__class__.__name__}({self.name!r})"


class Station(Node):
    """End-system station (source or destination of flows).

    Stations are the endpoints of AFDX communication:
      - As a SOURCE: they serialize frames onto the output link.
        The serialization delay is L_max / R (maximum frame size / link rate).
      - As a DESTINATION: they are the final hop in a target's path.

    Stations have zero technological latency (per project assumptions).
    """

    def __init__(self, name: str, service_policy: str = "FIRST_IN_FIRST_OUT"):
        super().__init__(name, service_policy)

    def is_switch(self) -> bool:
        return False


class Switch(Node):
    """AFDX switch node with a configurable technological latency.

    Switches perform store-and-forward relaying of Ethernet frames:
      - They receive a complete frame on an input port.
      - Apply the forwarding decision (based on VL routing tables).
      - Queue and transmit the frame on the appropriate output port(s).

    The technological latency (`self.latency`, in seconds) models the
    internal switching fabric delay. It is read from the XML attribute
    `tech-latency` (given in µs, converted to seconds during parsing).

    In Network Calculus terms, a switch output port is modelled as a
    rate-latency server with:
      - Rate R = link capacity (bps)
      - Latency T = technological latency of the switch
    """

    def __init__(self, name: str, latency: float = 0.0,
                 service_policy: str = "FIRST_IN_FIRST_OUT"):
        super().__init__(name, service_policy)
        self.latency = latency          # seconds (converted from µs at parse time)

    def is_switch(self) -> bool:
        return True


# ─────────────── Edge (physical link) ─────────────── #

class Edge:
    """A physical full-duplex link between two nodes.

    In AFDX, every physical cable provides two independent unidirectional
    channels. This class models both directions on a single Edge object:

      - load_direct  : aggregate traffic flowing source → dest  (bps)
      - load_reverse : aggregate traffic flowing dest   → source (bps)

    The `capacity` (in bps, default 100 Mbps) applies identically to
    both directions — each direction can sustain up to `capacity` bps.

    Attributes:
      name      — link identifier from the XML (e.g. "AFDX Edge 1")
      source    — Node object at one end
      dest      — Node object at the other end
      src_port  — port number on the source node
      dest_port — port number on the destination node
      capacity  — maximum throughput per direction (bits per second)
    """

    def __init__(self, name: str, source: Node, dest: Node,
                 src_port: int = 0, dest_port: int = 0,
                 capacity: float = 100_000_000):
        self.name = name
        self.source = source            # Node object at the "from" end
        self.dest = dest                # Node object at the "to" end
        self.src_port = src_port
        self.dest_port = dest_port
        self.capacity = capacity        # bits per second (both directions)

        # Accumulated traffic load in each direction (initialised to zero).
        # These are filled in by Network.calculate_loads().
        self.load_direct = 0.0          # bps,  source → dest
        self.load_reverse = 0.0         # bps,  dest   → source

    def add_load(self, amount: float, direction_source_node: Node) -> None:
        """Accumulate `amount` bps of traffic in the direction originating
        from `direction_source_node`.

        Since the Edge stores two endpoints (source, dest), we compare
        the caller's node reference to determine which counter to increment:
          - If direction_source_node IS self.source → load_direct  (source→dest)
          - Otherwise                               → load_reverse (dest→source)
        """
        if direction_source_node is self.source:
            self.load_direct += amount
        else:
            self.load_reverse += amount

    def __repr__(self):
        return (f"Edge({self.name!r}, {self.source.name}:{self.src_port} "
                f"→ {self.dest.name}:{self.dest_port}, "
                f"C={self.capacity:.0f} bps)")


# ─────────────── Target ─────────────── #

class Target:
    """One destination inside a (possibly multicast) flow.

    A flow (Virtual Link) may be multicast, meaning it is delivered to
    multiple destination stations. Each destination is represented by a
    Target object.

    Attributes:
      flow — back-reference to the parent Flow object
      to   — name of the destination station (e.g. "Dest1")
      path — ordered list of node names from source to destination,
             e.g. ["ES1", "SWA", "SWB", "Dest1"].
             The first element is always the flow's source station.
             Intermediate elements are switches on the route.
             The last element is this target's destination station.
    """

    def __init__(self, flow: "Flow", to: str):
        self.flow = flow
        self.to = to
        self.path: list[str] = []       # ordered node names from source → dest

    def __repr__(self):
        return f"Target({self.to!r}, path={self.path})"


# ─────────────── Flow (Virtual Link) ─────────────── #

class ArrivalCurve:
    """Affine arrival curve  α(t) = σ + ρ·t   (leaky-bucket model).

    In Network Calculus, every flow is characterised by an arrival curve
    that upper-bounds the cumulative number of bits injected into the
    network over any time interval of length t:

        α(t) = σ + ρ·t

    Where:
      σ (sigma) — burst, in bits. Represents the maximum instantaneous
                  amount of data the source can inject (one maximum-size
                  frame for a single AFDX VL).
      ρ (rho)   — long-term average rate, in bits per second.
                  For an AFDX VL: ρ = (payload + overhead) × 8 / period.

    The `+` operator implements arrival curve aggregation:
        (σ_A + σ_B,  ρ_A + ρ_B)
    This is used when multiple flows share the same output port on a
    switch — their individual curves are summed to obtain the aggregate
    arrival curve seen by the shared server (Theorem 1 prerequisite).
    """

    def __init__(self, sigma: float, rho: float):
        self.sigma = sigma      # bits
        self.rho = rho          # bps

    def __add__(self, other: "ArrivalCurve") -> "ArrivalCurve":
        """Aggregate two arrival curves by summing their parameters.

        This is valid because the sum of two leaky-bucket curves is
        itself a leaky-bucket curve (closure property).
        """
        return ArrivalCurve(self.sigma + other.sigma,
                            self.rho + other.rho)

    def __repr__(self):
        return f"ArrivalCurve(σ={self.sigma:.2f} bits, ρ={self.rho:.2f} bps)"


class Flow:
    """An AFDX Virtual Link (VL).

    Each VL is characterised by:
      - source:   the originating end-system station name
      - payload:  maximum payload size (bytes), from XML `max-payload`
      - overhead: protocol overhead (bytes), typically 67 for AFDX
                  (20B IP + 8B UDP + 8B preamble + 12B IFG + 14B Eth + 4B FCS + 1B ...)
      - period:   minimum inter-frame gap (seconds), from XML `period` (ms → s)

    Bandwidth formula (rate consumed on every link along the path):
        BW = (payload + overhead) × 8 / period   [bps]

    Network Calculus initial state:
      - initial_sigma = (payload + overhead) × 8   [bits]  — one max frame
      - initial_rho   = BW                          [bps]  — long-term rate
    These define the leaky-bucket arrival curve at the source station.

    Delay results are stored in `delays_per_target`: a dictionary mapping
    target station name → worst-case end-to-end delay in seconds.
    """

    def __init__(self, name: str, source: str,
                 payload: float, overhead: float, period: float):
        self.name = name
        self.source = source            # source station name (string)
        self.payload = payload          # bytes
        self.overhead = overhead        # bytes (e.g., 67 for AFDX)
        self.period = period            # seconds  (already converted from ms)
        self.targets: list[Target] = [] # ← instance-level list (avoids class-var bug
                                        #    present in the original __base.py where
                                        #    Flow.targets was a class attribute shared
                                        #    across all Flow instances)

        # --- Arrival curve initial state (used in Step 3 delay computation) ---
        # Initial burst: one maximum-sized frame in bits
        self.initial_sigma: float = (payload + overhead) * 8.0   # bits
        # Long-term arrival rate = bandwidth consumed by this VL
        self.initial_rho: float = self.get_bandwidth()           # bps

        # End-to-end delay results: {target_name: delay_in_seconds}
        # Populated by Network.calculate_delays()
        self.delays_per_target: dict[str, float] = {}

    def max_frame_bits(self) -> float:
        """Maximum frame size in bits (payload + overhead).

        This is used in delay computation (Theorem 1) as L_max — the
        largest frame that can arrive at a server, which determines
        the serialization component of the service latency.
        """
        return (self.payload + self.overhead) * 8.0

    def get_bandwidth(self) -> float:
        """Return the bandwidth consumed by this VL in bits per second.

        BW = (payload + overhead) × 8 / period

        This value is:
          - Added to edge loads during load calculation (Step 2)
          - Used as the long-term rate (ρ) of the arrival curve (Step 3)
        """
        return (self.payload + self.overhead) * 8.0 / self.period

    def __repr__(self):
        return (f"Flow({self.name!r}, src={self.source}, "
                f"BW={self.get_bandwidth():.2f} bps)")


################################################################@
#  Phase 3 — Strategy Pattern for Routing (stub)
#
#  When no explicit <path> tags are given in the XML for a target,
#  the program can invoke a routing strategy to compute the path
#  automatically. This uses the Strategy design pattern so that
#  different algorithms (Dijkstra, X-Y routing, etc.) can be
#  swapped in without modifying the rest of the code.

################################################################@

class IRoutingStrategy(ABC):
    """Interface for path-computation strategies.

    Any concrete routing strategy must implement `compute_path()`,
    which returns an ordered list of intermediate + destination node
    names from `source` to `dest` (the source itself is NOT included
    in the returned list — it is prepended by the caller).
    """

    @abstractmethod
    def compute_path(self, network: "Network",
                     source: str, dest: str) -> list[str]:
        """Return an ordered list of node names from *source* to *dest*
        (excluding the source, including the destination)."""
        ...


class DijkstraStrategy(IRoutingStrategy):
    """Stub implementation — prints a notice and returns an empty path.

    A real implementation would build an adjacency graph from
    Network.edges and run Dijkstra's shortest-path algorithm.
    """

    def compute_path(self, network: "Network",
                     source: str, dest: str) -> list[str]:
        print(f"  [DijkstraStrategy] Using Dijkstra to route "
              f"{source} → {dest}")
        return []


################################################################@
#  Phase 2 — Network Container
#
#  The Network class is the central data structure that holds the
#  entire graph (nodes, edges, flows). It replaces the global lists
#  (`nodes`, `edges`, `flows`) from the original __base.py.
#
#  It also contains the core analysis methods:
#    - calculate_loads()  : Step 2 — link utilisation
#    - check_stability()  : Step 2 — verify ρ < 1
#    - calculate_delays() : Step 3 — Network Calculus delay bounds
#    - export_results()   : generate the output XML file
################################################################@

class Network:
    """Central container that owns the full AFDX network graph state.

    Attributes:
      name               — network name from XML <network> tag
      overhead           — global frame overhead in bytes (default 67)
      default_capacity   — default link capacity in bps (default 100 Mbps)
      shortest_path_policy — routing algorithm name (default "DIJKSTRA")
      nodes              — dict mapping node name → Node object
      edges              — list of Edge objects (physical links)
      flows              — list of Flow objects (Virtual Links)
      routing_strategy   — pluggable IRoutingStrategy for automatic path computation
    """

    def __init__(self):
        # --- network-level attributes (populated from XML <network> tag) ---
        self.name: str = ""
        self.overhead: int = 67                          # bytes
        self.default_capacity: float = 100_000_000       # bps (100 Mbps)
        self.shortest_path_policy: str = "DIJKSTRA"

        # --- graph data ---
        self.nodes: dict[str, Node] = {}                 # name → Node
        self.edges: list[Edge] = []
        self.flows: list[Flow] = []

        # --- routing strategy (Strategy Pattern) ---
        self.routing_strategy: IRoutingStrategy = DijkstraStrategy()

    # ──────── helpers ──────── #

    def add_node(self, node: Node) -> None:
        """Register a node in the network's node dictionary."""
        self.nodes[node.name] = node

    def get_node(self, name: str) -> Node:
        """Lookup a node by name. Raises KeyError if not found."""
        return self.nodes[name]

    def get_edge(self, node_a_name: str, node_b_name: str) -> Edge | None:
        """Find the edge connecting two nodes, regardless of which end
        is labelled 'source' vs 'dest' in the Edge object.

        Returns None if no such edge exists. This bidirectional search
        is necessary because the XML may define a link as from=A to=B,
        but a flow might traverse it in the B→A direction.
        """
        for e in self.edges:
            if ((e.source.name == node_a_name and e.dest.name == node_b_name) or
                    (e.source.name == node_b_name and e.dest.name == node_a_name)):
                return e
        return None

    def trace(self) -> None:
        """Print a human-readable summary of the network to the console.

        This is a debugging tool — call it after parsing to verify
        that all stations, switches, edges, and flows were read correctly.
        Note: do NOT print anything to stdout during the actual analysis,
        because Open-Timaeus-Net expects only the XML output on stdout.
        """
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

    # ──────── Step 2: Load Calculation & Stability Verification ──────── #

    def calculate_loads(self) -> None:
        """Walk every flow/target path and accumulate bandwidth on edges.

        For each flow, compute its bandwidth BW = (payload+overhead)*8/period.
        Then, for each target, walk the path node-by-node. For each
        consecutive pair (A → B), find the physical Edge and add BW to
        the appropriate direction (direct or reverse).

        MULTICAST HANDLING:
        In AFDX, a frame is sent ONCE from the source station, and
        switches replicate it at branching points of the multicast tree.
        Therefore, if multiple targets of the SAME flow share a common
        link segment, we must count the bandwidth only ONCE on that link.

        Implementation: we maintain a `visited_links_for_this_flow` set
        keyed by (edge, is_direct) tuples. Before adding load, we check
        if this (edge, direction) was already visited for the current flow.
        """
        for flow in self.flows:
            bw = flow.get_bandwidth()

            # Track which (edge, direction) pairs have been charged for THIS flow.
            # This prevents double-counting bandwidth on shared multicast tree segments.
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

                    # Determine which direction the traffic flows on this edge.
                    # node_a is the sending node for this hop.
                    node_a = self.get_node(node_a_name)

                    # is_direct = True if node_a matches edge.source (source→dest direction)
                    # is_direct = False if node_a matches edge.dest (dest→source = reverse)
                    is_direct = (node_a is edge.source)
                    link_key = (edge, is_direct)

                    # Only add load if we haven't visited this link in this direction
                    # for this flow yet (multicast tree deduplication)
                    if link_key not in visited_links_for_this_flow:
                        edge.add_load(bw, node_a)
                        visited_links_for_this_flow.add(link_key)

    def check_stability(self) -> bool:
        """Verify the stability condition: ρ < 1 on every link direction.

        The stability condition requires that the total load on every
        link (in each direction independently) must be strictly less
        than the link capacity:
            load_direct  < capacity    AND    load_reverse < capacity

        This is equivalent to ρ = load/capacity < 1.

        If ρ ≥ 1 on any link, the delay bounds from Network Calculus
        become infinite (the server cannot drain the queue), and the
        network is considered UNSTABLE.

        Returns:
            True if the network is stable (all links satisfy ρ < 1),
            False otherwise.
        """
        stable = True
        for e in self.edges:
            # Check direct direction (source → dest)
            if e.load_direct >= e.capacity:
                print(f"  [UNSTABLE] {e.name} direct "
                      f"({e.source.name}→{e.dest.name}): "
                      f"load={e.load_direct:.2f} bps >= "
                      f"capacity={e.capacity:.0f} bps  "
                      f"(ρ={e.load_direct / e.capacity:.4f})")
                stable = False
            # Check reverse direction (dest → source)
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

    # ──────── Step 3: End-to-End Delay Analysis (Network Calculus) ──────── #

    def calculate_delays(self) -> None:
        """Compute worst-case end-to-end path delay bounds using Network Calculus.

        This implements the iterative hop-by-hop procedure using:
          - Theorem 1 (Delay Bound): For a flow with arrival curve α(t) = σ + ρt
            passing through a rate-latency server β(t) = R·[t - T]+, the
            maximum delay is bounded by:
                D ≤ σ_agg / R + T
            where σ_agg is the aggregate burst of ALL flows sharing that server.

          - Theorem 2 (Output Arrival Curve): The arrival curve of a flow at the
            OUTPUT of a server is an affine curve with updated parameters:
                σ_out = σ_in + ρ_in × D
                ρ_out = ρ_in   (the long-term rate is preserved)
            This is called the "output burstiness theorem" — the burst grows
            as the flow experiences delay, but the rate stays the same.

        Algorithm — Hop-by-hop feed-forward iteration:

        For each hop index h = 0, 1, ..., max_hops-1:
          1. GROUP: Identify every (edge, direction) that serves as the h-th
             link for at least one (flow, target) pair.
          2. For each such (edge, direction) group:
             a. AGGREGATE arrival curves of all *distinct flows* on this edge
                at this hop (multicast dedup — a flow counts once even if
                multiple targets share the edge at the same hop).
             b. Determine the service curve parameters:
                  R = edge capacity (bps)
                  T = { L_max / R              if upstream node is a Station
                      { switch.latency          if upstream node is a Switch
                Note: For a Station (source), T = L_max/R represents the
                serialization delay of the largest frame onto the wire.
                For a Switch, T is the technological latency read from XML.
             c. Compute delay bound:  D = σ_agg / R + T
             d. Update every (flow, target) state traversing this edge:
                  - accumulated_delay += D
                  - σ_new = σ_old + ρ_old × D   (Theorem 2: output burstiness)
                  - ρ stays unchanged

        After all hops are processed, the accumulated delay for each
        (flow, target) pair is the worst-case end-to-end delay bound.

        Note on Open-Timaeus compatibility:
            The delay values are designed to match the "Blocks wo line shaping"
            engine of Open-Timaeus-Net, which uses the same serialization
            policy described above.
        """
        # Configuration flags (kept for clarity and potential future extension)
        SERIALIZATION_AT_SOURCE_ONLY = True  # Serialization (L_max/R) only at source stations
        USE_FULL_FRAME_SIZE = True           # Use (payload+overhead)*8 as L_max, not just payload*8

        # ─── Build per-(flow, target) state ───
        # Each (flow, target) pair tracks:
        #   - "curve": current arrival curve (starts as the flow's initial curve)
        #   - "delay": accumulated end-to-end delay so far (starts at 0)
        states: dict[tuple[Flow, Target], dict] = {}
        for flow in self.flows:
            for target in flow.targets:
                states[(flow, target)] = {
                    "curve": ArrivalCurve(flow.initial_sigma, flow.initial_rho),
                    "delay": 0.0,
                }

        # ─── Determine max hops across all targets ───
        # This tells us how many iterations of the hop-by-hop loop we need.
        max_hops = 0
        for flow in self.flows:
            for target in flow.targets:
                # Number of edges traversed = len(path) - 1
                hops = len(target.path) - 1
                if hops > max_hops:
                    max_hops = hops

        # ─── Iterate hop by hop ───
        for hop in range(max_hops):

            # Step 1: Collect all (edge, direction) groups active at this hop.
            #
            # For every (flow, target) that has a hop-th edge, determine
            # which physical edge and direction it uses. Group them so we
            # can aggregate arrival curves of flows sharing the same
            # output port at the same hop.
            #
            # Key:   (edge, is_direct)  — identifies a specific output port direction
            # Value: list of (flow, target, upstream_node_name) tuples
            edge_groups: dict[tuple[Edge, bool], list[tuple[Flow, Target, str]]] = {}

            for flow in self.flows:
                for target in flow.targets:
                    path = target.path
                    if hop >= len(path) - 1:
                        continue  # this target's path has fewer hops; skip it

                    node_a_name = path[hop]       # upstream node at this hop
                    node_b_name = path[hop + 1]   # downstream node at this hop

                    edge = self.get_edge(node_a_name, node_b_name)
                    if edge is None:
                        continue

                    node_a = self.get_node(node_a_name)
                    is_direct = (node_a is edge.source)
                    key = (edge, is_direct)

                    if key not in edge_groups:
                        edge_groups[key] = []
                    edge_groups[key].append((flow, target, node_a_name))

            # Step 2: Process each (edge, direction) group
            for (edge, is_direct), ft_list in edge_groups.items():

                R = edge.capacity  # bps — the server rate

                # ─── Aggregate arrival curves (dedup by flow for multicast) ───
                # If multiple targets of the same flow pass through this edge
                # at this hop, the flow's arrival curve is counted only ONCE
                # (the switch replicates the frame, it doesn't receive it twice).
                seen_flows: set[Flow] = set()
                agg_curve = ArrivalCurve(0.0, 0.0)  # starts empty
                l_max = 0.0  # track max frame size among all flows on this edge

                for flow, target, upstream_name in ft_list:
                    if flow not in seen_flows:
                        seen_flows.add(flow)
                        state = states[(flow, target)]
                        # Sum the arrival curve of this flow into the aggregate
                        agg_curve = agg_curve + state["curve"]
                        # Track maximum frame size for serialization delay
                        if USE_FULL_FRAME_SIZE:
                            frame_bits = flow.max_frame_bits()
                        else:
                            frame_bits = flow.payload * 8.0
                        if frame_bits > l_max:
                            l_max = frame_bits

                # ─── Determine service curve latency T ───
                # The upstream node determines the latency component:
                upstream_name = ft_list[0][2]
                upstream_node = self.get_node(upstream_name)

                if upstream_node.is_switch():
                    # Switch: T = technological latency of the switch.
                    # The serialization delay is already captured in σ/R.
                    T = upstream_node.latency
                else:
                    # Station (source end-system): T = L_max / R.
                    # This is the time to serialize the largest frame onto the wire.
                    # It models the first hop from the source station.
                    T = l_max / R

                # ─── Compute delay bound (Theorem 1) ───
                # D = σ_agg / R + T
                # where σ_agg is the aggregate burst of all competing flows,
                # R is the server rate, and T is the service curve latency.
                D = agg_curve.sigma / R + T

                # ─── Update every (flow, target) state on this edge ───
                for flow, target, _ in ft_list:
                    s = states[(flow, target)]
                    # Accumulate delay along the path
                    s["delay"] += D
                    # Apply Theorem 2: output burstiness theorem
                    # σ_new = σ_old + ρ_old × D  (burst increases by rate × delay)
                    # ρ_new = ρ_old              (long-term rate is preserved)
                    s["curve"] = ArrivalCurve(
                        s["curve"].sigma + s["curve"].rho * D,
                        s["curve"].rho,  # rho stays the same
                    )

        # ─── Write results back to Flow objects ───
        # Store the computed end-to-end delay for each (flow, target) pair
        # in the flow's delays_per_target dictionary (in seconds).
        for (flow, target), state in states.items():
            flow.delays_per_target[target.to] = max(
                0.0, state["delay"] 
            )

    # ──────── Step 2+3: XML Output Generation ──────── #

    def export_results(self, filename: str) -> None:
        """Generate an XML results file compatible with Open-Timaeus-Net.

        The output format (defined in Appendix A) contains two sections:
          1. <delays>: end-to-end delay bounds for each flow/target, in µs.
          2. <load>:   link utilisation for each edge, in both directions,
                       showing absolute load (bps) and percentage of capacity.

        Output structure:
            <?xml version="1.0" encoding="UTF-8"?>
            <results>
                <delays>
                    <flow name="…">
                        <target name="…" value="[delay in µs]" />
                    </flow>
                </delays>
                <load>
                    <edge name="…">
                        <usage type="direct"  value="[bps]" percent="[%]" />
                        <usage type="reverse" value="[bps]" percent="[%]" />
                    </edge>
                </load>
            </results>

        The delay values are converted from seconds to microseconds (×10⁶)
        and formatted to 2 decimal places to match Open-Timaeus conventions.
        """
        with open(filename, "w", encoding="utf-8") as f:
            f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
            f.write('<results>\n')

            # --- Section 1: DELAYS (from Step 3 Network Calculus) ---
            f.write('\t<delays>\n')
            for flow in self.flows:
                if flow.delays_per_target:
                    f.write(f'\t\t<flow name="{flow.name}">\n')
                    for target in flow.targets:
                        # Convert seconds → microseconds for the output
                        delay_us = flow.delays_per_target.get(target.to, 0.0) * 1e6
                        f.write(f'\t\t\t<target name="{target.to}" '
                                f'value="{delay_us:.2f}" />\n')
                    f.write(f'\t\t</flow>\n')
            f.write('\t</delays>\n')

            # --- Section 2: LOAD (from Step 2 load calculation) ---
            f.write('\t<load>\n')

            for e in self.edges:
                # Calculate utilisation percentage (guard against division by zero)
                if e.capacity > 0:
                    pct_direct = (e.load_direct / e.capacity) * 100.0
                    pct_reverse = (e.load_reverse / e.capacity) * 100.0
                else:
                    pct_direct = 0.0
                    pct_reverse = 0.0

                f.write(f'\t\t<edge name="{e.name}">\n')
                # Direct direction: source → dest
                f.write(f'\t\t\t<usage type="direct" '
                        f'value="{e.load_direct:.0f}" '
                        f'percent="{pct_direct:.2f}%" />\n')
                # Reverse direction: dest → source
                f.write(f'\t\t\t<usage type="reverse" '
                        f'value="{e.load_reverse:.0f}" '
                        f'percent="{pct_reverse:.2f}%" />\n')
                f.write(f'\t\t</edge>\n')

            f.write('\t</load>\n')
            f.write('</results>\n')

################################################################@
#  Phase 4 — XML Parsing
#
#  These functions parse the input XML file (format defined in
#  Appendix A) and populate a Network object with all nodes,
#  edges, and flows.
#
#  The XML structure expected is:
#    <wpanets>
#      <network name="..." overhead="67" ... />
#      <station name="ES1" ... />
#      <switch name="SW" tech-latency="60" ... />
#      <link name="Edge1" from="ES1" to="SW" ... />
#      <flow name="F1" source="ES1" max-payload="1000" period="1" ...>
#        <target name="ES2">
#          <path node="SW" />
#          <path node="ES2" />
#        </target>
#      </flow>
#    </wpanets>
################################################################@

def _parse_capacity(raw: str | None, default: float = 100_000_000) -> float:
    """Convert a transmission-capacity string from XML to bits-per-second.

    The XML may express capacity in several formats:
      • Pure numeric string:  "100000000"
      • With Gbps suffix:     "1Gbps"      → 1 × 10⁹ bps
      • With Mbps suffix:     "100Mbps"    → 100 × 10⁶ bps
      • With kbps suffix:     "500kbps"    → 500 × 10³ bps

    Falls back to `default` (100 Mbps) when the attribute is missing or
    cannot be parsed.
    """
    if raw is None:
        return default
    raw = raw.strip()
    # Try pure number first
    try:
        return float(raw)
    except ValueError:
        pass
    # Handle common suffixed forms (case-insensitive)
    lower = raw.lower()
    if lower.endswith("gbps"):
        return float(raw[:-4]) * 1e9
    if lower.endswith("mbps"):
        return float(raw[:-4]) * 1e6
    if lower.endswith("kbps"):
        return float(raw[:-4]) * 1e3
    # Last resort — return the default
    return default


def _parse_network_attributes(root: ET.Element, net: Network) -> None:
    """Read the <network> tag and populate the Network's global attributes.

    Extracts:
      - name:                   network name
      - overhead:               global frame overhead in bytes (default 67)
      - shortest-path-policy:   routing algorithm hint (default "DIJKSTRA")
      - transmission-capacity:  default link capacity (default 100 Mbps)
    """
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
    """Parse every <station> tag and add Station nodes to the network.

    Each station has a name and an optional service-policy attribute
    (defaults to FIRST_IN_FIRST_OUT).
    """
    for el in root.findall("station"):
        name = el.get("name")
        policy = el.get("service-policy", "FIRST_IN_FIRST_OUT")
        net.add_node(Station(name, service_policy=policy))


def _parse_switches(root: ET.Element, net: Network) -> None:
    """Parse every <switch> tag and add Switch nodes to the network.

    Each switch has:
      - name:         switch identifier
      - tech-latency: internal switching latency in µs (converted to seconds)
      - service-policy: scheduling discipline (default FIFO)
    """
    for el in root.findall("switch"):
        name = el.get("name")
        # Convert tech-latency from microseconds to seconds
        latency = float(el.get("tech-latency", "0")) * 1e-6   # µs → s
        policy = el.get("service-policy", "FIRST_IN_FIRST_OUT")
        net.add_node(Switch(name, latency=latency, service_policy=policy))


def _parse_edges(root: ET.Element, net: Network) -> None:
    """Parse every <link> tag and create Edge objects with Node references.

    Each link connects two nodes (identified by 'from' and 'to' attributes)
    on specific port numbers ('fromPort', 'toPort'). The capacity can be
    specified per-link or inherited from the network default.
    """
    for el in root.findall("link"):
        name = el.get("name")
        # Look up the actual Node objects (not just names) so that
        # identity comparisons (node_a is edge.source) work correctly.
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
    """Parse every <flow> tag, including multicast targets and explicit paths.

    For each flow, extracts:
      - name, source station, max-payload (bytes), period (ms → seconds)
      - Overhead is taken from the network-level setting

    For each <target> sub-element within a flow:
      - Creates a Target object
      - If explicit <path> tags are present, builds the path as:
            [source, intermediate_nodes..., destination]
      - If no <path> tags exist, invokes the routing strategy to compute
        the path automatically.

    Path convention (inherited from __base.py):
        target.path = [source, node1, …, destination]
        The source is always the first element.
    """
    for el in root.findall("flow"):
        name = el.get("name")
        source = el.get("source")
        payload = float(el.get("max-payload", "0"))
        # Period is given in milliseconds in the XML; convert to seconds
        period = float(el.get("period", "1")) * 1e-3   # ms → s

        flow = Flow(name, source,
                    payload=payload,
                    overhead=net.overhead,
                    period=period)
        net.flows.append(flow)

        for tg in el.findall("target"):
            target = Target(flow, tg.get("name"))
            flow.targets.append(target)

            # Collect explicit <path> node names (if provided in the XML)
            path_nodes = [pt.get("node") for pt in tg.findall("path")]

            if path_nodes:
                # Explicit path given: prepend the source station
                target.path = [source] + path_nodes
            else:
                # No explicit path → ask the routing strategy to compute one
                computed = net.routing_strategy.compute_path(
                    net, source, target.to
                )
                target.path = [source] + computed


# ─────────────── Top-level entry point ─────────────── #

def parse_network(xml_file: str) -> Network:
    """Parse an AFDX XML file and return a fully populated Network.

    This is the main entry point for XML parsing. It:
      1. Creates an empty Network object
      2. Validates that the file exists
      3. Parses the XML tree
      4. Populates network attributes, stations, switches, edges, and flows
      5. Returns the complete Network ready for analysis
    """
    net = Network()

    if not os.path.isfile(xml_file):
        print(f"File not found: {xml_file}")
        return net

    tree = ET.parse(xml_file)
    root = tree.getroot()

    # Parse in dependency order: attributes → nodes → edges → flows
    # (Flows reference nodes, edges reference nodes, etc.)
    _parse_network_attributes(root, net)
    _parse_stations(root, net)
    _parse_switches(root, net)
    _parse_edges(root, net)
    _parse_flows(root, net)

    return net


################################################################@
#  Main Execution
#
#  The program follows this pipeline:
#    1. PARSE:     Read the XML input file into a Network object
#    2. ANALYSE:   Compute link loads, verify stability, compute delays
#    3. EXPORT:    Write results to an XML file (<input>_res.xml)
#    4. OUTPUT:    Print the XML results to stdout for Open-Timaeus-Net
#
#  Usage:
#    python tomas_coelho.py <path_to_input.xml>
#
#  If no argument is given, defaults to ./Samples/ES2E_M.xml for testing.
################################################################@

def file_to_stdout(filename: str):
    """Read the generated XML results file and print it to standard output.

    Open-Timaeus-Net captures stdout to read the analysis results,
    so this function is the final handoff step. Nothing else should
    be printed to stdout during normal execution.
    """
    if os.path.isfile(filename):
        with open(filename, "r") as f:
            print(f.read())

# ─── Command-line argument handling ───
if len(sys.argv) >= 2:
    xml_file = sys.argv[1]
else:
    xml_file = "./Samples/ES2E_M.xml"  # Default test file

# ─── Step 1: Parse the input XML ───
network = parse_network(xml_file)

# ─── Step 2: Calculate link loads and verify stability ───
# (Do NOT print anything extra to console here, or Timaeus will break)
network.calculate_loads()
network.check_stability()

# ─── Step 3: Compute worst-case end-to-end delay bounds ───
network.calculate_delays()

# ─── Generate output file name: input.xml → input_res.xml ───
dot = xml_file.rfind(".")
if dot != -1:
    out_file = xml_file[:dot] + "_res.xml"
else:
    out_file = xml_file + "_res.xml"

# ─── Export results to XML ───
network.export_results(out_file)

# ─── Final Handoff: print XML to stdout for Open-Timaeus-Net ───
# This is the ONLY thing that should appear in the console output
file_to_stdout(out_file)
