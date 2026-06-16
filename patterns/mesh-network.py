"""
Offline-First AI Mesh Network Pattern
Extracted from HearthNet (Build Small Hackathon 2026).

Pattern: Nodes discover each other via mDNS/UDP broadcast. Each node advertises
AI capabilities. Requests route to the most capable node. E2E encryption with
shared passphrase. Works with zero internet.
"""

from __future__ import annotations

import hashlib
import json
import socket
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable


# ---------------------------------------------------------------------------
# 1. CAPABILITY DECLARATION — What each node can do
# ---------------------------------------------------------------------------

@dataclass
class NodeCapability:
    """One thing a node can do (e.g. 'text-generation', 'embedding', 'search')."""
    name: str                   # "text-generation", "embedding", "tts"
    model: str = ""             # "llama-3-8b", "all-MiniLM-L6-v2", etc.
    device: str = "cpu"         # "cpu", "cuda", "mps"
    max_tokens: int = 2048
    score: float = 1.0          # Higher = more capable. GPU nodes score higher.


@dataclass
class NodeInfo:
    """Identity and capabilities of one node in the mesh."""
    node_id: str                # Unique ID (hostname or generated)
    host: str                   # IP address
    port: int                   # HTTP port
    capabilities: list[NodeCapability] = field(default_factory=list)
    last_seen: float = 0.0      # Timestamp of last heartbeat

    def can_handle(self, task: str) -> NodeCapability | None:
        """Check if this node can handle a given task type."""
        for cap in self.capabilities:
            if cap.name == task:
                return cap
        return None

    def to_dict(self) -> dict:
        return {
            "node_id": self.node_id,
            "host": self.host,
            "port": self.port,
            "capabilities": [
                {"name": c.name, "model": c.model, "device": c.device,
                 "score": c.score}
                for c in self.capabilities
            ],
        }


# ---------------------------------------------------------------------------
# 2. MESH REGISTRY — Track all known nodes
# ---------------------------------------------------------------------------

class MeshRegistry:
    """Thread-safe registry of nodes discovered on the network."""

    def __init__(self, stale_timeout: float = 30.0):
        self._nodes: dict[str, NodeInfo] = {}
        self._lock = threading.Lock()
        self._stale_timeout = stale_timeout

    def register(self, node: NodeInfo) -> None:
        """Add or update a node."""
        with self._lock:
            node.last_seen = time.time()
            self._nodes[node.node_id] = node

    def remove(self, node_id: str) -> None:
        with self._lock:
            self._nodes.pop(node_id, None)

    def get_active_nodes(self) -> list[NodeInfo]:
        """Return all nodes seen within the stale timeout."""
        cutoff = time.time() - self._stale_timeout
        with self._lock:
            return [n for n in self._nodes.values() if n.last_seen > cutoff]

    def find_best_node(self, task: str) -> NodeInfo | None:
        """
        Find the best node for a given task, scored by capability.
        This is the routing decision — highest score wins.
        """
        best = None
        best_score = -1.0
        for node in self.get_active_nodes():
            cap = node.can_handle(task)
            if cap and cap.score > best_score:
                best = node
                best_score = cap.score
        return best


# ---------------------------------------------------------------------------
# 3. ENCRYPTION — Fernet symmetric encryption with shared passphrase
# ---------------------------------------------------------------------------

def derive_key(passphrase: str) -> bytes:
    """Derive a Fernet key from a shared passphrase. SHA-256 -> base64."""
    import base64
    return base64.urlsafe_b64encode(hashlib.sha256(passphrase.encode()).digest())

def encrypt_message(data: dict, key: bytes) -> bytes:
    """Encrypt a dict as a Fernet token."""
    from cryptography.fernet import Fernet
    return Fernet(key).encrypt(json.dumps(data).encode())

def decrypt_message(token: bytes, key: bytes) -> dict:
    """Decrypt a Fernet token back to a dict."""
    from cryptography.fernet import Fernet
    return json.loads(Fernet(key).decrypt(token))


# ---------------------------------------------------------------------------
# 4. UDP DISCOVERY — Broadcast presence on the LAN (use zeroconf in prod)
# ---------------------------------------------------------------------------

BROADCAST_PORT = 51900
MAGIC = b"MESH:"

class UDPAnnouncer:
    """Broadcasts this node's presence on the LAN every N seconds."""
    def __init__(self, node_info: NodeInfo, interval: float = 5.0):
        self.node_info, self.interval, self._running = node_info, interval, False

    def start(self) -> threading.Thread:
        self._running = True
        t = threading.Thread(target=self._loop, daemon=True); t.start(); return t

    def _loop(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        while self._running:
            try:
                sock.sendto(MAGIC + json.dumps(self.node_info.to_dict()).encode(),
                            ("<broadcast>", BROADCAST_PORT))
            except OSError: pass
            time.sleep(self.interval)
        sock.close()

    def stop(self): self._running = False

class UDPListener:
    """Listens for node announcements and registers them."""
    def __init__(self, registry: MeshRegistry):
        self.registry, self._running = registry, False

    def start(self) -> threading.Thread:
        self._running = True
        t = threading.Thread(target=self._loop, daemon=True); t.start(); return t

    def _loop(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(("", BROADCAST_PORT)); sock.settimeout(2.0)
        while self._running:
            try:
                data, _ = sock.recvfrom(4096)
                if data.startswith(MAGIC):
                    p = json.loads(data[len(MAGIC):])
                    self.registry.register(NodeInfo(
                        node_id=p["node_id"], host=p["host"], port=p["port"],
                        capabilities=[NodeCapability(**c) for c in p.get("capabilities", [])]))
            except (socket.timeout, json.JSONDecodeError, KeyError): continue
        sock.close()

    def stop(self): self._running = False


# ---------------------------------------------------------------------------
# EXAMPLE USAGE
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Define this node
    my_node = NodeInfo(
        node_id="laptop-01",
        host="192.168.1.42",
        port=8080,
        capabilities=[
            NodeCapability("text-generation", model="llama-3-8b",
                           device="cuda", score=8.0),
            NodeCapability("embedding", model="all-MiniLM-L6-v2",
                           device="cpu", score=3.0),
        ],
    )

    # Create the registry
    registry = MeshRegistry()
    registry.register(my_node)

    # Simulate another node joining
    pi_node = NodeInfo(
        node_id="pi-kitchen",
        host="192.168.1.100",
        port=8080,
        capabilities=[
            NodeCapability("tts", model="piper", device="cpu", score=2.0),
        ],
    )
    registry.register(pi_node)

    # Route a request to the best node
    best = registry.find_best_node("text-generation")
    if best:
        print(f"Route text-generation to: {best.node_id} ({best.host})")

    best_tts = registry.find_best_node("tts")
    if best_tts:
        print(f"Route tts to: {best_tts.node_id} ({best_tts.host})")

    missing = registry.find_best_node("image-generation")
    print(f"Route image-generation to: {missing}")  # None — no node can do it

    # List all active nodes
    print(f"\nActive nodes: {[n.node_id for n in registry.get_active_nodes()]}")
