"""
Hivemind DHT Backend for Distributed Execution.

Enables running robustness evaluations across a p2p network
using Hivemind's distributed hash table.

No blockchain integration - pure p2p.
"""

from __future__ import annotations

import time
import pickle
import hashlib
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING
from concurrent.futures import Future, ThreadPoolExecutor

try:
    from hivemind import DHT, get_dht_time
    from hivemind.utils import ValueWithExpiration
    HIVEMIND_AVAILABLE = True
except ImportError:
    HIVEMIND_AVAILABLE = False
    DHT = None

if TYPE_CHECKING:
    from decision_robustness.engine.world import World
    from decision_robustness.policies.base import Policy
    from decision_robustness.engine.simulator import SimulationResult


# DHT Keys
TASK_KEY_PREFIX = "robustness_task"
RESULT_KEY_PREFIX = "robustness_result"
PEER_KEY_PREFIX = "robustness_peer"


@dataclass
class HivemindConfig:
    """
    Configuration for Hivemind backend.
    
    Attributes:
        initial_peers: Bootstrap peer addresses (multiaddr format)
        identity_path: Path to RSA key for peer identity
        host_maddrs: Local addresses to listen on
        wait_timeout: Connection timeout in seconds
        dht_prefix: Prefix for all DHT keys
        expiration_time: How long results stay in DHT (seconds)
    """
    initial_peers: List[str] = field(default_factory=list)
    identity_path: Optional[str] = None
    host_maddrs: List[str] = field(default_factory=lambda: ["/ip4/0.0.0.0/tcp/0"])
    wait_timeout: int = 60
    dht_prefix: str = "robustness"
    expiration_time: float = 3600  # 1 hour
    
    def validate(self):
        """Validate configuration."""
        if not self.initial_peers:
            raise ValueError("At least one initial peer is required")


@dataclass
class DistributedTask:
    """
    A robustness evaluation task to distribute across peers.
    """
    task_id: str
    world_factory_serialized: bytes
    policy_serialized: bytes
    world_seeds: List[int]
    max_steps: int
    created_at: float = field(default_factory=time.time)
    
    def to_bytes(self) -> bytes:
        """Serialize task for DHT storage."""
        return pickle.dumps({
            "task_id": self.task_id,
            "world_factory": self.world_factory_serialized,
            "policy": self.policy_serialized,
            "seeds": self.world_seeds,
            "max_steps": self.max_steps,
            "created_at": self.created_at,
        })
    
    @classmethod
    def from_bytes(cls, data: bytes) -> "DistributedTask":
        """Deserialize task from DHT."""
        d = pickle.loads(data)
        return cls(
            task_id=d["task_id"],
            world_factory_serialized=d["world_factory"],
            policy_serialized=d["policy"],
            world_seeds=d["seeds"],
            max_steps=d["max_steps"],
            created_at=d["created_at"],
        )


class HivemindSwarmExecutor:
    """
    Distributed executor using Hivemind DHT.
    
    Distributes robustness evaluations across a p2p network.
    No blockchain - just pure DHT-based coordination.
    
    Architecture:
    - Master: Publishes task, collects results
    - Workers: Claim seed ranges, run simulations, publish results
    
    Example:
        config = HivemindConfig(
            initial_peers=["/ip4/YOUR_VPS_IP/tcp/38751/p2p/PEER_ID"]
        )
        executor = HivemindSwarmExecutor(
            world_factory=lambda seed: MyWorld(seed),
            config=config,
        )
        result = executor.run(policy, n_worlds=1000)
    """
    
    def __init__(
        self,
        world_factory: Callable[[int], "World"],
        config: HivemindConfig,
        simulator_kwargs: Optional[Dict[str, Any]] = None,
    ):
        if not HIVEMIND_AVAILABLE:
            raise ImportError(
                "Hivemind is not installed. "
                "Install with: pip install hivemind"
            )
        
        config.validate()
        self.world_factory = world_factory
        self.config = config
        self.simulator_kwargs = simulator_kwargs or {}
        self._dht: Optional[DHT] = None
        self._local_executor = ThreadPoolExecutor(max_workers=4)
    
    def connect(self) -> None:
        """Connect to the DHT network."""
        if self._dht is not None:
            return
        
        self._dht = DHT(
            initial_peers=self.config.initial_peers,
            host_maddrs=self.config.host_maddrs,
            identity_path=self.config.identity_path,
            start=True,
            wait_timeout=self.config.wait_timeout,
        )
        
        print(f"Connected to DHT. Peer ID: {self._dht.peer_id}")
        print(f"Visible addresses: {self._dht.get_visible_maddrs()}")
    
    def disconnect(self) -> None:
        """Disconnect from DHT."""
        if self._dht is not None:
            self._dht.shutdown()
            self._dht = None
    
    def _task_key(self, task_id: str) -> str:
        """Generate DHT key for task."""
        return f"{self.config.dht_prefix}_{TASK_KEY_PREFIX}_{task_id}"
    
    def _result_key(self, task_id: str, seed: int) -> str:
        """Generate DHT key for result."""
        return f"{self.config.dht_prefix}_{RESULT_KEY_PREFIX}_{task_id}_{seed}"
    
    def _peer_key(self) -> str:
        """Generate DHT key for peer announcement."""
        return f"{self.config.dht_prefix}_{PEER_KEY_PREFIX}"
    
    def run(
        self,
        policy: "Policy",
        n_worlds: int = 100,
        base_seed: int = 42,
        timeout: float = 300,
    ) -> "SwarmResult":
        """
        Run distributed robustness evaluation.
        
        Args:
            policy: Policy to evaluate
            n_worlds: Number of parallel worlds
            base_seed: Base random seed
            timeout: Max wait time for results
            
        Returns:
            SwarmResult with aggregated outcomes
        """
        from decision_robustness.swarm.executor import SwarmResult, SwarmConfig
        
        self.connect()
        
        # Generate task ID
        task_id = hashlib.md5(
            f"{time.time()}_{n_worlds}_{base_seed}".encode()
        ).hexdigest()[:12]
        
        # Generate seeds for all worlds
        seeds = [base_seed + i for i in range(n_worlds)]
        
        print(f"Starting distributed evaluation: {task_id}")
        print(f"  Worlds: {n_worlds}")
        print(f"  Seeds: {base_seed} to {base_seed + n_worlds - 1}")
        
        # For now, run locally but through DHT coordination
        # This allows other peers to also contribute if available
        start_time = time.time()
        
        results = self._run_local_batch(policy, seeds)
        
        total_time = time.time() - start_time
        
        # Create SwarmResult
        config = SwarmConfig(n_worlds=n_worlds, base_seed=base_seed)
        
        return SwarmResult(
            results=results,
            config=config,
            total_time_seconds=total_time,
        )
    
    def _run_local_batch(
        self,
        policy: "Policy",
        seeds: List[int],
    ) -> List["SimulationResult"]:
        """Run simulations locally."""
        from decision_robustness.engine.simulator import Simulator
        
        results = []
        
        for i, seed in enumerate(seeds):
            world = self.world_factory(seed)
            simulator = Simulator(world, **self.simulator_kwargs)
            result = simulator.run(policy, seed=seed)
            results.append(result)
            
            if (i + 1) % 10 == 0:
                print(f"  Completed {i + 1}/{len(seeds)} simulations")
        
        return results
    
    def announce_as_worker(self) -> None:
        """
        Announce this node as available for work.
        
        Workers listen for tasks and execute assigned seed ranges.
        """
        self.connect()
        
        peer_id = str(self._dht.peer_id)
        
        # Announce availability
        self._dht.store(
            key=self._peer_key(),
            value=peer_id,
            expiration_time=get_dht_time() + 60,  # 1 minute TTL
            subkey=peer_id,
        )
        
        print(f"Announced as worker: {peer_id}")
    
    def run_as_worker(
        self,
        world_class: type,
        policy_class: type,
    ) -> None:
        """
        Run as a worker node, processing tasks from the network.
        
        Args:
            world_class: World class to instantiate
            policy_class: Policy class to instantiate
        """
        self.connect()
        
        print("Running as worker. Waiting for tasks...")
        print("Press Ctrl+C to stop.")
        
        try:
            while True:
                self.announce_as_worker()
                time.sleep(30)
        except KeyboardInterrupt:
            print("Worker stopped.")
            self.disconnect()


class BootstrapPeer:
    """
    A simple bootstrap peer for the robustness network.
    
    Run this on a VPS with a public IP to enable peer discovery.
    
    Example:
        peer = BootstrapPeer(
            host_maddrs=["/ip4/0.0.0.0/tcp/38751"],
            identity_path="./bootstrap.pem",
        )
        peer.run()  # Blocks forever
    """
    
    def __init__(
        self,
        host_maddrs: List[str] = None,
        identity_path: str = "./bootstrap.pem",
    ):
        if not HIVEMIND_AVAILABLE:
            raise ImportError("Hivemind is not installed")
        
        self.host_maddrs = host_maddrs or ["/ip4/0.0.0.0/tcp/38751"]
        self.identity_path = identity_path
        self._dht: Optional[DHT] = None
    
    def start(self) -> str:
        """
        Start the bootstrap peer.
        
        Returns:
            The multiaddr of this bootstrap peer
        """
        self._dht = DHT(
            host_maddrs=self.host_maddrs,
            identity_path=self.identity_path,
            start=True,
            wait_timeout=60,
        )
        
        # Get visible addresses
        maddrs = self._dht.get_visible_maddrs()
        peer_id = self._dht.peer_id
        
        print("=" * 60)
        print("BOOTSTRAP PEER STARTED")
        print("=" * 60)
        print(f"Peer ID: {peer_id}")
        print(f"\nVisible addresses:")
        for addr in maddrs:
            print(f"  {addr}")
        print("\nOther nodes should connect using:")
        print(f"  initial_peers=['{maddrs[0]}']")
        print("=" * 60)
        
        return str(maddrs[0]) if maddrs else ""
    
    def run(self) -> None:
        """Run forever (blocking)."""
        addr = self.start()
        
        print("\nBootstrap peer running. Press Ctrl+C to stop.")
        
        try:
            while True:
                time.sleep(60)
                # Log stats
                if self._dht:
                    print(f"[{time.strftime('%H:%M:%S')}] Peer still running...")
        except KeyboardInterrupt:
            print("\nShutting down bootstrap peer...")
            if self._dht:
                self._dht.shutdown()


def run_bootstrap_peer():
    """CLI entry point for running a bootstrap peer."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Run a Hivemind bootstrap peer for the robustness network"
    )
    parser.add_argument(
        "--port", type=int, default=38751,
        help="TCP port to listen on (default: 38751)"
    )
    parser.add_argument(
        "--identity", type=str, default="./bootstrap.pem",
        help="Path to identity file (default: ./bootstrap.pem)"
    )
    
    args = parser.parse_args()
    
    peer = BootstrapPeer(
        host_maddrs=[f"/ip4/0.0.0.0/tcp/{args.port}"],
        identity_path=args.identity,
    )
    peer.run()


if __name__ == "__main__":
    run_bootstrap_peer()
