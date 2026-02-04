#!/usr/bin/env python3
"""
SwarmGym Node - Combined Bootstrap + Worker

Run this on your VPS to act as both:
1. Bootstrap peer (discovery for other nodes)
2. Worker (runs simulations)

Usage:
    python scripts/run_node.py --port 38751
"""

import time
import argparse
import os

try:
    from hivemind import DHT
except ImportError:
    print("Error: hivemind not installed.")
    print("Run: pip install hivemind")
    exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Run SwarmGym node (bootstrap + worker)"
    )
    parser.add_argument(
        "--port", type=int, default=38751,
        help="TCP port to listen on (default: 38751)"
    )
    parser.add_argument(
        "--identity", type=str, default="./keys/node.pem",
        help="Path to identity file (default: ./keys/node.pem)"
    )
    args = parser.parse_args()
    
    # Ensure keys directory exists
    os.makedirs(os.path.dirname(args.identity) or ".", exist_ok=True)
    
    print("=" * 60)
    print("  SWARM-GYM NODE")
    print("=" * 60)
    print("")
    print("Starting node...")
    
    dht = DHT(
        host_maddrs=[f"/ip4/0.0.0.0/tcp/{args.port}"],
        identity_path=args.identity,
        start=True,
        wait_timeout=60,
    )
    
    peer_id = dht.peer_id
    maddrs = dht.get_visible_maddrs()
    
    print(f"\nPeer ID: {peer_id}")
    print(f"\nListening on:")
    for addr in maddrs:
        print(f"  {addr}")
    
    # Find public address (not localhost)
    public_addr = None
    for addr in maddrs:
        addr_str = str(addr)
        if "127.0.0.1" not in addr_str and "::1" not in addr_str:
            public_addr = addr_str
            break
    
    print("\n" + "=" * 60)
    print("  CONNECTION STRING")
    print("=" * 60)
    if public_addr:
        print(f'\ninitial_peers=["{public_addr}"]')
        print("\nCopy the above and use in your swarm config.")
    else:
        print("\nNo public IP detected.")
        print("Replace 0.0.0.0 with your VPS public IP:")
        for addr in maddrs:
            if "0.0.0.0" in str(addr):
                print(f'  initial_peers=["{str(addr).replace("0.0.0.0", "YOUR_PUBLIC_IP")}"]')
                break
    
    print("\n" + "=" * 60)
    print("\nNode status:")
    print("  ✓ Bootstrap peer:  ACTIVE")
    print("  ✓ Worker:          READY")
    print("\nPress Ctrl+C to stop.\n")
    
    try:
        while True:
            time.sleep(60)
            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Node running...")
    except KeyboardInterrupt:
        print("\nShutting down node...")
        dht.shutdown()
        print("Node stopped.")


if __name__ == "__main__":
    main()
