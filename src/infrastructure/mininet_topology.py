"""
Mininet/Containernet Topology for IIoMT Agentic Security Framework.

This topology replicates the physical testbed described in the paper:
- 1 Central Gateway Switch
- N Edge Nodes (Docker containers running System 1 FP32 Model)
- 1 Gateway Node (Docker container running System 2 SLM reasoning)
- Replay Nodes attached to inject traffic via tcpreplay

Note: This requires a Linux environment with Containernet installed.
"""
import argparse
from mininet.net import Containernet
from mininet.node import Controller
from mininet.cli import CLI
from mininet.link import TCLink
from mininet.log import info, setLogLevel

def create_topology(num_edges=3):
    setLogLevel('info')
    
    net = Containernet(controller=Controller)
    info('*** Adding controller\n')
    net.addController('c0')
    
    info('*** Adding central gateway switch\n')
    s1 = net.addSwitch('s1')
    
    info('*** Adding Gateway Node (System 2)\n')
    # Expected to run the SLM agentic framework
    gateway = net.addDocker(
        'gateway', ip='10.0.0.254',
        dimage="iiomt_gateway:latest",
        environment={"ROLE": "gateway"}
    )
    net.addLink(gateway, s1)
    
    edge_nodes = []
    for i in range(1, num_edges + 1):
        info(f'*** Adding Edge Node {i} (System 1)\n')
        # Expected to run the INT8 CNN-BiGRU inference engine
        edge = net.addDocker(
            f'edge{i}', ip=f'10.0.0.{i}',
            dimage="iiomt_edge:latest",
            environment={"ROLE": "edge", "EDGE_ID": str(i)},
            mem_limit='128m',
            cpu_quota=50000  # 0.5 CPU core
        )
        edge_nodes.append(edge)
        
        # Link edge to central switch with simulated IIoMT latency
        net.addLink(edge, s1, cls=TCLink, delay='5ms', bw=100)
        
        info(f'*** Adding Replay Node {i} (tcpreplay)\n')
        replay = net.addDocker(
            f'replay{i}', ip=f'10.0.0.{100+i}',
            dimage="ubuntu:focal",
            environment={"ROLE": "replay"}
        )
        # Link replay node to inject traffic targeted at edge node
        net.addLink(replay, s1, cls=TCLink, delay='1ms', bw=1000)

    info('*** Starting network\n')
    net.start()
    
    info('*** Running CLI\n')
    CLI(net)
    
    info('*** Stopping network\n')
    net.stop()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Containernet Topology for IIoMT")
    parser.add_argument('--edges', type=int, default=3, help="Number of edge nodes")
    args = parser.parse_args()
    create_topology(args.edges)
