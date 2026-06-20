#!/usr/bin/env python3
import time
from mininet.net import Containernet
from mininet.node import Controller
from mininet.cli import CLI
from mininet.link import TCLink
from mininet.log import info, setLogLevel

setLogLevel('info')

def create_topology():
    net = Containernet(controller=Controller)
    
    info('*** Adding controller\n')
    net.addController('c0')
    
    info('*** Adding Docker containers (Edge nodes with 128MB RAM and 0.5 CPU limits)\n')
    # Edge Node 1: Represents an Infusion Pump Sensor Gateway
    edge1 = net.addDocker('edge1', ip='10.0.0.101', dimage='ubuntu:trusty', 
                          mem_limit=128 * 1024 * 1024, cpu_quota=50000)
    
    # Edge Node 2: Represents a Modbus Industrial Actuator Gateway
    edge2 = net.addDocker('edge2', ip='10.0.0.102', dimage='ubuntu:trusty', 
                          mem_limit=128 * 1024 * 1024, cpu_quota=50000)
    
    info('*** Adding Gateway Server (System 2 with SLM access and Mosquitto Broker)\n')
    # Gateway node has larger constraints as it represents the localized edge server
    gateway = net.addDocker('gateway', ip='10.0.0.254', dimage='ubuntu:trusty')
    
    info('*** Adding switch\n')
    s1 = net.addSwitch('s1')
    
    info('*** Creating links with TC limits (1Gbps, 1ms delay)\n')
    net.addLink(edge1, s1, cls=TCLink, bw=1000, delay='1ms')
    net.addLink(edge2, s1, cls=TCLink, bw=1000, delay='1ms')
    net.addLink(gateway, s1, cls=TCLink, bw=1000, delay='1ms')
    
    info('*** Starting network\n')
    net.start()
    
    info('*** Testing connectivity\n')
    net.ping([edge1, gateway])
    net.ping([edge2, gateway])
    
    info('*** Starting MQTT Broker on Gateway\n')
    gateway.cmd('apt-get update && apt-get install -y mosquitto mosquitto-clients')
    gateway.cmd('/etc/init.d/mosquitto start')
    
    info('*** Topology Ready. Running CLI...\n')
    CLI(net)
    
    info('*** Stopping network\n')
    net.stop()

if __name__ == '__main__':
    create_topology()
