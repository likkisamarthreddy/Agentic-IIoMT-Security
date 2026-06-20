import subprocess
import argparse
import time
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s")
logger = logging.getLogger("iimt.pcap_replay")

def run_tcpreplay(pcap_path: str, interface: str = "eth0", pps: int = 500):
    """
    Blasts a raw PCAP file onto the network interface using tcpreplay.
    Matches the Phase 1 specification of the paper (500 packets/sec).
    """
    pcap_file = Path(pcap_path)
    if not pcap_file.exists():
        logger.error(f"PCAP file not found: {pcap_path}")
        return

    logger.info(f"Starting Phase 1 PCAP Replay...")
    logger.info(f"Target Interface: {interface}")
    logger.info(f"Target Rate:      {pps} packets/sec")
    logger.info(f"PCAP File:        {pcap_file.name}")
    
    cmd = [
        "tcpreplay",
        "--intf1", interface,
        "--pps", str(pps),
        "--loop", "1",
        str(pcap_file)
    ]
    
    try:
        start_time = time.time()
        # Ensure we don't block the caller completely, but wait for completion
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                logger.info(f"tcpreplay: {output.strip()}")
                
        stderr = process.stderr.read()
        if stderr:
            logger.warning(f"tcpreplay stderr: {stderr.strip()}")
            
        elapsed = time.time() - start_time
        logger.info(f"PCAP replay finished in {elapsed:.2f} seconds.")
        
    except FileNotFoundError:
        logger.error("tcpreplay executable not found! Please install it (e.g., sudo apt install tcpreplay) to run real PCAP injection.")
    except Exception as e:
        logger.error(f"Error executing tcpreplay: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 1: Raw PCAP Traffic Replay")
    parser.add_argument("--pcap", type=str, required=True, help="Path to CICIoMT2023 or Edge-IIoTset PCAP file")
    parser.add_argument("--interface", type=str, default="eth0", help="Network interface to inject packets into")
    parser.add_argument("--rate", type=int, default=500, help="Packets per second (pps)")
    args = parser.parse_args()
    
    run_tcpreplay(args.pcap, args.interface, args.rate)
