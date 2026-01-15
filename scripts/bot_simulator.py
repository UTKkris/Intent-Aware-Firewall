
import time
import warnings
from scapy.all import IP, TCP, send, RandShort
import numpy as np
import sys 


warnings.filterwarnings("ignore")


TARGET_IP = "192.0.2.1" 
TARGET_PORT = 443 


PACKETS_PER_BURST = 5
TIME_BETWEEN_PACKETS = 0 
PAUSE_DURATION = 8

print(f"--- SCAPY BOT SIMULATOR (V5 - Safe Mode): ON ---")
print(f"Targeting: {TARGET_IP}:{TARGET_PORT}")
print(f"Mode: Very short bursts ({PACKETS_PER_BURST} pkts @ max speed), {PAUSE_DURATION}s pauses.")
print("Press Ctrl+C to stop.")

count = 0
try:
    while True:
        
        print(f"\n--- STARTING BURST ({PACKETS_PER_BURST} packets) ---", flush=True)

        
        fixed_source_port = np.random.randint(10000, 65000)
        
        
        packet = IP(dst=TARGET_IP) / TCP(dport=TARGET_PORT, sport=fixed_source_port) 

       
        send(packet, count=PACKETS_PER_BURST, inter=TIME_BETWEEN_PACKETS, verbose=0)

        count += PACKETS_PER_BURST
        print(f"Burst complete. Total sent: {count}", flush=True)

        
        print(f"--- PAUSE PHASE ({PAUSE_DURATION} seconds) ---", flush=True)
        time.sleep(PAUSE_DURATION)

except KeyboardInterrupt:
    print(f"\n--- BOT SIMULATOR: OFF ---")
    print(f"Total packets sent: {count}")
except Exception as e:
    print(f"\nAn error occurred: {e}")
    print("This script must be run as administrator (or with sudo).")