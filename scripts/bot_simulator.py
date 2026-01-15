# bot_simulator.py (Scapy Version - V5 Short Bursts - SAFE TARGET)
import time
import warnings
from scapy.all import IP, TCP, send, RandShort
import numpy as np
import sys 

# --- Forcefully disable ALL warnings ---
warnings.filterwarnings("ignore")

# --- SAFETY CHANGE ---
# We use a reserved Test IP so we don't block real internet access
TARGET_IP = "192.0.2.1" 
TARGET_PORT = 443 

# --- ADJUSTMENTS FOR BOT PATTERN ---
PACKETS_PER_BURST = 5
TIME_BETWEEN_PACKETS = 0 # Max speed
PAUSE_DURATION = 8

print(f"--- SCAPY BOT SIMULATOR (V5 - Safe Mode): ON ---")
print(f"Targeting: {TARGET_IP}:{TARGET_PORT}")
print(f"Mode: Very short bursts ({PACKETS_PER_BURST} pkts @ max speed), {PAUSE_DURATION}s pauses.")
print("Press Ctrl+C to stop.")

count = 0
try:
    while True:
        # --- BURST PHASE ---
        print(f"\n--- STARTING BURST ({PACKETS_PER_BURST} packets) ---", flush=True)

        # Pick ONE fixed source port for this entire burst
        fixed_source_port = np.random.randint(10000, 65000)
        
        # Create a minimal IP/TCP packet
        packet = IP(dst=TARGET_IP) / TCP(dport=TARGET_PORT, sport=fixed_source_port) 

        # Send the burst
        send(packet, count=PACKETS_PER_BURST, inter=TIME_BETWEEN_PACKETS, verbose=0)

        count += PACKETS_PER_BURST
        print(f"Burst complete. Total sent: {count}", flush=True)

        # --- PAUSE PHASE ---
        print(f"--- PAUSE PHASE ({PAUSE_DURATION} seconds) ---", flush=True)
        time.sleep(PAUSE_DURATION)

except KeyboardInterrupt:
    print(f"\n--- BOT SIMULATOR: OFF ---")
    print(f"Total packets sent: {count}")
except Exception as e:
    print(f"\nAn error occurred: {e}")
    print("This script must be run as administrator (or with sudo).")