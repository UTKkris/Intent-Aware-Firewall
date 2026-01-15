import pandas as pd
import numpy as np
import joblib
import json
import time
from collections import defaultdict
import sys
import threading
import queue
import os
import platform

try:
    from scapy.all import sniff, IP, TCP, UDP
except ImportError:
    print("Scapy not found. Please install it: pip install scapy")
    sys.exit(1)

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
warnings.filterwarnings("ignore", category=FutureWarning)

try:
    print("Loading model and assets...")
    clf = joblib.load('intent_firewall_model_LGBM.joblib')
    scaler = joblib.load('data_scaler_LGBM.joblib')
    with open('size_stats.json', 'r') as f:
        size_stats = json.load(f)
    TRAIN_MEAN = size_stats['mean']
    TRAIN_STD = size_stats['std']
    with open('shared_features.json', 'r') as f:
        SHARED_COLS = json.load(f)
    print("Model loaded successfully. Starting firewall service...")
except FileNotFoundError:
    print("Error: Could not find model/asset files.")
    print("Please run 'load_data.py' first to train and save the model.")
    sys.exit(1)

PACKET_QUEUE = queue.Queue()
BLOCKED_IPS = set()
FLOW_MANAGER = {}
FLOW_TIMEOUT = 15

def packet_sniffer(iface_name):
    print(f"\n[Sniffer Thread] Starting packet sniffing on {iface_name}...")
    try:
        sniff(prn=PACKET_QUEUE.put, store=0, iface=iface_name)
    except Exception as e:
        print(f"[Sniffer Thread] Error: {e}", flush=True)

def block_ip(ip_address):
    system = platform.system()
    try:
        if system == "Windows":
            command = f'netsh advfirewall firewall add rule name="Block Bot {ip_address}" dir=out action=block remoteip={ip_address}'
            print(f"   Running: {command}", flush=True)
            os.system(command)
        elif system == "Linux" or system == "Darwin":
            command = f"iptables -A OUTPUT -d {ip_address} -j DROP"
            print(f"   Running: {command}", flush=True)
            os.system(command)
    except Exception as e:
        print(f"   ERROR trying to block {ip_address}: {e}", flush=True)

def get_iat_stats(iat_list):
    if len(iat_list) < 2:
        return 0, 0, 0, 0, 0
    iats = np.array(iat_list[1:])
    return np.mean(iats), np.max(iats), np.min(iats), np.std(iats), np.sum(iats)

def get_pkt_len_stats(pkt_len_list):
    if not pkt_len_list:
        return 0, 0, 0, 0
    return np.max(pkt_len_list), np.min(pkt_len_list), np.mean(pkt_len_list), np.std(pkt_len_list)

def process_packet(packet):
    global FLOW_MANAGER, BLOCKED_IPS
    if not (packet.haslayer(IP) and (packet.haslayer(TCP) or packet.haslayer(UDP))):
        return
    now = pd.to_datetime(packet.time, unit='s', utc=True)
    src_ip = packet[IP].src
    dst_ip = packet[IP].dst
    src_port = packet[TCP].sport if packet.haslayer(TCP) else packet[UDP].sport
    dst_port = packet[TCP].dport if packet.haslayer(TCP) else packet[UDP].dport
    protocol = packet[IP].proto
    if (src_ip, src_port) < (dst_ip, dst_port):
        flow_key = (src_ip, src_port, dst_ip, dst_port, protocol)
        direction = 'fwd'
    else:
        flow_key = (dst_ip, dst_port, src_ip, src_port, protocol)
        direction = 'bwd'
    if flow_key not in FLOW_MANAGER:
        FLOW_MANAGER[flow_key] = {
            'Src IP': src_ip, 'Dst Port': dst_port, 'start_time': now,
            'fwd_packet_count': 0, 'bwd_packet_count': 0,
            'fwd_lengths': [], 'bwd_lengths': [],
            'fwd_header_lengths': [], 'bwd_header_lengths': [],
            'fwd_iat_list': [], 'bwd_iat_list': [],
            'last_fwd_time': now, 'last_bwd_time': now,
            'flags': {'FIN': 0, 'SYN': 0, 'RST': 0, 'PSH': 0, 'ACK': 0, 'URG': 0, 'CWE': 0, 'ECE': 0}
        }
    flow = FLOW_MANAGER[flow_key]
    packet_len = len(packet)
    header_len = packet[IP].ihl * 4
    if direction == 'fwd':
        flow['fwd_packet_count'] += 1
        flow['fwd_lengths'].append(packet_len)
        flow['fwd_header_lengths'].append(header_len)
        fwd_iat = (now - flow['last_fwd_time']).total_seconds()
        flow['fwd_iat_list'].append(fwd_iat)
        flow['last_fwd_time'] = now
    else:
        flow['bwd_packet_count'] += 1
        flow['bwd_lengths'].append(packet_len)
        flow['bwd_header_lengths'].append(header_len)
        bwd_iat = (now - flow['last_bwd_time']).total_seconds()
        flow['bwd_iat_list'].append(bwd_iat)
        flow['last_bwd_time'] = now
    flow_duration = (now - flow['start_time']).total_seconds()
    if packet.haslayer(TCP):
        if packet[TCP].flags.F: flow['flags']['FIN'] += 1
        if packet[TCP].flags.S: flow['flags']['SYN'] += 1
        if packet[TCP].flags.R: flow['flags']['RST'] += 1
        if packet[TCP].flags.P: flow['flags']['PSH'] += 1
        if packet[TCP].flags.A: flow['flags']['ACK'] += 1
        if packet[TCP].flags.U: flow['flags']['URG'] += 1
        if packet[TCP].flags.E: flow['flags']['ECE'] += 1
        if packet[TCP].flags.C: flow['flags']['CWE'] += 1
    try:
        features = {}
        for col in SHARED_COLS: features[col] = 0.0
        features['Dst Port'] = dst_port
        features['Protocol'] = protocol
        features['Flow Duration'] = flow_duration
        features['Tot Fwd Pkts'] = flow['fwd_packet_count']
        features['Tot Bwd Pkts'] = flow['bwd_packet_count']
        features['TotLen Fwd Pkts'] = np.sum(flow['fwd_lengths'])
        features['TotLen Bwd Pkts'] = np.sum(flow['bwd_lengths'])
        fwd_max, fwd_min, fwd_mean, fwd_std = get_pkt_len_stats(flow['fwd_lengths'])
        features['Fwd Pkt Len Max'] = fwd_max
        features['Fwd Pkt Len Min'] = fwd_min
        features['Fwd Pkt Len Mean'] = fwd_mean
        features['Fwd Pkt Len Std'] = fwd_std
        bwd_max, bwd_min, bwd_mean, bwd_std = get_pkt_len_stats(flow['bwd_lengths'])
        features['Bwd Pkt Len Max'] = bwd_max
        features['Bwd Pkt Len Min'] = bwd_min
        features['Bwd Pkt Len Mean'] = bwd_mean
        features['Bwd Pkt Len Std'] = bwd_std
        if flow_duration > 0:
            features['Flow Byts/s'] = (features['TotLen Fwd Pkts'] + features['TotLen Bwd Pkts']) / flow_duration
            features['Flow Pkts/s'] = (flow['fwd_packet_count'] + flow['bwd_packet_count']) / flow_duration
        else:
            features['Flow Byts/s'] = 0.0
            features['Flow Pkts/s'] = 0.0
        all_iats = flow['fwd_iat_list'][1:] + flow['bwd_iat_list'][1:]
        if all_iats:
            features['Flow IAT Mean'] = np.mean(all_iats)
            features['Flow IAT Max'] = np.max(all_iats)
            features['Flow IAT Min'] = np.min(all_iats)
            features['Flow IAT Std'] = np.std(all_iats)
        fwd_iat_mean, fwd_iat_max, fwd_iat_min, fwd_iat_std, fwd_iat_tot = get_iat_stats(flow['fwd_iat_list'])
        features['Fwd IAT Tot'] = fwd_iat_tot
        features['Fwd IAT Mean'] = fwd_iat_mean
        features['Fwd IAT Max'] = fwd_iat_max
        features['Fwd IAT Min'] = fwd_iat_min
        features['Fwd IAT Std'] = fwd_iat_std
        bwd_iat_mean, bwd_iat_max, bwd_iat_min, bwd_iat_std, bwd_iat_tot = get_iat_stats(flow['bwd_iat_list'])
        features['Bwd IAT Tot'] = bwd_iat_tot
        features['Bwd IAT Mean'] = bwd_iat_mean
        features['Bwd IAT Max'] = bwd_iat_max
        features['Bwd IAT Min'] = bwd_iat_min
        features['Bwd IAT Std'] = bwd_iat_std
        if fwd_iat_tot > 0: features['Fwd Pkts/s'] = flow['fwd_packet_count'] / fwd_iat_tot
        if bwd_iat_tot > 0: features['Bwd Pkts/s'] = flow['bwd_packet_count'] / bwd_iat_tot
        if flow['fwd_header_lengths']: features['Fwd Header Len'] = np.sum(flow['fwd_header_lengths'])
        if flow['bwd_header_lengths']: features['Bwd Header Len'] = np.sum(flow['bwd_header_lengths'])
        features['FIN Flag Cnt'] = flow['flags']['FIN']
        features['SYN Flag Cnt'] = flow['flags']['SYN']
        features['RST Flag Cnt'] = flow['flags']['RST']
        features['PSH Flag Cnt'] = flow['flags']['PSH']
        features['ACK Flag Cnt'] = flow['flags']['ACK']
        features['URG Flag Cnt'] = flow['flags']['URG']
        features['CWE Flag Count'] = flow['flags']['CWE']
        features['ECE Flag Cnt'] = flow['flags']['ECE']
        features['Down/Up Ratio'] = flow['bwd_packet_count'] / flow['fwd_packet_count'] if flow['fwd_packet_count'] > 0 else 0
        features['Pkt Size Avg'] = (features['TotLen Fwd Pkts'] + features['TotLen Bwd Pkts']) / (flow['fwd_packet_count'] + flow['bwd_packet_count']) if (flow['fwd_packet_count'] + flow['bwd_packet_count']) > 0 else 0
        final_features_df = pd.DataFrame([features])
    except Exception as e:
        print(f"Feature engineering error: {e}", flush=True)
        return
    drop_cols = ['Flow ID', 'Src IP', 'Src Port', 'Dst IP', 'Timestamp']
    final_features_df = final_features_df.drop(columns=drop_cols, errors='ignore')
    final_features_df = final_features_df.reindex(columns=SHARED_COLS, fill_value=0)
    try:
        final_features_scaled = scaler.transform(final_features_df)
        prediction = clf.predict(final_features_scaled)
        intent = prediction[0]
    except Exception as e:
        print(f"DEBUG: Scaling or Prediction error: {e}", flush=True)
        return
    print(f"Time: {now.isoformat()} | Flow: {src_ip}:{src_port} -> {dst_ip}:{dst_port} | Intent: {intent} | IAT Min: {features.get('Fwd IAT Min', 0):.4f}s", flush=True)
    if intent == "Attack":
        print(f"---! MALICIOUS ACTIVITY DETECTED !---", flush=True)
        print(f"   Flow: {src_ip}:{src_port} -> {dst_ip}:{dst_port} flagged as '{intent}'.", flush=True)
        if dst_ip and (dst_ip not in BLOCKED_IPS):
            print(f"   ACTION: Blocking destination IP {dst_ip}...", flush=True)
            block_ip(dst_ip)
            BLOCKED_IPS.add(dst_ip)
        elif dst_ip:
            print(f"   Destination IP {dst_ip} is already blocked.", flush=True)

def timeout_flows():
    global FLOW_MANAGER
    while True:
        time.sleep(FLOW_TIMEOUT)
        now = pd.to_datetime('now', utc=True)
        flows_to_delete = []
        try:
            for flow_key in list(FLOW_MANAGER.keys()):
                if flow_key in FLOW_MANAGER:
                    flow_data = FLOW_MANAGER[flow_key]
                    if (now - flow_data['last_fwd_time']).total_seconds() > FLOW_TIMEOUT and (now - flow_data['last_bwd_time']).total_seconds() > FLOW_TIMEOUT:
                        flows_to_delete.append(flow_key)
            if flows_to_delete:
                print(f"[Flow Manager] Timing out {len(flows_to_delete)} old flows.", flush=True)
                for key in flows_to_delete:
                    if key in FLOW_MANAGER:
                        del FLOW_MANAGER[key]
        except Exception as e:
            print(f"[Flow Manager] Error during timeout: {e}", flush=True)

if __name__ == "__main__":
    INTERFACE_NAME = "Realtek RTL8852BE WiFi 6 802.11ax PCIe Adapter"
    print(f"\nStarting firewall. Sniffing on: {INTERFACE_NAME}")
    print("NOTE: This requires administrator or sudo privileges.")
    try:
        sniffer_thread = threading.Thread(
            target=packet_sniffer,
            args=(INTERFACE_NAME,),
            daemon=True
        )
        sniffer_thread.start()
        timeout_thread = threading.Thread(target=timeout_flows, daemon=True)
        timeout_thread.start()
        print("[Main Thread] Waiting for packets from queue...")
        while True:
            packet = PACKET_QUEUE.get()
            process_packet(packet)
            PACKET_QUEUE.task_done()
    except PermissionError:
        print("\n[Main Thread] Error: Permission denied.")
        print("Please run this script as administrator (or with 'sudo').")
    except KeyboardInterrupt:
        print("\n[MainThread] Shutting down...")
        sys.exit(0)
    except Exception as e:
        print(f"\n[Main Thread] An error occurred: {e}")
