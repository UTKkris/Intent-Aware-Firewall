# Intent-Aware Firewall: Real-Time AI Intrusion Prevention System

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Machine Learning](https://img.shields.io/badge/ML-LightGBM%20%7C%20Random%20Forest-orange)
![Status](https://img.shields.io/badge/Status-Research%20Prototype-green)

## 📌 Overview
The **Intent-Aware Firewall** is a next-generation Intrusion Prevention System (IPS) that moves beyond static rules (IP/Port blocking) to analyze the **behavioral intent** of network traffic in real-time.

By leveraging Machine Learning (LightGBM & Random Forest) trained on the **CIC-IDS-2018 dataset**, this system calculates complex statistical features (Inter-Arrival Time, Jitter, Flow Duration) on live packets to distinguish between legitimate human users and malicious botnets/DDoS attacks. When malicious intent is detected, it autonomously interacts with the OS kernel (`netsh`) to block the threat.

**Key Research Finding:** This project bridges the "fidelity gap" between offline batch training and real-time packet stream analysis using a custom stateful Flow Manager.

---

## 🚀 Key Features
* **Behavioral Analysis:** Does not rely on blacklists. Detects zero-day botnets based on traffic patterns (speed, variance, burstiness).
* **Real-Time Flow Manager:** A custom bi-directional engine that aggregates raw packets into "Flows" and calculates 79+ statistical features on the fly.
* **Hybrid AI Engine:**
    * **LightGBM:** Precision-focused model (98.66% Accuracy, Low False Positives).
    * **Random Forest:** High-recall model (99.87% Accuracy) for aggressive defense.
* **Dynamic Blocking:** Automatically inserts firewall rules into the Windows Kernel via `netsh advfirewall` to terminate active attacks.
* **Safe-Listing Module:** Includes logic to handle "Concept Drift" (e.g., modern QUIC/HTTP3 traffic from Google) to prevent disruption of legitimate services.

---

## 🛠️ Architecture
1.  **Packet Sniffer (Scapy):** Intercepts raw frames from the Network Interface Card (NIC).
2.  **Flow Manager:** Groups packets by 5-tuple `(SrcIP, DstIP, SrcPort, DstPort, Proto)` and maintains a 30-second state window.
3.  **Feature Extractor:** Calculates statistical variances (e.g., `Fwd IAT Min`, `Pkt Len Std`) in real-time.
4.  **ML Inference:** The pre-trained LightGBM model classifies the flow as `Benign` or `Attack`.
5.  **Actuator:** If `Attack` is detected -> Execute `netsh` block rule.

---

## 📂 Dataset Setup (Important)
Due to file size limits, the training data is **not** included in this repository.

1.  **Download the Dataset:**
    * Access the **CSE-CIC-IDS2018** dataset from the [Canadian Institute for Cybersecurity](https://www.unb.ca/cic/datasets/ids-2018.html) or [Kaggle](https://www.kaggle.com/datasets/solarmainframe/ids-intrusion-csv).
2.  **Prepare the Folder:**
    * Create a folder named `data` in the root directory.
    * Place the CSV files (or your processed `.parquet` file) inside `data/`.
3.  **Note:** The `.gitignore` file is configured to ignore heavy data files (`*.csv`, `*.parquet`) to keep the repository lightweight.

---

## 📊 Performance Metrics
The models were trained on the **CIC-IDS-2018** dataset (16.2 Million rows) using a high-performance workstation.

| Metric | LightGBM (Recommended) | Random Forest |
| :--- | :--- | :--- |
| **Accuracy** | **98.66%** | 99.87% |
| **Precision** | 96.11% | ~99.0% |
| **Recall** | 95.99% | ~99.5% |
| **Inference Speed**| Fast (<10ms) | Moderate |
| **Behavior** | Stable, Low False Positives | Aggressive, prone to blocking QUIC/HTTPS |

---

## 💻 Tech Stack
* **Language:** Python 3.x
* **Packet Manipulation:** Scapy
* **Machine Learning:** LightGBM, Scikit-Learn, Joblib
* **Data Processing:** Pandas, NumPy (Parquet format for Big Data)
* **OS Integration:** Windows `netsh` (Network Shell)

---

## ⚙️ Installation

### Prerequisites
* Windows 10/11 (Required for `netsh` blocking commands).
* Python 3.9 or higher.
* **Npcap** (Required for Scapy to sniff packets). [Download Npcap](https://npcap.com/).

### Setup
1.  Clone the repository:
    ```bash
    git clone [https://github.com/yourusername/Intent-Aware-Firewall.git](https://github.com/yourusername/Intent-Aware-Firewall.git)
    cd Intent-Aware-Firewall
    ```

2.  Create a Virtual Environment:
    ```bash
    python -m venv venv
    venv\Scripts\activate
    ```

3.  Install Dependencies:
    ```bash
    pip install -r requirements.txt
    ```

---

## 🏃 Usage

### 1. Training the Model (Optional)
*Note: Pre-trained models (`.joblib`) are included. Only run this if you want to retrain on new data.*
```bash
python scripts/load_data.py
# Or for Random Forest:
# python scripts/train_rf.py