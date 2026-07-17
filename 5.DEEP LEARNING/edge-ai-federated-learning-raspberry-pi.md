# Edge AI + Federated Learning on Raspberry Pi

*A Beginner-Friendly Technical Breakdown*

---

## Table of Contents

1. [Why Can't We Deploy Heavy Models on Raspberry Pi?](#1-why-cant-we-deploy-heavy-models-on-raspberry-pi)
2. [Lightweight Model Architectures](#2-lightweight-model-architectures)
3. [Model Compression Techniques](#3-model-compression-techniques)
4. [Edge Deployment Ecosystem](#4-edge-deployment-ecosystem)
5. [Federated Learning](#5-federated-learning)
6. [The FedAvg Algorithm](#6-the-fedavg-algorithm)
7. [Communication Efficiency](#7-communication-efficiency)
8. [Privacy Protection](#8-privacy-protection)
9. [Non-IID Data and FedProx](#9-non-iid-data-and-fedprox)
10. [End-to-End Deployment Pipeline](#10-end-to-end-deployment-pipeline)
11. [Example Interview Answer](#example-interview-answer)
12. [Interview Memory Trick](#interview-memory-trick)

---

## 1. Why Can't We Deploy Heavy Models on Raspberry Pi?

Large deep learning models such as **ResNet-152** or **Transformers** require:

- Large memory (hundreds of MBs to GBs)
- Powerful GPUs
- Billions of computations
- High energy consumption

A Raspberry Pi, in contrast, has:

- Limited RAM (2–8 GB)
- An ARM CPU
- A limited power budget
- No dedicated high-end GPU

> **Conclusion:** We need lightweight, edge-native architectures.

---

## 2. Lightweight Model Architectures

### A. MobileNet (v2/v3/v4)

**Core Idea:** Depthwise Separable Convolutions

Traditional convolution performs spatial filtering and channel mixing in a single, expensive step:

```
Input → Spatial Filtering + Channel Mixing → Output
```

MobileNet splits this into two cheaper operations:

**Step 1 — Depthwise Convolution:** apply one filter per channel.

```
Red Channel   → Filter
Green Channel → Filter
Blue Channel  → Filter
```

**Step 2 — Pointwise Convolution (1×1):** combine the filtered channels.

```
Filtered Channels
        ↓
  1×1 Convolution
        ↓
      Output
```

**Advantages**

- Reduces FLOPs by ~8×–9×
- Much smaller models
- Minimal accuracy loss
- Ideal for edge devices

---

### B. ShuffleNet

**Problem:** Standard convolutions are computationally expensive.

**Solution:** Combine pointwise **group convolutions** with **channel shuffling**.

```
Input
   ↓
Group Convolution
   ↓
Channel Shuffle
   ↓
Output
```

**Why shuffle?**

Without shuffling, groups never communicate:

```
Group A talks only to Group A
Group B talks only to Group B
```

After shuffling, all groups exchange information.

**Advantages**

- Extremely computationally efficient
- Maintains feature communication across groups
- Excellent for mobile and edge hardware

---

### C. SqueezeNet

**Core Idea:** Fire Modules — squeeze channels down, then expand them back out.

```
Input
   ↓
Squeeze Layer (1×1)
   ↓
Expand Layer (1×1 + 3×3)
   ↓
Output
```

**Example:**

```
256 channels → 32 channels → 256 channels
```

**Advantages**

- ~50× fewer parameters than AlexNet
- Similar accuracy
- Very small memory footprint

---

## 3. Model Compression Techniques

After training, models are compressed to fit edge device constraints.

### A. Quantization

Reduces the numerical precision used to store weights.

**Original FP32 weights** (32 bits each):

```
 0.23456789
-0.98765432
 0.54321987
```

**Converted to INT8** (8 bits each):

```
 24
-99
 54
```

**Benefits**

| Metric | Before | After |
|---|---|---|
| Size | 100 MB | 25 MB |
| RAM usage | High | Low |
| Speed | Slow | Faster |

**Advantages**

- ~4× smaller models
- Lower memory usage
- Faster inference
- Reduced power consumption

#### Types of Quantization

**Post-Training Quantization (PTQ)**

```
Train Model → Convert to INT8
```

- ✅ Fast, easy
- ⚠️ Slight accuracy drop

**Quantization-Aware Training (QAT)**

```
Train while simulating INT8 errors → Export INT8 model
```

- ✅ Preserves accuracy
- ✅ Best choice for edge deployment

---

### B. Pruning

Many weights contribute almost nothing to the model's output:

```
 0.00001
 0.000003
-0.000005
```

These low-magnitude weights can be safely removed:

```
100 Million Parameters → 40 Million Parameters
```

**Advantages**

- Smaller models
- Faster execution
- Less memory usage

---

### C. Knowledge Distillation

A large **teacher model** transfers its knowledge to a small **student model**.

| | Teacher | Student |
|---|---|---|
| Model | ResNet-152 | MobileNet |
| Accuracy | 98% | — |
| Size | 500 MB | 5 MB |

Instead of learning only hard labels:

```
Cat
Dog
Bird
```

The student learns the teacher's full probability distribution:

```
Cat:  95%
Dog:   3%
Bird:  2%
```

**Benefits**

- Small model size
- Retains large-model accuracy knowledge
- Efficient deployment

---

## 4. Edge Deployment Ecosystem

### TensorFlow Lite (TFLite)

```
TensorFlow Model → Convert to .tflite → Deploy on Raspberry Pi
```

**Advantages**

- Industry standard
- Supports INT8 quantization
- Hardware acceleration
- Low memory consumption

### ONNX Runtime Mobile

```
PyTorch/TensorFlow → ONNX → ONNX Runtime Mobile
```

**Advantages**

- Cross-platform
- Graph optimization
- Fast execution

### PyTorch Edge / ExecuTorch

Provides:

- Native PyTorch deployment
- Modular edge execution
- Mobile and embedded support

---

## 5. Federated Learning

### Traditional (Centralized) ML

```
Device Data → Cloud Server → Training
```

**Problems**

- Privacy risks
- Massive data transfer

### Federated Learning

```
Device 1 → Train Locally
Device 2 → Train Locally         Send Model Updates        Central Server → Aggregate
Device 3 → Train Locally
```

**Key Idea:** Data never leaves the device — only model updates are shared.

---

## 6. The FedAvg Algorithm

Suppose three devices hold different amounts of data:

```
Device A : 1,000 samples
Device B : 2,000 samples
Device C : 7,000 samples
```

Each device trains locally, then the server computes a **weighted average**:

```
Global Model = Weighted Average of Local Models
```

Device C contributes more to the global model because it has more data. This is **Federated Averaging (FedAvg)**.

---

## 7. Communication Efficiency

The major bottleneck in Federated Learning is **communication**, not computation.

**Example:**

```
10,000 devices × 20 MB updates = huge bandwidth consumption
```

### Top-k Sparsification

Instead of sending all gradients:

```
1,000,000 gradients
```

Send only the most important ones:

```
Top 10,000 gradients
```

### SignSGD

Instead of transmitting full-precision values:

```
 0.345
-0.654
 0.234
```

Transmit only their sign:

```
+
-
+
```

This drastically reduces communication costs.

---

## 8. Privacy Protection

Model updates can still leak sensitive information about local data.

### Differential Privacy (DP)

Add calibrated noise to gradients before sharing:

```
Original Gradient + Random Noise = Private Gradient
```

**Benefits**

- Mathematical privacy guarantees
- Prevents reconstruction attacks

### Secure Multi-Party Computation (SMPC)

The server computes:

```
Average(Model Updates)
```

...without ever seeing any individual device's update.

**Benefits**

- Strong privacy
- Cryptographic guarantees

---

## 9. Non-IID Data and FedProx

Real-world data is rarely identical (IID) across devices.

**Example:**

| Location | Environment |
|---|---|
| Home A | Quiet |
| Home B | Children talking |
| Home C | Television noise |

This variation creates **Non-IID data**.

### Problem with FedAvg

Local models can drift too far apart from one another, hurting convergence.

### FedProx Solution

Adds a regularization term that keeps local models closer to the global model.

**Benefits**

- Stable convergence
- Better performance on heterogeneous data

---

## 10. End-to-End Deployment Pipeline

```
Collect Data
     ↓
Train Model
     ↓
Compress Model
     ↓
Quantize Model
     ↓
Convert to Edge Format
     ↓
Deploy on Raspberry Pi
     ↓
Run Federated Learning
     ↓
Optimize Communication
     ↓
Ensure Privacy
     ↓
Monitor Device Resources
```

---

## Example Interview Answer

**Question:** *"How would you design a keyword detection system on 10,000 Raspberry Pi smart-home devices using Federated Learning?"*

**Step 1 — Model Selection**
> I would choose a lightweight architecture such as MobileNetV3-Small or a custom 1D CNN operating on audio spectrograms.

**Step 2 — Model Compression**
> I would apply Quantization-Aware Training to generate an INT8 model with a memory footprint below 5 MB while preserving accuracy.

**Step 3 — Edge Deployment**
> I would convert the model into TensorFlow Lite format and deploy it on Raspberry Pi devices using the TFLite runtime.

**Step 4 — Federated Learning**
> I would use the Flower framework to orchestrate federated learning among the devices.

**Step 5 — Handling Non-IID Data**
> Since user audio data differs significantly due to accents and background noise, I would prefer FedProx over FedAvg for more stable convergence.

**Step 6 — Communication Optimization**
> I would apply techniques such as Top-k gradient sparsification or SignSGD to reduce network bandwidth requirements.

**Step 7 — Privacy Protection**
> I would combine Federated Learning with Differential Privacy and Secure Multi-Party Computation to ensure user privacy.

**Step 8 — Resource Management**
> To preserve device health and user experience, local training would execute only when:
> - The device is idle
> - Connected to unmetered Wi-Fi
> - CPU temperature is below 60°C
> - Adequate battery/power is available

---

## Interview Memory Trick

```
Choose Lightweight Model
         ↓
   Compress Model
         ↓
   Quantize Model
         ↓
   Deploy to Edge
         ↓
Apply Federated Learning
         ↓
Optimize Communication
         ↓
   Protect Privacy
         ↓
Manage Device Resources
```

> 💡 If you can explain these eight steps clearly, you can answer most Edge AI + Federated Learning interview questions confidently.
