# Replication of the paper "[Generalized Data Placement Strategies for Racetrack Memories](https://arxiv.org/abs/1912.03507)"

## Setup

```bash
git clone --recurse-submodules <url>
```

If you cloned w/o `--recurse-submodules`, please do the follwoing to correctly clone the submodule as well:

```bash
cd gdpRTM
git submodule update --init --recursive
```

If you need to update the submodule:

```bash
git submodule update --remote --merge
```

## RTSim

Prerequisites

- `scons`
- `gcc-11`
- `python`

Build RTSim, which utilizes `scons` and `gcc-11`

```bash
make rtsim
```

The executable is at `./RTSim/nvmain.fast`

## Usage

The `main.py` is the implementation of the algorithm in the paper, execute it by the following command, no extra dependencies needed.

```bash
python ./main.py
```

## TODOs

### 1. Algorithm Implementation

- [x] Main logics of Algorithm 1 (DMA)
- [ ] [Chen](https://ieeexplore.ieee.org/document/7445241)'s S-DBC-P algorithm for intra-DBC placement of variables in $V_{ndj}$
- [ ] [ShiftReduce](https://arxiv.org/abs/1903.03597) algorithm for same purpose of the Chen's algorithm
- [ ] AFD-OFU
  - [ ] AFD
  - [ ] OFU
- [ ] DMA-OFU
  - [x] DMA
  - [ ] OFU
- [ ] DMA-Chen
  - [x] DMA
  - [ ] Chen
- [ ] DMA-SR
  - [x] DMA
  - [ ] ShiftReduce
- [ ] GA (In paper's III-C)
- [ ] RW (In paper's III-C)

### 2. RTSim Integration & Automation

- [ ] Convert variable DBC placement to `.nvt` trace for RTSim
- [ ] Prepare RTSim hardware configurations according to Paper Table I (2, 4, 8, 16 DBCs matching 512, 256, 128, 64 domains)
  - [x] 2 DBCs
  - [ ] 4 DBCs
  - [ ] 8 DBCs
  - [ ] 16 DBCs
- [ ] Use python (directly implement in `main.py`) or bash scripts to invoke RTSim and store simulation results

### 3. Evaluation & Verification

- [ ] Implement a result parser to extract `totalnumShifts`, `*Latency`, and `*Energy` metrics from RTSim output
- [ ] Calculate baseline (AFD-OFU) results for normalization
- [ ] Generate comparative plots to verify against Paper Fig. 4 (Shift reduction) and Fig. 5 (Energy consumption)