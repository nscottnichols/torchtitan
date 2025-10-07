> [!NOTE]
> This has been moved to:
> [🍹 BlendCorpus + TorchTitan @ ALCF](https://samforeman.me/posts/2025/09/12/torchtitan-blendcorpus/index.html)


<details closed><summary>[OLD]</summary>


# BlendCorpus + TorchTitan on Multiple Nodes of Aurora

- Using the branches at:
  - [saforem2/`blendcorpus` @ saforem2/reorg-imports](https://github.com/saforem2/blendcorpus/tree/saforem2/reorg-imports)[^PR]
  - [auroraGPT-ANL/`torchtitan` @ saforem2/blendcorpus](https://github.com/auroraGPT-ANL/torchtitan/tree/saforem2/blendcorpus)

[^PR]: Submitted [PR #2](https://github.com/zhenghh04/blendcorpus/pull/2)

## 🏃‍♂️ Running

- Clone repo:

  ```bash
  git clone https://github.com/auroraGPT-ANL/torchtitan
  cd torchtitan
  checkout saforem2/blendcorpus
  ```

- Setup env:

  ```bash
  source <(curl -L https://bit.ly/ezpz-utils)
  ezpz_setup_env
  ```

  <details closed><summary><code>output</code>:</summary>

  ```bash
  ; ssh x4112c1s0b0n0

  #[🐍 aurora_nre_models_frameworks-2025.2.0]
  #[/f/A/A/E/A/t/a/torchtitan][🌱 saforem2/blendcorpus][📝🤷✓]
  #[09/11/25 @ 14:08:35][x4112c1s0b0n0]
  ; source <(curl -L https://bit.ly/ezpz-utils)

  #[🐍 aurora_nre_models_frameworks-2025.2.0]
  #[/f/A/A/E/A/t/a/torchtitan][🌱 saforem2/blendcorpus][📝🤷✓]
  #[09/11/25 @ 14:08:37][x4112c1s0b0n0]
  ; ezpz_setup_env
  [2025-09-11-140838][I][/home/foremans/ezpz/src/ezpz/bin/utils.sh:2720] Detected PBS scheduler environment.
  [2025-09-11-140838][W][/home/foremans/ezpz/src/ezpz/bin/utils.sh:2756] Current working directory does not match PBS_O_WORKDIR! This may cause issues with the job submission.
  [2025-09-11-140838][W][/home/foremans/ezpz/src/ezpz/bin/utils.sh:2757] PBS_O_WORKDIR /lus/flare/projects/AuroraGPT/AuroraGPT-v1/Experiments/AuroraGPT-2B/tt/zhenghh04/torchtitan
  [2025-09-11-140838][W][/home/foremans/ezpz/src/ezpz/bin/utils.sh:2758] WORKING_DIR /lus/flare/projects/AuroraGPT/AuroraGPT-v1/Experiments/AuroraGPT-2B/tt/auroraGPT-ANL/torchtitan
  [2025-09-11-140838][W][/home/foremans/ezpz/src/ezpz/bin/utils.sh:2759] Exporting PBS_O_WORKDIR=WORKING_DIR=/lus/flare/projects/AuroraGPT/AuroraGPT-v1/Experiments/AuroraGPT-2B/tt/auroraGPT-ANL/torchtitan and continuing...
  [2025-09-11-140839][I][/home/foremans/ezpz/src/ezpz/bin/utils.sh:2486] running [ezpz_setup_env]...
  [2025-09-11-140839][I][/home/foremans/ezpz/src/ezpz/bin/utils.sh:1298] [PYTHON]
  [2025-09-11-140839][I][/home/foremans/ezpz/src/ezpz/bin/utils.sh:1327]   - Conda active, conda=/opt/aurora/25.190.0/frameworks/aurora_nre_models_frameworks-2025.2.0...
  [2025-09-11-140839][I][/home/foremans/ezpz/src/ezpz/bin/utils.sh:1328]   - No virtual_env found in environment
  [2025-09-11-140839][I][/home/foremans/ezpz/src/ezpz/bin/utils.sh:1142]   - Found python root at /opt/aurora/25.190.0/frameworks/aurora_nre_models_frameworks-2025.2.0
  [2025-09-11-140839][I][/home/foremans/ezpz/src/ezpz/bin/utils.sh:1157]   - No VIRTUAL_ENV found in environment!
  [2025-09-11-140839][I][/home/foremans/ezpz/src/ezpz/bin/utils.sh:1160]   - Looking for venv in venvs/aurora/torchtitan-aurora_nre_models_frameworks-2025.2.0...
  [2025-09-11-140839][I][/home/foremans/ezpz/src/ezpz/bin/utils.sh:1182]   - Activating existing venv in VENV_DIR=venvs/torchtitan-aurora_nre_models_frameworks-2025.2.0
  [2025-09-11-140839][I][/home/foremans/ezpz/src/ezpz/bin/utils.sh:1184]   - Found /lus/flare/projects/AuroraGPT/AuroraGPT-v1/Experiments/AuroraGPT-2B/tt/auroraGPT-ANL/torchtitan/venvs/aurora/torchtitan-aurora_nre_models_frameworks-2025.2.0/bin/activate
  [2025-09-11-140839][I][/home/foremans/ezpz/src/ezpz/bin/utils.sh:1353]   - Using python from: /lus/flare/projects/AuroraGPT/AuroraGPT-v1/Experiments/AuroraGPT-2B/tt/auroraGPT-ANL/torchtitan/venvs/aurora/torchtitan-aurora_nre_models_frameworks-2025.2.0/bin/python3
  [2025-09-11-140839][I][/home/foremans/ezpz/src/ezpz/bin/utils.sh:2335] [JOB]
  [2025-09-11-140839][I][/home/foremans/ezpz/src/ezpz/bin/utils.sh:2336]   - Setting up env for foremans
  [2025-09-11-140839][I][/home/foremans/ezpz/src/ezpz/bin/utils.sh:2337]   - Detected pbs scheduler
  [2025-09-11-140839][I][/home/foremans/ezpz/src/ezpz/bin/utils.sh:2338]   - Machine: aurora
  [2025-09-11-140839][I][/home/foremans/ezpz/src/ezpz/bin/utils.sh:2339]   - Hostname: x4112c1s0b0n0
  [2025-09-11-140842][I][/home/foremans/ezpz/src/ezpz/bin/utils.sh:2249]   - PBS_JOBID=7559761.aurora-pbs-0001.hostmgmt.cm.aurora.alcf.anl.gov
      to calculate:
        - num_hosts: 2
        - num_cores_per_host: 208
        - num_cpus_per_host: 104
        - num_gpus_per_host: 12
        - depth: 8
        - num_gpus: 24
  [2025-09-11-140842][I][/home/foremans/ezpz/src/ezpz/bin/utils.sh:1754] [HOSTS] - ezpz_print_hosts
  [2025-09-11-140842][I][/home/foremans/ezpz/src/ezpz/bin/utils.sh:1756]   - Detected PBS Scheduler
  [2025-09-11-140842][I][/home/foremans/ezpz/src/ezpz/bin/utils.sh:1774] [HOSTS]
  [2025-09-11-140842][I][/home/foremans/ezpz/src/ezpz/bin/utils.sh:1775]   - HOSTFILE=/var/spool/pbs/aux/7559761.aurora-pbs-0001.hostmgmt.cm.aurora.alcf.anl.gov
  [2025-09-11-140842][I][/home/foremans/ezpz/src/ezpz/bin/utils.sh:1776]   - NHOSTS=2
  [2025-09-11-140842][I][/home/foremans/ezpz/src/ezpz/bin/utils.sh:1777]   - HOSTS:
  [2025-09-11-140842][I][/home/foremans/ezpz/src/ezpz/bin/utils.sh:1780]     - [host:0] - x4112c1s0b0n0.hsn.cm.aurora.alcf.anl.gov
  [2025-09-11-140842][I][/home/foremans/ezpz/src/ezpz/bin/utils.sh:1780]     - [host:1] - x4112c1s1b0n0.hsn.cm.aurora.alcf.anl.gov
  [2025-09-11-140842][I][/home/foremans/ezpz/src/ezpz/bin/utils.sh:1941] [DIST_INFO]
  [2025-09-11-140842][I][/home/foremans/ezpz/src/ezpz/bin/utils.sh:1942]   - HOSTFILE=/var/spool/pbs/aux/7559761.aurora-pbs-0001.hostmgmt.cm.aurora.alcf.anl.gov
  [2025-09-11-140842][I][/home/foremans/ezpz/src/ezpz/bin/utils.sh:1943]   - NHOSTS=2
  [2025-09-11-140842][I][/home/foremans/ezpz/src/ezpz/bin/utils.sh:1944]   - NGPU_PER_HOST=12
  [2025-09-11-140842][I][/home/foremans/ezpz/src/ezpz/bin/utils.sh:1945]   - NGPUS=24
  [2025-09-11-140842][I][/home/foremans/ezpz/src/ezpz/bin/utils.sh:1947] [LAUNCH]
  [2025-09-11-140842][I][/home/foremans/ezpz/src/ezpz/bin/utils.sh:1948]   - To launch across all available GPUs, use: 'launch'
  [2025-09-11-140842][I][/home/foremans/ezpz/src/ezpz/bin/utils.sh:1949]     launch = mpiexec --verbose --envall -n 24 -ppn 12 --hostfile /var/spool/pbs/aux/7559761.aurora-pbs-0001.hostmgmt.cm.aurora.alcf.anl.gov --no-vni --cpu-bind=verbose,list:2-4:10-12:18-20:26-28:34-36:42-44:54-56:62-64:70-72:78-8
  0:86-88:94-96
  [2025-09-11-140842][I][/home/foremans/ezpz/src/ezpz/bin/utils.sh:1950]   - Run 'which launch' to ensure that the alias is set correctly
  [2025-09-11-140842][I][/home/foremans/ezpz/src/ezpz/bin/utils.sh:2495] [✓] Finished [ezpz_setup_env]
  took: 0h:00m:04s
  ```

  </details>

- Install dependencies:

  ```bash
  # uv not required, but useful!
  # to download: curl -LsSf https://astral.sh/uv/install.sh | sh
  uv pip install "git+https://github.com/saforem2/ezpz"
  # from my fork until PR #2 merged in zhenghh04/blendcorpus
  python3 -m pip install "git+https://github.com/saforem2/blendcorpus/tree/saforem2/reorg-imports"
  # from inside auroraGPT-ANL/torchtitan @ saforem2/blendcorpus
  python3 -m pip install -e "."
  ```

- Launch:

  ```bash
  ; ezpz-launch -m torchtitan.experiments.blendcorpus.train --job.config_file torchtitan/experiments/blendcorpus/train_configs/llama2_7b.toml | tee tt-bc-$(tstamp).log
  ```

  <details closed><summary><code>output</code>:</summary>

  ```bash
  #[🐍 aurora_nre_models_frameworks-2025.2.0](👻 torchtitan-aurora_nre_models_frameworks-2025.2.0)
  #[/f/A/A/E/A/t/a/torchtitan][🌱 saforem2/blendcorpus][📝🤷✓] [⏱️ 3s]
  #[09/11/25 @ 14:08:45][x4112c1s0b0n0]
  [W911 14:08:52.571468379 OperatorEntry.cpp:218] Warning: Warning only once for all operators,  other operators may also be overridden.
    Overriding a previously registered kernel for the same operator and the same dispatch key
    operator: aten::geometric_(Tensor(a!) self, float p, *, Generator? generator=None) -> Tensor(a!)
      registered at /lus/tegu/projects/datasets/software/wheelforge/repositories/pytorch_2p8_rel_07_18_2025/pytorch/build/aten/src/ATen/RegisterSchema.cpp:6
    dispatch key: XPU
    previous kernel: registered at /lus/tegu/projects/datasets/software/wheelforge/repositories/pytorch_2p8_rel_07_18_2025/pytorch/aten/src/ATen/VmapModeRegistrations.cpp:37
         new kernel: registered at /lus/tegu/projects/datasets/software/wheelforge/repositories/ipex_2.8.10_xpu_rel_08_18_2025/intel-extension-for-pytorch/build/Release/csrc/gpu/csrc/gpu/xpu/ATen/RegisterXPU_0.cpp:172 (function operator())
  [2025-09-11 14:08:53,284] [INFO] [real_accelerator.py:260:get_accelerator] Setting ds_accelerator to xpu (auto detect)
  [2025-09-11 14:08:57,635] [INFO] [logging.py:107:log_dist] [Rank -1] [TorchCheckpointEngine] Initialized with serialization = False
  [2025-09-11 14:08:59,509553][I][ezpz/__init__:266:<module>] Setting logging level to 'INFO' on 'RANK == 0'
  [2025-09-11 14:08:59,511925][I][ezpz/__init__:267:<module>] Setting logging level to 'CRITICAL' on all others 'RANK != 0'


  [2025-09-11 14:08:59,517265][I][ezpz/launch:340:launch] ----[🍋 ezpz.launch][started][2025-09-11-140859]----
  [2025-09-11 14:09:00,864914][I][ezpz/launch:345:launch] Job ID: 7559761
  [2025-09-11 14:09:00,865931][I][ezpz/launch:346:launch] nodelist: ['x4112c1s0b0n0', 'x4112c1s1b0n0']
  [2025-09-11 14:09:00,866335][I][ezpz/launch:347:launch] hostfile: /var/spool/pbs/aux/7559761.aurora-pbs-0001.hostmgmt.cm.aurora.alcf.anl.gov
  [2025-09-11 14:09:00,867635][I][ezpz/pbs:188:get_pbs_launch_cmd] ✅ Using [24/24] GPUs [2 hosts] x [12 GPU/host]
  [2025-09-11 14:09:00,868845][I][ezpz/launch:316:build_executable] Building command to execute by piecing together:
  [2025-09-11 14:09:00,869280][I][ezpz/launch:317:build_executable] (1.) launch_cmd: mpiexec --verbose --envall --np=24 --ppn=12 --hostfile=/var/spool/pbs/aux/7559761.aurora-pbs-0001.hostmgmt.cm.aurora.alcf.anl.gov --no-vni --cpu-bind=verbose,list:2-4:10-12:18-20:26-28:34-36:42-44:54-56:62-64:70-72:78
  -80:86-88:94-96
  [2025-09-11 14:09:00,869994][I][ezpz/launch:318:build_executable] (2.) cmd_to_launch: /lus/flare/projects/AuroraGPT/AuroraGPT-v1/Experiments/AuroraGPT-2B/tt/auroraGPT-ANL/torchtitan/venvs/aurora/torchtitan-aurora_nre_models_frameworks-2025.2.0/bin/python3 -m torchtitan.experiments.blendcorpus.train
  --job.config_file torchtitan/experiments/blendcorpus/train_configs/llama2_7b.toml
  [2025-09-11 14:09:00,871005][I][ezpz/launch:360:launch] Took: 1.35 seconds to build command.
  [2025-09-11 14:09:00,871369][I][ezpz/launch:363:launch] Executing:
  mpiexec
    --verbose
    --envall
    --np=24
    --ppn=12
    --hostfile=/var/spool/pbs/aux/7559761.aurora-pbs-0001.hostmgmt.cm.aurora.alcf.anl.gov
    --no-vni
    --cpu-bind=verbose,list:2-4:10-12:18-20:26-28:34-36:42-44:54-56:62-64:70-72:78-80:86-88:94-96
    /lus/flare/projects/AuroraGPT/AuroraGPT-v1/Experiments/AuroraGPT-2B/tt/auroraGPT-ANL/torchtitan/venvs/aurora/torchtitan-aurora_nre_models_frameworks-2025.2.0/bin/python3
    -m
    torchtitan.experiments.blendcorpus.train
    --job.config_file
    torchtitan/experiments/blendcorpus/train_configs/llama2_7b.toml
  [2025-09-11 14:09:00,872950][I][ezpz/launch:179:get_aurora_filters] Filtering for Aurora-specific messages. To view list of filters, run with EZPZ_LOG_LEVEL=DEBUG
  [2025-09-11 14:09:00,873469][I][ezpz/launch:370:launch] Execution started @ 2025-09-11-140900...
  [2025-09-11 14:09:00,873924][I][ezpz/launch:371:launch] ----[🍋 ezpz.launch][stop][2025-09-11-140900]----
  [2025-09-11 14:09:00,874383][I][ezpz/launch:99:run_command] Caught 24 filters
  [2025-09-11 14:09:00,874729][I][ezpz/launch:100:run_command] Running command:
   mpiexec --verbose --envall --np=24 --ppn=12 --hostfile=/var/spool/pbs/aux/7559761.aurora-pbs-0001.hostmgmt.cm.aurora.alcf.anl.gov --no-vni --cpu-bind=verbose,list:2-4:10-12:18-20:26-28:34-36:42-44:54-56:62-64:70-72:78-80:86-88:94-96 /lus/flare/projects/AuroraGPT/AuroraGPT-v1/Experiments/AuroraGPT-2
  B/tt/auroraGPT-ANL/torchtitan/venvs/aurora/torchtitan-aurora_nre_models_frameworks-2025.2.0/bin/python3 -m torchtitan.experiments.blendcorpus.train --job.config_file torchtitan/experiments/blendcorpus/train_configs/llama2_7b.toml
  Disabling local launch: multi-node application
  Connected to tcp://x4112c1s0b0n0.hsn.cm.aurora.alcf.anl.gov:7919
  Launching application e5b5c28b-462e-4fcb-8498-357b81b06d4b
  cpubind:list x4112c1s1b0n0 pid 132744 rank 12 0: mask 0x1c
  cpubind:list x4112c1s1b0n0 pid 132745 rank 13 1: mask 0x1c00
  cpubind:list x4112c1s1b0n0 pid 132746 rank 14 2: mask 0x1c0000
  cpubind:list x4112c1s1b0n0 pid 132747 rank 15 3: mask 0x1c000000
  cpubind:list x4112c1s1b0n0 pid 132748 rank 16 4: mask 0x1c00000000
  cpubind:list x4112c1s1b0n0 pid 132749 rank 17 5: mask 0x1c0000000000
  cpubind:list x4112c1s1b0n0 pid 132750 rank 18 6: mask 0x1c0000000000000
  cpubind:list x4112c1s1b0n0 pid 132751 rank 19 7: mask 0x1c000000000000000
  cpubind:list x4112c1s1b0n0 pid 132752 rank 20 8: mask 0x1c00000000000000000
  cpubind:list x4112c1s1b0n0 pid 132753 rank 21 9: mask 0x1c0000000000000000000
  cpubind:list x4112c1s1b0n0 pid 132754 rank 22 10: mask 0x1c000000000000000000000
  cpubind:list x4112c1s1b0n0 pid 132755 rank 23 11: mask 0x1c00000000000000000000000
  Application e5b5c28b-462e-4fcb-8498-357b81b06d4b started execution
  cpubind:list x4112c1s0b0n0 pid 169700 rank 0 0: mask 0x1c
  cpubind:list x4112c1s0b0n0 pid 169701 rank 1 1: mask 0x1c00
  cpubind:list x4112c1s0b0n0 pid 169702 rank 2 2: mask 0x1c0000
  cpubind:list x4112c1s0b0n0 pid 169703 rank 3 3: mask 0x1c000000
  cpubind:list x4112c1s0b0n0 pid 169704 rank 4 4: mask 0x1c00000000
  cpubind:list x4112c1s0b0n0 pid 169705 rank 5 5: mask 0x1c0000000000
  cpubind:list x4112c1s0b0n0 pid 169706 rank 6 6: mask 0x1c0000000000000
  cpubind:list x4112c1s0b0n0 pid 169707 rank 7 7: mask 0x1c000000000000000
  cpubind:list x4112c1s0b0n0 pid 169708 rank 8 8: mask 0x1c00000000000000000
  cpubind:list x4112c1s0b0n0 pid 169709 rank 9 9: mask 0x1c0000000000000000000
  cpubind:list x4112c1s0b0n0 pid 169710 rank 10 10: mask 0x1c000000000000000000000
  cpubind:list x4112c1s0b0n0 pid 169711 rank 11 11: mask 0x1c00000000000000000000000
    operator: aten::geometric_(Tensor(a!) self, float p, *, Generator? generator=None) -> Tensor(a!)
      registered at /lus/tegu/projects/datasets/software/wheelforge/repositories/pytorch_2p8_rel_07_18_2025/pytorch/build/aten/src/ATen/RegisterSchema.cpp:6
    operator: aten::geometric_(Tensor(a!) self, float p, *, Generator? generator=None) -> Tensor(a!)
      registered at /lus/tegu/projects/datasets/software/wheelforge/repositories/pytorch_2p8_rel_07_18_2025/pytorch/build/aten/src/ATen/RegisterSchema.cpp:6
    operator: aten::geometric_(Tensor(a!) self, float p, *, Generator? generator=None) -> Tensor(a!)
      registered at /lus/tegu/projects/datasets/software/wheelforge/repositories/pytorch_2p8_rel_07_18_2025/pytorch/build/aten/src/ATen/RegisterSchema.cpp:6
    operator: aten::geometric_(Tensor(a!) self, float p, *, Generator? generator=None) -> Tensor(a!)
      registered at /lus/tegu/projects/datasets/software/wheelforge/repositories/pytorch_2p8_rel_07_18_2025/pytorch/build/aten/src/ATen/RegisterSchema.cpp:6
    operator: aten::geometric_(Tensor(a!) self, float p, *, Generator? generator=None) -> Tensor(a!)
      registered at /lus/tegu/projects/datasets/software/wheelforge/repositories/pytorch_2p8_rel_07_18_2025/pytorch/build/aten/src/ATen/RegisterSchema.cpp:6
    operator: aten::geometric_(Tensor(a!) self, float p, *, Generator? generator=None) -> Tensor(a!)
      registered at /lus/tegu/projects/datasets/software/wheelforge/repositories/pytorch_2p8_rel_07_18_2025/pytorch/build/aten/src/ATen/RegisterSchema.cpp:6
    operator: aten::geometric_(Tensor(a!) self, float p, *, Generator? generator=None) -> Tensor(a!)
      registered at /lus/tegu/projects/datasets/software/wheelforge/repositories/pytorch_2p8_rel_07_18_2025/pytorch/build/aten/src/ATen/RegisterSchema.cpp:6
    operator: aten::geometric_(Tensor(a!) self, float p, *, Generator? generator=None) -> Tensor(a!)
      registered at /lus/tegu/projects/datasets/software/wheelforge/repositories/pytorch_2p8_rel_07_18_2025/pytorch/build/aten/src/ATen/RegisterSchema.cpp:6
    operator: aten::geometric_(Tensor(a!) self, float p, *, Generator? generator=None) -> Tensor(a!)
      registered at /lus/tegu/projects/datasets/software/wheelforge/repositories/pytorch_2p8_rel_07_18_2025/pytorch/build/aten/src/ATen/RegisterSchema.cpp:6
    operator: aten::geometric_(Tensor(a!) self, float p, *, Generator? generator=None) -> Tensor(a!)
      registered at /lus/tegu/projects/datasets/software/wheelforge/repositories/pytorch_2p8_rel_07_18_2025/pytorch/build/aten/src/ATen/RegisterSchema.cpp:6
    operator: aten::geometric_(Tensor(a!) self, float p, *, Generator? generator=None) -> Tensor(a!)
      registered at /lus/tegu/projects/datasets/software/wheelforge/repositories/pytorch_2p8_rel_07_18_2025/pytorch/build/aten/src/ATen/RegisterSchema.cpp:6
    operator: aten::geometric_(Tensor(a!) self, float p, *, Generator? generator=None) -> Tensor(a!)
      registered at /lus/tegu/projects/datasets/software/wheelforge/repositories/pytorch_2p8_rel_07_18_2025/pytorch/build/aten/src/ATen/RegisterSchema.cpp:6
    operator: aten::geometric_(Tensor(a!) self, float p, *, Generator? generator=None) -> Tensor(a!)
      registered at /lus/tegu/projects/datasets/software/wheelforge/repositories/pytorch_2p8_rel_07_18_2025/pytorch/build/aten/src/ATen/RegisterSchema.cpp:6
    operator: aten::geometric_(Tensor(a!) self, float p, *, Generator? generator=None) -> Tensor(a!)
      registered at /lus/tegu/projects/datasets/software/wheelforge/repositories/pytorch_2p8_rel_07_18_2025/pytorch/build/aten/src/ATen/RegisterSchema.cpp:6
    operator: aten::geometric_(Tensor(a!) self, float p, *, Generator? generator=None) -> Tensor(a!)
      registered at /lus/tegu/projects/datasets/software/wheelforge/repositories/pytorch_2p8_rel_07_18_2025/pytorch/build/aten/src/ATen/RegisterSchema.cpp:6
    operator: aten::geometric_(Tensor(a!) self, float p, *, Generator? generator=None) -> Tensor(a!)
      registered at /lus/tegu/projects/datasets/software/wheelforge/repositories/pytorch_2p8_rel_07_18_2025/pytorch/build/aten/src/ATen/RegisterSchema.cpp:6
    operator: aten::geometric_(Tensor(a!) self, float p, *, Generator? generator=None) -> Tensor(a!)
      registered at /lus/tegu/projects/datasets/software/wheelforge/repositories/pytorch_2p8_rel_07_18_2025/pytorch/build/aten/src/ATen/RegisterSchema.cpp:6
    operator: aten::geometric_(Tensor(a!) self, float p, *, Generator? generator=None) -> Tensor(a!)
      registered at /lus/tegu/projects/datasets/software/wheelforge/repositories/pytorch_2p8_rel_07_18_2025/pytorch/build/aten/src/ATen/RegisterSchema.cpp:6
    operator: aten::geometric_(Tensor(a!) self, float p, *, Generator? generator=None) -> Tensor(a!)
      registered at /lus/tegu/projects/datasets/software/wheelforge/repositories/pytorch_2p8_rel_07_18_2025/pytorch/build/aten/src/ATen/RegisterSchema.cpp:6
    operator: aten::geometric_(Tensor(a!) self, float p, *, Generator? generator=None) -> Tensor(a!)
      registered at /lus/tegu/projects/datasets/software/wheelforge/repositories/pytorch_2p8_rel_07_18_2025/pytorch/build/aten/src/ATen/RegisterSchema.cpp:6
    operator: aten::geometric_(Tensor(a!) self, float p, *, Generator? generator=None) -> Tensor(a!)
      registered at /lus/tegu/projects/datasets/software/wheelforge/repositories/pytorch_2p8_rel_07_18_2025/pytorch/build/aten/src/ATen/RegisterSchema.cpp:6
    operator: aten::geometric_(Tensor(a!) self, float p, *, Generator? generator=None) -> Tensor(a!)
      registered at /lus/tegu/projects/datasets/software/wheelforge/repositories/pytorch_2p8_rel_07_18_2025/pytorch/build/aten/src/ATen/RegisterSchema.cpp:6
    operator: aten::geometric_(Tensor(a!) self, float p, *, Generator? generator=None) -> Tensor(a!)
      registered at /lus/tegu/projects/datasets/software/wheelforge/repositories/pytorch_2p8_rel_07_18_2025/pytorch/build/aten/src/ATen/RegisterSchema.cpp:6
    operator: aten::geometric_(Tensor(a!) self, float p, *, Generator? generator=None) -> Tensor(a!)
      registered at /lus/tegu/projects/datasets/software/wheelforge/repositories/pytorch_2p8_rel_07_18_2025/pytorch/build/aten/src/ATen/RegisterSchema.cpp:6
  [2025-09-11 14:09:15,351212][I][ezpz/__init__:266:<module>] Setting logging level to 'INFO' on 'RANK == 0'
  [2025-09-11 14:09:15,353793][I][ezpz/__init__:267:<module>] Setting logging level to 'CRITICAL' on all others 'RANK != 0'
  [2025-09-11 14:09:16,104864][I][ezpz/dist:1181:setup_torch_distributed] Using fw='ddp' with torch_{device,backend}= {xpu, xccl}
  [2025-09-11 14:09:16,105881][I][ezpz/dist:1039:setup_torch_DDP] Caught MASTER_PORT=57059 from environment!
  [2025-09-11 14:09:16,106685][I][ezpz/dist:1055:setup_torch_DDP] Using torch.distributed.init_process_group with
  - master_addr='x4112c1s0b0n0.hsn.cm.aurora.alcf.anl.gov'
  - master_port='57059'
  - world_size=24
  - rank=0
  - local_rank=0
  - timeout=datetime.timedelta(seconds=3600)
  - backend='xccl'
  [2025-09-11 14:09:16,107715][I][ezpz/dist:772:init_process_group] Calling torch.distributed.init_process_group_with: rank=0 world_size=24 backend=xccl
  [2025-09-11 14:09:17,262727][I][ezpz/pbs:188:get_pbs_launch_cmd] ✅ Using [24/24] GPUs [2 hosts] x [12 GPU/host]
  [2025-09-11 14:09:17,264645][I][ezpz/dist:450:print_dist_setup] [device='xpu'][rank=0/23][local_rank=0/11][node=0/1]
  [2025-09-11 14:09:17,265339][W][utils/_logger:68:warning] Using [24 / 24] available "xpu" devices !!
  2025:09:11-14:09:17:(169700) |CCL_WARN| value of CCL_LOG_LEVEL changed to be error (default:warn)
  [2025-09-11 14:09:18,089625][I][ezpz/dist:1401:setup_torch] Using device='xpu' with backend='xccl' + 'xccl' for distributed training.
  [2025-09-11 14:09:18,090618][I][ezpz/dist:1448:setup_torch] ['x4112c1s0b0n0'][ 0/23]
  [2025-09-11 14:09:18,089411][I][ezpz/dist:1448:setup_torch] ['x4112c1s0b0n0'][ 1/23]
  [2025-09-11 14:09:18,089411][I][ezpz/dist:1448:setup_torch] ['x4112c1s0b0n0'][ 4/23]
  [2025-09-11 14:09:18,089466][I][ezpz/dist:1448:setup_torch] ['x4112c1s0b0n0'][ 2/23]
  [2025-09-11 14:09:18,089413][I][ezpz/dist:1448:setup_torch] ['x4112c1s0b0n0'][ 3/23]
  [2025-09-11 14:09:18,089412][I][ezpz/dist:1448:setup_torch] ['x4112c1s0b0n0'][ 5/23]
  [2025-09-11 14:09:18,089309][I][ezpz/dist:1448:setup_torch] ['x4112c1s1b0n0'][13/23]
  [2025-09-11 14:09:18,089309][I][ezpz/dist:1448:setup_torch] ['x4112c1s1b0n0'][14/23]
  [2025-09-11 14:09:18,089475][I][ezpz/dist:1448:setup_torch] ['x4112c1s0b0n0'][ 6/23]
  [2025-09-11 14:09:18,089349][I][ezpz/dist:1448:setup_torch] ['x4112c1s0b0n0'][ 7/23]
  [2025-09-11 14:09:18,089350][I][ezpz/dist:1448:setup_torch] ['x4112c1s0b0n0'][ 8/23]
  [2025-09-11 14:09:18,089411][I][ezpz/dist:1448:setup_torch] ['x4112c1s0b0n0'][ 9/23]
  [2025-09-11 14:09:18,089349][I][ezpz/dist:1448:setup_torch] ['x4112c1s0b0n0'][10/23]
  [2025-09-11 14:09:18,089469][I][ezpz/dist:1448:setup_torch] ['x4112c1s0b0n0'][11/23]
  [2025-09-11 14:09:18,089328][I][ezpz/dist:1448:setup_torch] ['x4112c1s1b0n0'][15/23]
  [2025-09-11 14:09:18,089309][I][ezpz/dist:1448:setup_torch] ['x4112c1s1b0n0'][16/23]
  [2025-09-11 14:09:18,089216][I][ezpz/dist:1448:setup_torch] ['x4112c1s1b0n0'][18/23]
  [2025-09-11 14:09:18,089330][I][ezpz/dist:1448:setup_torch] ['x4112c1s1b0n0'][19/23]
  [2025-09-11 14:09:18,089333][I][ezpz/dist:1448:setup_torch] ['x4112c1s1b0n0'][20/23]
  [2025-09-11 14:09:18,089338][I][ezpz/dist:1448:setup_torch] ['x4112c1s1b0n0'][21/23]
  [2025-09-11 14:09:18,089329][I][ezpz/dist:1448:setup_torch] ['x4112c1s1b0n0'][23/23]
  [2025-09-11 14:09:18,089445][I][ezpz/dist:1448:setup_torch] ['x4112c1s1b0n0'][12/23]
  [2025-09-11 14:09:18,089370][I][ezpz/dist:1448:setup_torch] ['x4112c1s1b0n0'][17/23]
  [2025-09-11 14:09:18,089351][I][ezpz/dist:1448:setup_torch] ['x4112c1s1b0n0'][22/23]
  [2025-09-11 14:09:18,361340][I][blendcorpus/train:81:__init__] Starting job: Llama 3 debug training
  Number of ranks per node: 12
  Is initialized already
  [2025-09-11 14:09:18,368957][I][distributed/parallel_dims:158:_build_mesh_without_ep] Building 1-D device mesh with ['dp_shard'], [24]
  [2025-09-11 14:09:18,370523][I][tools/utils:63:collect] [GC] Initial GC collection 0.00 seconds
  [Tokenizer] Using backend: sptoken (SentencePiece)
  [SPTokenizer] Using model path: ./assets/hf/Llama-2-7b
  [Tokenizer] Using backend: sptoken (SentencePiece)
  [SPTokenizer] Using model path: ./assets/hf/Llama-2-7b
  [Tokenizer] Using backend: sptoken (SentencePiece)
  [SPTokenizer] Using model path: ./assets/hf/Llama-2-7b
  [Tokenizer] Using backend: sptoken (SentencePiece)
  [SPTokenizer] Using model path: ./assets/hf/Llama-2-7b
  [Tokenizer] Using backend: sptoken (SentencePiece)
  [SPTokenizer] Using model path: ./assets/hf/Llama-2-7b
  [Tokenizer] Using backend: sptoken (SentencePiece)
  [SPTokenizer] Using model path: ./assets/hf/Llama-2-7b
  [Tokenizer] Using backend: sptoken (SentencePiece)
  [SPTokenizer] Using model path: ./assets/hf/Llama-2-7b
  [Tokenizer] Using backend: sptoken (SentencePiece)
  [SPTokenizer] Using model path: ./assets/hf/Llama-2-7b
  [Tokenizer] Using backend: sptoken (SentencePiece)
  [SPTokenizer] Using model path: ./assets/hf/Llama-2-7b
  [Tokenizer] Using backend: sptoken (SentencePiece)
  [SPTokenizer] Using model path: ./assets/hf/Llama-2-7b
  [Tokenizer] Using backend: sptoken (SentencePiece)
  [SPTokenizer] Using model path: ./assets/hf/Llama-2-7b
  [Tokenizer] Using backend: sptoken (SentencePiece)
  [SPTokenizer] Using model path: ./assets/hf/Llama-2-7b
  [Tokenizer] Using backend: sptoken (SentencePiece)
  [SPTokenizer] Using model path: ./assets/hf/Llama-2-7b
  [Tokenizer] Using backend: sptoken (SentencePiece)
  [SPTokenizer] Using model path: ./assets/hf/Llama-2-7b
  [Tokenizer] Using backend: sptoken (SentencePiece)
  [SPTokenizer] Using model path: ./assets/hf/Llama-2-7b
  [Tokenizer] Using backend: sptoken (SentencePiece)
  [SPTokenizer] Using model path: ./assets/hf/Llama-2-7b
  [Tokenizer] Using backend: sptoken (SentencePiece)
  [SPTokenizer] Using model path: ./assets/hf/Llama-2-7b
  [Tokenizer] Using backend: sptoken (SentencePiece)
  [SPTokenizer] Using model path: ./assets/hf/Llama-2-7b
  [Tokenizer] Using backend: sptoken (SentencePiece)
  [SPTokenizer] Using model path: ./assets/hf/Llama-2-7b
  [Tokenizer] Using backend: sptoken (SentencePiece)
  [Tokenizer] Using backend: sptoken (SentencePiece)
  [Tokenizer] Using backend: sptoken (SentencePiece)
  [SPTokenizer] Using model path: ./assets/hf/Llama-2-7b
  [SPTokenizer] Using model path: ./assets/hf/Llama-2-7b
  [SPTokenizer] Using model path: ./assets/hf/Llama-2-7b
  [Tokenizer] Using backend: sptoken (SentencePiece)
  [Tokenizer] Using backend: sptoken (SentencePiece)
  [SPTokenizer] Using model path: ./assets/hf/Llama-2-7b
  [SPTokenizer] Using model path: ./assets/hf/Llama-2-7b
  [SPTokenizer] Loaded model: ./assets/hf/Llama-2-7b/tokenizer.model, vocab size: 32000
  [SPTokenizer] Loaded model: ./assets/hf/Llama-2-7b/tokenizer.model, vocab size: 32000
  [SPTokenizer] Loaded model: ./assets/hf/Llama-2-7b/tokenizer.model, vocab size: 32000
   [INFO][2025-09-11 14:09:18.390018] Reading data from /flare/Aurora_deployment/AuroraGPT/datasets/dolma/dolma_v1_7_file_list_mini.txt
   [INFO][2025-09-11 14:09:18.390243] Number of datasets: 9
   [INFO][2025-09-11 14:09:18.390377] Global batch size: 24
  [SPTokenizer] Loaded model: ./assets/hf/Llama-2-7b/tokenizer.model, vocab size: 32000
   [INFO][2025-09-11 14:09:18.390498] Training iterations: 1000
   [INFO][2025-09-11 14:09:18.390618] Evaluation iterations: 0
   [INFO][2025-09-11 14:09:18.390734] Total number of training samples: 24000
   [INFO][2025-09-11 14:09:18.390861] Total number of evaluation samples: 0
   [INFO][2025-09-11 14:09:18.390977] Total number of testing samples: 0
  [SPTokenizer] Loaded model: ./assets/hf/Llama-2-7b/tokenizer.model, vocab size: 32000
  [SPTokenizer] Loaded model: ./assets/hf/Llama-2-7b/tokenizer.model, vocab size: 32000
  [SPTokenizer] Loaded model: ./assets/hf/Llama-2-7b/tokenizer.model, vocab size: 32000
  [SPTokenizer] Loaded model: ./assets/hf/Llama-2-7b/tokenizer.model, vocab size: 32000
  [SPTokenizer] Loaded model: ./assets/hf/Llama-2-7b/tokenizer.model, vocab size: 32000
  [SPTokenizer] Loaded model: ./assets/hf/Llama-2-7b/tokenizer.model, vocab size: 32000
  [SPTokenizer] Loaded model: ./assets/hf/Llama-2-7b/tokenizer.model, vocab size: 32000
  [SPTokenizer] Loaded model: ./assets/hf/Llama-2-7b/tokenizer.model, vocab size: 32000
  [SPTokenizer] Loaded model: ./assets/hf/Llama-2-7b/tokenizer.model, vocab size: 32000
  [SPTokenizer] Loaded model: ./assets/hf/Llama-2-7b/tokenizer.model, vocab size: 32000
  [SPTokenizer] Loaded model: ./assets/hf/Llama-2-7b/tokenizer.model, vocab size: 32000
  [2025-09-11 14:09:18,394736][I][data/gpt_dataset:242:__init__] [BuildCorpusDataset] Caught args.shuffle_sample_in_corpus=True across 19984 samples
  [SPTokenizer] Loaded model: ./assets/hf/Llama-2-7b/tokenizer.model, vocab size: 32000
  [SPTokenizer] Loaded model: ./assets/hf/Llama-2-7b/tokenizer.model, vocab size: 32000
  [SPTokenizer] Loaded model: ./assets/hf/Llama-2-7b/tokenizer.model, vocab size: 32000
  [2025-09-11 14:09:18,395925][I][data/gpt_dataset:242:__init__] [BuildCorpusDataset] Caught args.shuffle_sample_in_corpus=True across 4140 samples
  [SPTokenizer] Loaded model: ./assets/hf/Llama-2-7b/tokenizer.model, vocab size: 32000
  [SPTokenizer] Loaded model: ./assets/hf/Llama-2-7b/tokenizer.model, vocab size: 32000
  [SPTokenizer] Loaded model: ./assets/hf/Llama-2-7b/tokenizer.model, vocab size: 32000
  [SPTokenizer] Loaded model: ./assets/hf/Llama-2-7b/tokenizer.model, vocab size: 32000
  [2025-09-11 14:09:18,397729][I][data/gpt_dataset:242:__init__] [BuildCorpusDataset] Caught args.shuffle_sample_in_corpus=True across 19984 samples
  [2025-09-11 14:09:18,398015][I][data/gpt_dataset:242:__init__] [BuildCorpusDataset] Caught args.shuffle_sample_in_corpus=True across 19984 samples
  [2025-09-11 14:09:18,398146][I][data/gpt_dataset:242:__init__] [BuildCorpusDataset] Caught args.shuffle_sample_in_corpus=True across 19984 samples
  [2025-09-11 14:09:18,398854][I][data/gpt_dataset:242:__init__] [BuildCorpusDataset] Caught args.shuffle_sample_in_corpus=True across 4140 samples
  [SPTokenizer] Loaded model: ./assets/hf/Llama-2-7b/tokenizer.model, vocab size: 32000
  [2025-09-11 14:09:18,399129][I][data/gpt_dataset:242:__init__] [BuildCorpusDataset] Caught args.shuffle_sample_in_corpus=True across 4140 samples
  [2025-09-11 14:09:18,399188][I][data/gpt_dataset:242:__init__] [BuildCorpusDataset] Caught args.shuffle_sample_in_corpus=True across 4140 samples
  [2025-09-11 14:09:18,398998][I][data/gpt_dataset:242:__init__] [BuildCorpusDataset] Caught args.shuffle_sample_in_corpus=True across 19984 samples
  [SPTokenizer] Loaded model: ./assets/hf/Llama-2-7b/tokenizer.model, vocab size: 32000
  [2025-09-11 14:09:18,399209][I][data/gpt_dataset:242:__init__] [BuildCorpusDataset] Caught args.shuffle_sample_in_corpus=True across 19984 samples
  [2025-09-11 14:09:18,399543][I][data/gpt_dataset:242:__init__] [BuildCorpusDataset] Caught args.shuffle_sample_in_corpus=True across 19984 samples
  [2025-09-11 14:09:18,399664][I][data/gpt_dataset:242:__init__] [BuildCorpusDataset] Caught args.shuffle_sample_in_corpus=True across 19984 samples
  [2025-09-11 14:09:18,400017][I][data/gpt_dataset:242:__init__] [BuildCorpusDataset] Caught args.shuffle_sample_in_corpus=True across 4140 samples
  [2025-09-11 14:09:18,400082][I][data/gpt_dataset:242:__init__] [BuildCorpusDataset] Caught args.shuffle_sample_in_corpus=True across 19984 samples
  [2025-09-11 14:09:18,400294][I][data/gpt_dataset:242:__init__] [BuildCorpusDataset] Caught args.shuffle_sample_in_corpus=True across 4140 samples
  [2025-09-11 14:09:18,400464][I][data/gpt_dataset:242:__init__] [BuildCorpusDataset] Caught args.shuffle_sample_in_corpus=True across 19984 samples
  [2025-09-11 14:09:18,400867][I][data/gpt_dataset:242:__init__] [BuildCorpusDataset] Caught args.shuffle_sample_in_corpus=True across 4140 samples
  [2025-09-11 14:09:18,400938][I][data/gpt_dataset:242:__init__] [BuildCorpusDataset] Caught args.shuffle_sample_in_corpus=True across 4140 samples
  [2025-09-11 14:09:18,401188][I][data/gpt_dataset:242:__init__] [BuildCorpusDataset] Caught args.shuffle_sample_in_corpus=True across 4140 samples
  [2025-09-11 14:09:18,401087][I][data/gpt_dataset:242:__init__] [BuildCorpusDataset] Caught args.shuffle_sample_in_corpus=True across 19984 samples
  [2025-09-11 14:09:18,401783][I][data/gpt_dataset:242:__init__] [BuildCorpusDataset] Caught args.shuffle_sample_in_corpus=True across 4140 samples
  [2025-09-11 14:09:18,402202][I][data/gpt_dataset:242:__init__] [BuildCorpusDataset] Caught args.shuffle_sample_in_corpus=True across 4140 samples
  [2025-09-11 14:09:18,402008][I][data/gpt_dataset:242:__init__] [BuildCorpusDataset] Caught args.shuffle_sample_in_corpus=True across 19984 samples
  [2025-09-11 14:09:18,402584][I][data/gpt_dataset:242:__init__] [BuildCorpusDataset] Caught args.shuffle_sample_in_corpus=True across 19984 samples
  [2025-09-11 14:09:18,403101][I][data/gpt_dataset:242:__init__] [BuildCorpusDataset] Caught args.shuffle_sample_in_corpus=True across 4140 samples
  [2025-09-11 14:09:18,402939][I][data/gpt_dataset:242:__init__] [BuildCorpusDataset] Caught args.shuffle_sample_in_corpus=True across 19984 samples
  [2025-09-11 14:09:18,403307][I][data/gpt_dataset:242:__init__] [BuildCorpusDataset] Caught args.shuffle_sample_in_corpus=True across 19984 samples
  [2025-09-11 14:09:18,403670][I][data/gpt_dataset:242:__init__] [BuildCorpusDataset] Caught args.shuffle_sample_in_corpus=True across 4140 samples
  [2025-09-11 14:09:18,403473][I][data/gpt_dataset:242:__init__] [BuildCorpusDataset] Caught args.shuffle_sample_in_corpus=True across 19984 samples
  [2025-09-11 14:09:18,404053][I][data/gpt_dataset:242:__init__] [BuildCorpusDataset] Caught args.shuffle_sample_in_corpus=True across 4140 samples
  [2025-09-11 14:09:18,404556][I][data/gpt_dataset:242:__init__] [BuildCorpusDataset] Caught args.shuffle_sample_in_corpus=True across 4140 samples
  [2025-09-11 14:09:18,404372][I][data/gpt_dataset:242:__init__] [BuildCorpusDataset] Caught args.shuffle_sample_in_corpus=True across 19984 samples
  [2025-09-11 14:09:18,404577][I][data/gpt_dataset:242:__init__] [BuildCorpusDataset] Caught args.shuffle_sample_in_corpus=True across 4140 samples
  [2025-09-11 14:09:18,404796][I][data/gpt_dataset:242:__init__] [BuildCorpusDataset] Caught args.shuffle_sample_in_corpus=True across 19984 samples
  [2025-09-11 14:09:18,405496][I][data/gpt_dataset:242:__init__] [BuildCorpusDataset] Caught args.shuffle_sample_in_corpus=True across 4140 samples
  [2025-09-11 14:09:18,406076][I][data/gpt_dataset:242:__init__] [BuildCorpusDataset] Caught args.shuffle_sample_in_corpus=True across 4140 samples
  > building indices for corpus datasets ...
  [2025-09-11 14:09:18,406862][I][data/gpt_dataset:242:__init__] [BuildCorpusDataset] Caught args.shuffle_sample_in_corpus=True across 19984 samples
  [2025-09-11 14:09:18,408007][I][data/gpt_dataset:242:__init__] [BuildCorpusDataset] Caught args.shuffle_sample_in_corpus=True across 4140 samples
  [2025-09-11 14:09:18,408060][I][data/gpt_dataset:242:__init__] [BuildCorpusDataset] Caught args.shuffle_sample_in_corpus=True across 19984 samples
  > building indices for corpus datasets ...
  [2025-09-11 14:09:18,409160][I][data/gpt_dataset:242:__init__] [BuildCorpusDataset] Caught args.shuffle_sample_in_corpus=True across 4140 samples
  [2025-09-11 14:09:18,410697][I][data/gpt_dataset:242:__init__] [BuildCorpusDataset] Caught args.shuffle_sample_in_corpus=True across 19984 samples
  [2025-09-11 14:09:18,411798][I][data/gpt_dataset:242:__init__] [BuildCorpusDataset] Caught args.shuffle_sample_in_corpus=True across 4140 samples
  [2025-09-11 14:09:18,451150][I][data/gpt_dataset:242:__init__] [BuildCorpusDataset] Caught args.shuffle_sample_in_corpus=True across 19984 samples
  [2025-09-11 14:09:18,452190][I][data/gpt_dataset:242:__init__] [BuildCorpusDataset] Caught args.shuffle_sample_in_corpus=True across 4140 samples
  [2025-09-11 14:09:18,454276][I][data/gpt_dataset:242:__init__] [BuildCorpusDataset] Caught args.shuffle_sample_in_corpus=True across 19984 samples
  [2025-09-11 14:09:18,455354][I][data/gpt_dataset:242:__init__] [BuildCorpusDataset] Caught args.shuffle_sample_in_corpus=True across 4140 samples
  [2025-09-11 14:09:18,457103][I][data/gpt_dataset:242:__init__] [BuildCorpusDataset] Caught args.shuffle_sample_in_corpus=True across 19984 samples
  [2025-09-11 14:09:18,458223][I][data/gpt_dataset:242:__init__] [BuildCorpusDataset] Caught args.shuffle_sample_in_corpus=True across 4140 samples
  > building indices for blendable datasets ...
   > sample ratios:
     dataset 0, input: 0.828387, achieved: 0.828387
     dataset 1, input: 0.171613, achieved: 0.171613
  [2025-09-11 14:09:18,993214][I][data/blendable_dataset:48:_build_indices] > elapsed time for building blendable dataset indices: 0.00 (sec)
  [2025-09-11 14:09:18,993249][I][data/blendable_dataset:48:_build_indices] > elapsed time for building blendable dataset indices: 0.00 (sec)
  [2025-09-11 14:09:18,993206][I][data/blendable_dataset:48:_build_indices] > elapsed time for building blendable dataset indices: 0.00 (sec)
  [2025-09-11 14:09:18,993241][I][data/blendable_dataset:48:_build_indices] > elapsed time for building blendable dataset indices: 0.00 (sec)
  [2025-09-11 14:09:18,993219][I][data/blendable_dataset:48:_build_indices] > elapsed time for building blendable dataset indices: 0.00 (sec)
  [2025-09-11 14:09:18,993151][I][data/blendable_dataset:48:_build_indices] > elapsed time for building blendable dataset indices: 0.00 (sec)
  [2025-09-11 14:09:18,993429][I][data/blendable_dataset:48:_build_indices] > elapsed time for building blendable dataset indices: 0.00 (sec)
  [2025-09-11 14:09:18,993386][I][data/blendable_dataset:48:_build_indices] > elapsed time for building blendable dataset indices: 0.00 (sec)
  [2025-09-11 14:09:18,993387][I][data/blendable_dataset:48:_build_indices] > elapsed time for building blendable dataset indices: 0.00 (sec)
  [2025-09-11 14:09:18,993758][I][data/blendable_dataset:48:_build_indices] > elapsed time for building blendable dataset indices: 0.00 (sec)
  [2025-09-11 14:09:18,993617][I][data/blendable_dataset:48:_build_indices] > elapsed time for building blendable dataset indices: 0.00 (sec)
  [2025-09-11 14:09:18,993380][I][data/blendable_dataset:48:_build_indices] > elapsed time for building blendable dataset indices: 0.00 (sec)
  [2025-09-11 14:09:18,994324][I][data/blendable_dataset:126:__init__] > size of blendable dataset: 24124 samples
  [2025-09-11 14:09:18,993263][I][data/blendable_dataset:48:_build_indices] > elapsed time for building blendable dataset indices: 0.00 (sec)
  [2025-09-11 14:09:18,993140][I][data/blendable_dataset:48:_build_indices] > elapsed time for building blendable dataset indices: 0.00 (sec)
  [2025-09-11 14:09:18,993191][I][data/blendable_dataset:48:_build_indices] > elapsed time for building blendable dataset indices: 0.00 (sec)
  [2025-09-11 14:09:18,993209][I][data/blendable_dataset:48:_build_indices] > elapsed time for building blendable dataset indices: 0.00 (sec)
  [2025-09-11 14:09:18,993140][I][data/blendable_dataset:48:_build_indices] > elapsed time for building blendable dataset indices: 0.00 (sec)
  [2025-09-11 14:09:18,993205][I][data/blendable_dataset:48:_build_indices] > elapsed time for building blendable dataset indices: 0.00 (sec)
  [2025-09-11 14:09:18,993193][I][data/blendable_dataset:48:_build_indices] > elapsed time for building blendable dataset indices: 0.00 (sec)
  [2025-09-11 14:09:18,993189][I][data/blendable_dataset:48:_build_indices] > elapsed time for building blendable dataset indices: 0.00 (sec)
  [2025-09-11 14:09:18,993698][I][data/blendable_dataset:48:_build_indices] > elapsed time for building blendable dataset indices: 0.00 (sec)
  [2025-09-11 14:09:18,993754][I][data/blendable_dataset:48:_build_indices] > elapsed time for building blendable dataset indices: 0.00 (sec)
  [2025-09-11 14:09:18,993790][I][data/blendable_dataset:48:_build_indices] > elapsed time for building blendable dataset indices: 0.00 (sec)
  [2025-09-11 14:09:18,993778][I][data/blendable_dataset:48:_build_indices] > elapsed time for building blendable dataset indices: 0.00 (sec)
  [2025-09-11 14:09:19,019441][I][data/blendable_dataset:126:__init__] > size of blendable dataset: 24124 samples
  [2025-09-11 14:09:19,020538][I][data/blendable_dataset:126:__init__] > size of blendable dataset: 24124 samples
  [2025-09-11 14:09:19,020696][I][data/blendable_dataset:126:__init__] > size of blendable dataset: 24124 samples
  [2025-09-11 14:09:19,020679][I][data/blendable_dataset:126:__init__] > size of blendable dataset: 24124 samples
  [2025-09-11 14:09:19,021266][I][data/blendable_dataset:126:__init__] > size of blendable dataset: 24124 samples
  [2025-09-11 14:09:19,021538][I][data/blendable_dataset:126:__init__] > size of blendable dataset: 24124 samples
  [2025-09-11 14:09:19,021636][I][data/blendable_dataset:126:__init__] > size of blendable dataset: 24124 samples
  [2025-09-11 14:09:19,022228][I][data/blendable_dataset:126:__init__] > size of blendable dataset: 24124 samples
  [2025-09-11 14:09:19,022658][I][data/blendable_dataset:126:__init__] > size of blendable dataset: 24124 samples
  [2025-09-11 14:09:19,022708][I][data/blendable_dataset:126:__init__] > size of blendable dataset: 24124 samples
  [2025-09-11 14:09:19,022905][I][data/blendable_dataset:126:__init__] > size of blendable dataset: 24124 samples
  [2025-09-11 14:09:19,023011][I][data/blendable_dataset:126:__init__] > size of blendable dataset: 24124 samples
  [2025-09-11 14:09:19,023387][I][data/blendable_dataset:126:__init__] > size of blendable dataset: 24124 samples
  [2025-09-11 14:09:19,025873][I][data/blendable_dataset:126:__init__] > size of blendable dataset: 24124 samples
  [2025-09-11 14:09:19,027583][I][data/blendable_dataset:126:__init__] > size of blendable dataset: 24124 samples
  [2025-09-11 14:09:19,027682][I][data/blendable_dataset:126:__init__] > size of blendable dataset: 24124 samples
  [2025-09-11 14:09:19,027822][I][data/blendable_dataset:126:__init__] > size of blendable dataset: 24124 samples
  [2025-09-11 14:09:19,028637][I][data/blendable_dataset:126:__init__] > size of blendable dataset: 24124 samples
  [2025-09-11 14:09:19,029231][I][data/blendable_dataset:126:__init__] > size of blendable dataset: 24124 samples
  [2025-09-11 14:09:19,029317][I][data/blendable_dataset:126:__init__] > size of blendable dataset: 24124 samples
  [2025-09-11 14:09:19,029392][I][data/blendable_dataset:126:__init__] > size of blendable dataset: 24124 samples
  [2025-09-11 14:09:19,029927][I][data/blendable_dataset:126:__init__] > size of blendable dataset: 24124 samples
  [2025-09-11 14:09:19,030095][I][data/blendable_dataset:126:__init__] > size of blendable dataset: 24124 samples
  [2025-09-11 14:09:19,030365][I][blendcorpus/train:166:__init__] Using BlendCorpus dataloader.
  [2025-09-11 14:09:19,030855][I][blendcorpus/train:174:__init__] Building blendcorpus Llama-2-7b with TransformerModelArgs(_enforced='This field is used to enforce all fields have defaults.', dim=4096, n_layers=6, n_heads=32, n_kv_heads=None, vocab_size=32000, multiple_of=256, ffn_dim_multiplier=None
  , norm_eps=1e-05, rope_theta=500000, max_seq_len=4096, depth_init=True, use_flex_attn=False, attn_mask_type='causal', eos_id=0)
  [2025-09-11 14:09:19,079016][I][components/metrics:99:build_device_memory_monitor] XPU capacity: Intel(R) Data Center GPU Max 1550 with 63.98GiB memory
  [2025-09-11 14:09:19,553115][I][blendcorpus/train:201:__init__] Model blendcorpus Llama-2-7b size: 1,476,448,256 total parameters
  [2025-09-11 14:09:19,554318][I][infra/parallelize:345:apply_ac] Applied selective activation checkpointing to the model
  [2025-09-11 14:09:19,563233][W][protocols/state_dict_adapter:76:__init__] model.safetensors.index.json not found at hf_assets_path: ./assets/hf/Llama-2-7b/model.safetensors.index.json.                     Defaulting to saving a single safetensors file if checkpoint is saved in HF format
  [2025-09-11 14:09:19,584125][I][infra/parallelize:122:parallelize_llama] Applied FSDP to the model
  [2025-09-11 14:09:19,811279][W][protocols/state_dict_adapter:76:__init__] model.safetensors.index.json not found at hf_assets_path: ./assets/hf/Llama-2-7b/model.safetensors.index.json.                     Defaulting to saving a single safetensors file if checkpoint is saved in HF format
  [2025-09-11 14:09:19,869830][W][protocols/state_dict_adapter:76:__init__] model.safetensors.index.json not found at hf_assets_path: ./assets/hf/Llama-2-7b/model.safetensors.index.json.                     Defaulting to saving a single safetensors file if checkpoint is saved in HF format
  [2025-09-11 14:09:19,870851][W][protocols/state_dict_adapter:76:__init__] model.safetensors.index.json not found at hf_assets_path: ./assets/hf/Llama-2-7b/model.safetensors.index.json.                     Defaulting to saving a single safetensors file if checkpoint is saved in HF format
  [2025-09-11 14:09:19,882402][W][protocols/state_dict_adapter:76:__init__] model.safetensors.index.json not found at hf_assets_path: ./assets/hf/Llama-2-7b/model.safetensors.index.json.                     Defaulting to saving a single safetensors file if checkpoint is saved in HF format
  [2025-09-11 14:09:19,966755][W][protocols/state_dict_adapter:76:__init__] model.safetensors.index.json not found at hf_assets_path: ./assets/hf/Llama-2-7b/model.safetensors.index.json.                     Defaulting to saving a single safetensors file if checkpoint is saved in HF format
  [2025-09-11 14:09:20,101624][W][protocols/state_dict_adapter:76:__init__] model.safetensors.index.json not found at hf_assets_path: ./assets/hf/Llama-2-7b/model.safetensors.index.json.                     Defaulting to saving a single safetensors file if checkpoint is saved in HF format
  [2025-09-11 14:09:20,152791][W][protocols/state_dict_adapter:76:__init__] model.safetensors.index.json not found at hf_assets_path: ./assets/hf/Llama-2-7b/model.safetensors.index.json.                     Defaulting to saving a single safetensors file if checkpoint is saved in HF format
  [2025-09-11 14:09:20,207893][W][protocols/state_dict_adapter:76:__init__] model.safetensors.index.json not found at hf_assets_path: ./assets/hf/Llama-2-7b/model.safetensors.index.json.                     Defaulting to saving a single safetensors file if checkpoint is saved in HF format
  [2025-09-11 14:09:20,227251][W][protocols/state_dict_adapter:76:__init__] model.safetensors.index.json not found at hf_assets_path: ./assets/hf/Llama-2-7b/model.safetensors.index.json.                     Defaulting to saving a single safetensors file if checkpoint is saved in HF format
  [2025-09-11 14:09:20,240293][W][protocols/state_dict_adapter:76:__init__] model.safetensors.index.json not found at hf_assets_path: ./assets/hf/Llama-2-7b/model.safetensors.index.json.                     Defaulting to saving a single safetensors file if checkpoint is saved in HF format
  [2025-09-11 14:09:20,260280][W][protocols/state_dict_adapter:76:__init__] model.safetensors.index.json not found at hf_assets_path: ./assets/hf/Llama-2-7b/model.safetensors.index.json.                     Defaulting to saving a single safetensors file if checkpoint is saved in HF format
  [2025-09-11 14:09:20,260591][I][blendcorpus/train:281:__init__] Peak FLOPS used for computing MFU: 2.982e+14
  [2025-09-11 14:09:20,260475][W][protocols/state_dict_adapter:76:__init__] model.safetensors.index.json not found at hf_assets_path: ./assets/hf/Llama-2-7b/model.safetensors.index.json.                     Defaulting to saving a single safetensors file if checkpoint is saved in HF format
  [2025-09-11 14:09:20,261398][I][blendcorpus/train:283:__init__] XPU memory usage for model: 0.26GiB(0.41%)
  [2025-09-11 14:09:20,261680][W][protocols/state_dict_adapter:76:__init__] model.safetensors.index.json not found at hf_assets_path: ./assets/hf/Llama-2-7b/model.safetensors.index.json.                     Defaulting to saving a single safetensors file if checkpoint is saved in HF format
  [2025-09-11 14:09:20,262344][W][protocols/state_dict_adapter:76:__init__] model.safetensors.index.json not found at hf_assets_path: ./assets/hf/Llama-2-7b/model.safetensors.index.json.                     Defaulting to saving a single safetensors file if checkpoint is saved in HF format
  [2025-09-11 14:09:20,262977][I][distributed/utils:224:maybe_enable_amp] Mixed precision training is handled by fully_shard
  [2025-09-11 14:09:20,263362][I][blendcorpus/train:372:__init__] Trainer is initialized with local batch size 1, global batch size 24, gradient accumulation steps 1, sequence length 4096, total steps 1000 (warmup 2)
  [2025-09-11 14:09:20,263916][I][blendcorpus/train:671:<module>] Using SDPBackend.FLASH_ATTENTION backend for SDPA
  [2025-09-11 14:09:20,264479][I][blendcorpus/train:552:train] BlendCorpus dataloader advanced to consumed=0 samples (step=0).
  [2025-09-11 14:09:20,264934][I][blendcorpus/train:556:train] Training starts at step 1.
  [2025-09-11 14:09:20,277894][W][protocols/state_dict_adapter:76:__init__] model.safetensors.index.json not found at hf_assets_path: ./assets/hf/Llama-2-7b/model.safetensors.index.json.                     Defaulting to saving a single safetensors file if checkpoint is saved in HF format
  [2025-09-11 14:09:20,308252][W][protocols/state_dict_adapter:76:__init__] model.safetensors.index.json not found at hf_assets_path: ./assets/hf/Llama-2-7b/model.safetensors.index.json.                     Defaulting to saving a single safetensors file if checkpoint is saved in HF format
  [2025-09-11 14:09:20,322939][W][protocols/state_dict_adapter:76:__init__] model.safetensors.index.json not found at hf_assets_path: ./assets/hf/Llama-2-7b/model.safetensors.index.json.                     Defaulting to saving a single safetensors file if checkpoint is saved in HF format
  [2025-09-11 14:09:20,353381][W][protocols/state_dict_adapter:76:__init__] model.safetensors.index.json not found at hf_assets_path: ./assets/hf/Llama-2-7b/model.safetensors.index.json.                     Defaulting to saving a single safetensors file if checkpoint is saved in HF format
  [2025-09-11 14:09:20,354507][W][protocols/state_dict_adapter:76:__init__] model.safetensors.index.json not found at hf_assets_path: ./assets/hf/Llama-2-7b/model.safetensors.index.json.                     Defaulting to saving a single safetensors file if checkpoint is saved in HF format
  [2025-09-11 14:09:20,361048][W][protocols/state_dict_adapter:76:__init__] model.safetensors.index.json not found at hf_assets_path: ./assets/hf/Llama-2-7b/model.safetensors.index.json.                     Defaulting to saving a single safetensors file if checkpoint is saved in HF format
  [2025-09-11 14:09:20,361592][W][protocols/state_dict_adapter:76:__init__] model.safetensors.index.json not found at hf_assets_path: ./assets/hf/Llama-2-7b/model.safetensors.index.json.                     Defaulting to saving a single safetensors file if checkpoint is saved in HF format
  [2025-09-11 14:09:20,362237][W][protocols/state_dict_adapter:76:__init__] model.safetensors.index.json not found at hf_assets_path: ./assets/hf/Llama-2-7b/model.safetensors.index.json.                     Defaulting to saving a single safetensors file if checkpoint is saved in HF format
  [2025-09-11 14:09:20,363205][W][protocols/state_dict_adapter:76:__init__] model.safetensors.index.json not found at hf_assets_path: ./assets/hf/Llama-2-7b/model.safetensors.index.json.                     Defaulting to saving a single safetensors file if checkpoint is saved in HF format
  [2025-09-11 14:09:46,954794][I][components/metrics:437:log] step:  1  loss: 10.8595  grad_norm:  3.2105  memory:  8.44GiB(13.20%)  tps: 149  tflops: 1.39  mfu: 0.47%
  [2025-09-11 14:09:46,956945][I][distributed/utils:298:set_pg_timeouts] Synchronizing and adjusting timeout for all ProcessGroups to 0:01:40
  [2025-09-11 14:09:47,568581][I][components/metrics:437:log] step:  2  loss: 14.4458  grad_norm: 41.7880  memory:  8.44GiB(13.20%)  tps: 6,697  tflops: 62.15  mfu: 20.84%
  [2025-09-11 14:09:48,193890][I][components/metrics:437:log] step:  3  loss: 15.2005  grad_norm: 66.6053  memory:  8.44GiB(13.20%)  tps: 6,562  tflops: 60.90  mfu: 20.42%
  [2025-09-11 14:09:48,819161][I][components/metrics:437:log] step:  4  loss: 11.3134  grad_norm: 14.0597  memory:  8.44GiB(13.20%)  tps: 6,563  tflops: 60.90  mfu: 20.42%
  [2025-09-11 14:09:49,437653][I][components/metrics:437:log] step:  5  loss: 11.1154  grad_norm: 38.3529  memory:  8.44GiB(13.20%)  tps: 6,634  tflops: 61.57  mfu: 20.65%
  [2025-09-11 14:09:50,056802][I][components/metrics:437:log] step:  6  loss: 10.1939  grad_norm: 18.1903  memory:  8.44GiB(13.20%)  tps: 6,627  tflops: 61.50  mfu: 20.63%
  [2025-09-11 14:09:50,671861][I][components/metrics:437:log] step:  7  loss:  9.8287  grad_norm: 13.8677  memory:  8.44GiB(13.20%)  tps: 6,671  tflops: 61.91  mfu: 20.76%
  [2025-09-11 14:09:51,285366][I][components/metrics:437:log] step:  8  loss:  9.7202  grad_norm:  5.0234  memory:  8.44GiB(13.20%)  tps: 6,688  tflops: 62.07  mfu: 20.82%
  [2025-09-11 14:09:51,898897][I][components/metrics:437:log] step:  9  loss:  9.2838  grad_norm:  4.8296  memory:  8.44GiB(13.20%)  tps: 6,688  tflops: 62.07  mfu: 20.81%
  [2025-09-11 14:09:52,511523][I][components/metrics:437:log] step: 10  loss:  9.4166  grad_norm: 24.4741  memory:  8.44GiB(13.20%)  tps: 6,698  tflops: 62.15  mfu: 20.84%
  [2025-09-11 14:09:53,123794][I][components/metrics:437:log] step: 11  loss:  9.2494  grad_norm:  8.9617  memory:  8.44GiB(13.20%)  tps: 6,701  tflops: 62.19  mfu: 20.86%
  [2025-09-11 14:09:53,735590][I][components/metrics:437:log] step: 12  loss:  9.1097  grad_norm:  6.0902  memory:  8.44GiB(13.20%)  tps: 6,707  tflops: 62.24  mfu: 20.87%
  [2025-09-11 14:09:54,347347][I][components/metrics:437:log] step: 13  loss:  8.4231  grad_norm:  4.4072  memory:  8.44GiB(13.20%)  tps: 6,707  tflops: 62.24  mfu: 20.87%
  [2025-09-11 14:09:54,958006][I][components/metrics:437:log] step: 14  loss:  8.2500  grad_norm:  5.8164  memory:  8.44GiB(13.20%)  tps: 6,719  tflops: 62.35  mfu: 20.91%
  [2025-09-11 14:09:55,568354][I][components/metrics:437:log] step: 15  loss:  7.8394  grad_norm:  4.1659  memory:  8.44GiB(13.20%)  tps: 6,722  tflops: 62.38  mfu: 20.92%
  [2025-09-11 14:09:56,181240][I][components/metrics:437:log] step: 16  loss:  7.4599  grad_norm:  5.2217  memory:  8.44GiB(13.20%)  tps: 6,695  tflops: 62.13  mfu: 20.83%
  [2025-09-11 14:09:56,792654][I][components/metrics:437:log] step: 17  loss:  7.3650  grad_norm: 10.0954  memory:  8.44GiB(13.20%)  tps: 6,711  tflops: 62.28  mfu: 20.89%
  [2025-09-11 14:09:57,403952][I][components/metrics:437:log] step: 18  loss:  8.5406  grad_norm: 42.4819  memory:  8.44GiB(13.20%)  tps: 6,712  tflops: 62.29  mfu: 20.89%
  [2025-09-11 14:09:58,015552][I][components/metrics:437:log] step: 19  loss:  8.1060  grad_norm: 26.1239  memory:  8.44GiB(13.20%)  tps: 6,708  tflops: 62.25  mfu: 20.88%
  [2025-09-11 14:09:58,627123][I][components/metrics:437:log] step: 20  loss:  7.9984  grad_norm: 14.1964  memory:  8.44GiB(13.20%)  tps: 6,709  tflops: 62.26  mfu: 20.88%
  [2025-09-11 14:09:59,240129][I][components/metrics:437:log] step: 21  loss:  7.8776  grad_norm:  8.8600  memory:  8.44GiB(13.20%)  tps: 6,693  tflops: 62.11  mfu: 20.83%
  [2025-09-11 14:09:59,853788][I][components/metrics:437:log] step: 22  loss:  7.8586  grad_norm:  3.9666  memory:  8.44GiB(13.20%)  tps: 6,686  tflops: 62.05  mfu: 20.81%
  [2025-09-11 14:10:00,467962][I][components/metrics:437:log] step: 23  loss:  7.6389  grad_norm:  5.3035  memory:  8.44GiB(13.20%)  tps: 6,680  tflops: 62.00  mfu: 20.79%
  [2025-09-11 14:10:01,080471][I][components/metrics:437:log] step: 24  loss:  7.2202  grad_norm:  5.3859  memory:  8.44GiB(13.20%)  tps: 6,699  tflops: 62.17  mfu: 20.85%
  [2025-09-11 14:10:01,694753][I][components/metrics:437:log] step: 25  loss:  7.1780  grad_norm:  3.9941  memory:  8.44GiB(13.20%)  tps: 6,679  tflops: 61.98  mfu: 20.79%
  [2025-09-11 14:10:02,308875][I][components/metrics:437:log] step: 26  loss:  6.9265  grad_norm:  4.2152  memory:  8.44GiB(13.20%)  tps: 6,681  tflops: 62.00  mfu: 20.79%
  [2025-09-11 14:10:02,922493][I][components/metrics:437:log] step: 27  loss:  7.0430  grad_norm:  4.8777  memory:  8.44GiB(13.20%)  tps: 6,686  tflops: 62.05  mfu: 20.81%
  [2025-09-11 14:10:03,535394][I][components/metrics:437:log] step: 28  loss:  7.0926  grad_norm:  3.2871  memory:  8.44GiB(13.20%)  tps: 6,694  tflops: 62.12  mfu: 20.83%
  [2025-09-11 14:10:04,150741][I][components/metrics:437:log] step: 29  loss:  6.9620  grad_norm:  5.2030  memory:  8.44GiB(13.20%)  tps: 6,668  tflops: 61.88  mfu: 20.75%
  [2025-09-11 14:10:04,766409][I][components/metrics:437:log] step: 30  loss:  6.9985  grad_norm:  4.7326  memory:  8.44GiB(13.20%)  tps: 6,664  tflops: 61.84  mfu: 20.74%
  [2025-09-11 14:10:05,382267][I][components/metrics:437:log] step: 31  loss:  6.8697  grad_norm:  3.9675  memory:  8.44GiB(13.20%)  tps: 6,662  tflops: 61.83  mfu: 20.73%
  [2025-09-11 14:10:05,996739][I][components/metrics:437:log] step: 32  loss:  7.0787  grad_norm:  2.6731  memory:  8.44GiB(13.20%)  tps: 6,677  tflops: 61.96  mfu: 20.78%
  [2025-09-11 14:10:06,611131][I][components/metrics:437:log] step: 33  loss:  6.8755  grad_norm:  3.5877  memory:  8.44GiB(13.20%)  tps: 6,678  tflops: 61.97  mfu: 20.78%
  [2025-09-11 14:10:07,228592][I][components/metrics:437:log] step: 34  loss:  7.0408  grad_norm:  2.6362  memory:  8.44GiB(13.20%)  tps: 6,644  tflops: 61.66  mfu: 20.68%
  [2025-09-11 14:10:07,846150][I][components/metrics:437:log] step: 35  loss:  6.7944  grad_norm:  3.0056  memory:  8.44GiB(13.20%)  tps: 6,643  tflops: 61.65  mfu: 20.68%
  [2025-09-11 14:10:08,462487][I][components/metrics:437:log] step: 36  loss:  6.8286  grad_norm:  2.3594  memory:  8.44GiB(13.20%)  tps: 6,657  tflops: 61.77  mfu: 20.72%
  [2025-09-11 14:10:09,080194][I][components/metrics:437:log] step: 37  loss:  6.7154  grad_norm:  2.0318  memory:  8.44GiB(13.20%)  tps: 6,642  tflops: 61.64  mfu: 20.67%
  [2025-09-11 14:10:09,698650][I][components/metrics:437:log] step: 38  loss:  6.8278  grad_norm:  2.8517  memory:  8.44GiB(13.20%)  tps: 6,634  tflops: 61.56  mfu: 20.65%
  [2025-09-11 14:10:10,315144][I][components/metrics:437:log] step: 39  loss:  6.7203  grad_norm:  2.6356  memory:  8.44GiB(13.20%)  tps: 6,655  tflops: 61.76  mfu: 20.71%
  [2025-09-11 14:10:10,931986][I][components/metrics:437:log] step: 40  loss:  6.6178  grad_norm:  2.1263  memory:  8.44GiB(13.20%)  tps: 6,651  tflops: 61.73  mfu: 20.70%
  [2025-09-11 14:10:11,551633][I][components/metrics:437:log] step: 41  loss:  6.6646  grad_norm:  2.1531  memory:  8.44GiB(13.20%)  tps: 6,621  tflops: 61.45  mfu: 20.61%
  [2025-09-11 14:10:12,169671][I][components/metrics:437:log] step: 42  loss:  6.6434  grad_norm:  2.5737  memory:  8.44GiB(13.20%)  tps: 6,638  tflops: 61.60  mfu: 20.66%
  [2025-09-11 14:10:12,787708][I][components/metrics:437:log] step: 43  loss:  6.4677  grad_norm:  1.9998  memory:  8.44GiB(13.20%)  tps: 6,638  tflops: 61.61  mfu: 20.66%
  [2025-09-11 14:10:13,406286][I][components/metrics:437:log] step: 44  loss:  6.4482  grad_norm:  1.7345  memory:  8.44GiB(13.20%)  tps: 6,632  tflops: 61.55  mfu: 20.64%
  [2025-09-11 14:10:14,024357][I][components/metrics:437:log] step: 45  loss:  6.3464  grad_norm:  2.1871  memory:  8.44GiB(13.20%)  tps: 6,638  tflops: 61.60  mfu: 20.66%
  [2025-09-11 14:10:14,644998][I][components/metrics:437:log] step: 46  loss:  6.4503  grad_norm:  2.3965  memory:  8.44GiB(13.20%)  tps: 6,611  tflops: 61.35  mfu: 20.57%
  [2025-09-11 14:10:15,264097][I][components/metrics:437:log] step: 47  loss:  6.4022  grad_norm:  2.0624  memory:  8.44GiB(13.20%)  tps: 6,627  tflops: 61.50  mfu: 20.62%
  [2025-09-11 14:10:15,884593][I][components/metrics:437:log] step: 48  loss:  6.4093  grad_norm:  1.7753  memory:  8.44GiB(13.20%)  tps: 6,612  tflops: 61.36  mfu: 20.58%
  [2025-09-11 14:10:16,506339][I][components/metrics:437:log] step: 49  loss:  6.1545  grad_norm:  1.8817  memory:  8.44GiB(13.20%)  tps: 6,599  tflops: 61.24  mfu: 20.54%
  [2025-09-11 14:10:16,513354][I][tools/utils:63:collect] [GC] Performing periodical GC collection 0.01 seconds
  [2025-09-11 14:10:17,131680][I][components/metrics:437:log] step: 50  loss:  6.3506  grad_norm:  2.1493  memory:  8.44GiB(13.20%)  tps: 6,561  tflops: 60.88  mfu: 20.42%
  [2025-09-11 14:10:17,751098][I][components/metrics:437:log] step: 51  loss:  5.9018  grad_norm:  1.8436  memory:  8.44GiB(13.20%)  tps: 6,624  tflops: 61.47  mfu: 20.61%
  [2025-09-11 14:10:18,373376][I][components/metrics:437:log] step: 52  loss:  6.0532  grad_norm:  2.1721  memory:  8.44GiB(13.20%)  tps: 6,593  tflops: 61.18  mfu: 20.52%
  [2025-09-11 14:10:18,994459][I][components/metrics:437:log] step: 53  loss:  5.8463  grad_norm:  1.9916  memory:  8.44GiB(13.20%)  tps: 6,606  tflops: 61.30  mfu: 20.56%
  [2025-09-11 14:10:19,614958][I][components/metrics:437:log] step: 54  loss:  5.8591  grad_norm:  2.2927  memory:  8.44GiB(13.20%)  tps: 6,612  tflops: 61.36  mfu: 20.58%
  [2025-09-11 14:10:20,240067][I][components/metrics:437:log] step: 55  loss:  5.7876  grad_norm:  2.1256  memory:  8.44GiB(13.20%)  tps: 6,563  tflops: 60.91  mfu: 20.43%
  [2025-09-11 14:10:20,861372][I][components/metrics:437:log] step: 56  loss:  5.9472  grad_norm:  1.6179  memory:  8.44GiB(13.20%)  tps: 6,604  tflops: 61.29  mfu: 20.55%
  [2025-09-11 14:10:21,481566][I][components/metrics:437:log] step: 57  loss:  5.8461  grad_norm:  1.8138  memory:  8.44GiB(13.20%)  tps: 6,616  tflops: 61.40  mfu: 20.59%
  [2025-09-11 14:10:22,102375][I][components/metrics:437:log] step: 58  loss:  5.9919  grad_norm:  1.8530  memory:  8.44GiB(13.20%)  tps: 6,609  tflops: 61.33  mfu: 20.57%
  [2025-09-11 14:10:22,723432][I][components/metrics:437:log] step: 59  loss:  6.0051  grad_norm:  1.5441  memory:  8.44GiB(13.20%)  tps: 6,606  tflops: 61.31  mfu: 20.56%
  [2025-09-11 14:10:23,342099][I][components/metrics:437:log] step: 60  loss:  5.8526  grad_norm:  2.6306  memory:  8.44GiB(13.20%)  tps: 6,631  tflops: 61.54  mfu: 20.64%
  [2025-09-11 14:10:23,962145][I][components/metrics:437:log] step: 61  loss:  5.7798  grad_norm:  1.9169  memory:  8.44GiB(13.20%)  tps: 6,617  tflops: 61.40  mfu: 20.59%
  [2025-09-11 14:10:24,582054][I][components/metrics:437:log] step: 62  loss:  5.8291  grad_norm:  1.5668  memory:  8.44GiB(13.20%)  tps: 6,618  tflops: 61.42  mfu: 20.60%
  [2025-09-11 14:10:25,201408][I][components/metrics:437:log] step: 63  loss:  5.6550  grad_norm:  1.9091  memory:  8.44GiB(13.20%)  tps: 6,624  tflops: 61.48  mfu: 20.62%
  [2025-09-11 14:10:25,822004][I][components/metrics:437:log] step: 64  loss:  5.5856  grad_norm:  2.0817  memory:  8.44GiB(13.20%)  tps: 6,611  tflops: 61.35  mfu: 20.57%
  [2025-09-11 14:10:26,441642][I][components/metrics:437:log] step: 65  loss:  5.6917  grad_norm:  2.0554  memory:  8.44GiB(13.20%)  tps: 6,621  tflops: 61.45  mfu: 20.61%
  [2025-09-11 14:10:27,061924][I][components/metrics:437:log] step: 66  loss:  5.6905  grad_norm:  2.1699  memory:  8.44GiB(13.20%)  tps: 6,614  tflops: 61.38  mfu: 20.58%
  [2025-09-11 14:10:27,683182][I][components/metrics:437:log] step: 67  loss:  5.6973  grad_norm:  2.2035  memory:  8.44GiB(13.20%)  tps: 6,604  tflops: 61.28  mfu: 20.55%
  [2025-09-11 14:10:28,302396][I][components/metrics:437:log] step: 68  loss:  5.7700  grad_norm:  1.9694  memory:  8.44GiB(13.20%)  tps: 6,626  tflops: 61.49  mfu: 20.62%
  [2025-09-11 14:10:28,921775][I][components/metrics:437:log] step: 69  loss:  5.5425  grad_norm:  1.9954  memory:  8.44GiB(13.20%)  tps: 6,624  tflops: 61.47  mfu: 20.61%
  [2025-09-11 14:10:29,541704][I][components/metrics:437:log] step: 70  loss:  5.5834  grad_norm:  1.9733  memory:  8.44GiB(13.20%)  tps: 6,618  tflops: 61.42  mfu: 20.60%
  [2025-09-11 14:10:30,163329][I][components/metrics:437:log] step: 71  loss:  5.4515  grad_norm:  2.1977  memory:  8.44GiB(13.20%)  tps: 6,600  tflops: 61.25  mfu: 20.54%
  [2025-09-11 14:10:30,782215][I][components/metrics:437:log] step: 72  loss:  5.6627  grad_norm:  1.9666  memory:  8.44GiB(13.20%)  tps: 6,630  tflops: 61.52  mfu: 20.63%
  [2025-09-11 14:10:31,400804][I][components/metrics:437:log] step: 73  loss:  5.3765  grad_norm:  1.3680  memory:  8.44GiB(13.20%)  tps: 6,633  tflops: 61.55  mfu: 20.64%
  [2025-09-11 14:10:32,019076][I][components/metrics:437:log] step: 74  loss:  5.3842  grad_norm:  1.6721  memory:  8.44GiB(13.20%)  tps: 6,636  tflops: 61.58  mfu: 20.65%
  [2025-09-11 14:10:32,638430][I][components/metrics:437:log] step: 75  loss:  5.6086  grad_norm:  1.6692  memory:  8.44GiB(13.20%)  tps: 6,624  tflops: 61.47  mfu: 20.62%
  [2025-09-11 14:10:33,257389][I][components/metrics:437:log] step: 76  loss:  5.5859  grad_norm:  1.2709  memory:  8.44GiB(13.20%)  tps: 6,628  tflops: 61.51  mfu: 20.63%
  [2025-09-11 14:10:33,876899][I][components/metrics:437:log] step: 77  loss:  5.5390  grad_norm:  1.9309  memory:  8.44GiB(13.20%)  tps: 6,622  tflops: 61.46  mfu: 20.61%
  [2025-09-11 14:10:34,495668][I][components/metrics:437:log] step: 78  loss:  5.2713  grad_norm:  2.2211  memory:  8.44GiB(13.20%)  tps: 6,630  tflops: 61.53  mfu: 20.64%
  [2025-09-11 14:10:35,116794][I][components/metrics:437:log] step: 79  loss:  5.5750  grad_norm:  2.0357  memory:  8.44GiB(13.20%)  tps: 6,605  tflops: 61.30  mfu: 20.56%
  [2025-09-11 14:10:35,734839][I][components/metrics:437:log] step: 80  loss:  5.4112  grad_norm:  1.6118  memory:  8.44GiB(13.20%)  tps: 6,638  tflops: 61.60  mfu: 20.66%
  [2025-09-11 14:10:36,353056][I][components/metrics:437:log] step: 81  loss:  5.4936  grad_norm:  2.1059  memory:  8.44GiB(13.20%)  tps: 6,636  tflops: 61.59  mfu: 20.65%
  [2025-09-11 14:10:36,973885][I][components/metrics:437:log] step: 82  loss:  5.3963  grad_norm:  2.1175  memory:  8.44GiB(13.20%)  tps: 6,608  tflops: 61.33  mfu: 20.57%
  [2025-09-11 14:10:37,592879][I][components/metrics:437:log] step: 83  loss:  5.5316  grad_norm:  1.6825  memory:  8.44GiB(13.20%)  tps: 6,628  tflops: 61.51  mfu: 20.63%
  [2025-09-11 14:10:38,210657][I][components/metrics:437:log] step: 84  loss:  5.5413  grad_norm:  3.2314  memory:  8.44GiB(13.20%)  tps: 6,641  tflops: 61.63  mfu: 20.67%
  [2025-09-11 14:10:38,830715][I][components/metrics:437:log] step: 85  loss:  5.6706  grad_norm:  1.7961  memory:  8.44GiB(13.20%)  tps: 6,617  tflops: 61.40  mfu: 20.59%
  [2025-09-11 14:10:39,450180][I][components/metrics:437:log] step: 86  loss:  5.4076  grad_norm:  2.0268  memory:  8.44GiB(13.20%)  tps: 6,623  tflops: 61.46  mfu: 20.61%
  [2025-09-11 14:10:40,069226][I][components/metrics:437:log] step: 87  loss:  5.3278  grad_norm:  2.3101  memory:  8.44GiB(13.20%)  tps: 6,627  tflops: 61.50  mfu: 20.63%
  [2025-09-11 14:10:40,687323][I][components/metrics:437:log] step: 88  loss:  5.7212  grad_norm:  2.5316  memory:  8.44GiB(13.20%)  tps: 6,638  tflops: 61.60  mfu: 20.66%
  [2025-09-11 14:10:41,305510][I][components/metrics:437:log] step: 89  loss:  5.4569  grad_norm:  2.1162  memory:  8.44GiB(13.20%)  tps: 6,637  tflops: 61.59  mfu: 20.66%
  [2025-09-11 14:10:41,923057][I][components/metrics:437:log] step: 90  loss:  5.6080  grad_norm:  2.5761  memory:  8.44GiB(13.20%)  tps: 6,644  tflops: 61.66  mfu: 20.68%
  [2025-09-11 14:10:42,541577][I][components/metrics:437:log] step: 91  loss:  5.5149  grad_norm:  1.9985  memory:  8.44GiB(13.20%)  tps: 6,633  tflops: 61.56  mfu: 20.64%
  [2025-09-11 14:10:43,160727][I][components/metrics:437:log] step: 92  loss:  5.5408  grad_norm:  2.0675  memory:  8.44GiB(13.20%)  tps: 6,626  tflops: 61.49  mfu: 20.62%
  [2025-09-11 14:10:43,780841][I][components/metrics:437:log] step: 93  loss:  5.5718  grad_norm:  2.1958  memory:  8.44GiB(13.20%)  tps: 6,616  tflops: 61.40  mfu: 20.59%
  [2025-09-11 14:10:44,400406][I][components/metrics:437:log] step: 94  loss:  5.4010  grad_norm:  1.8711  memory:  8.44GiB(13.20%)  tps: 6,622  tflops: 61.45  mfu: 20.61%
  [2025-09-11 14:10:45,019376][I][components/metrics:437:log] step: 95  loss:  5.3711  grad_norm:  1.6309  memory:  8.44GiB(13.20%)  tps: 6,628  tflops: 61.51  mfu: 20.63%
  [2025-09-11 14:10:45,638017][I][components/metrics:437:log] step: 96  loss:  5.5253  grad_norm:  2.1529  memory:  8.44GiB(13.20%)  tps: 6,632  tflops: 61.54  mfu: 20.64%
  [2025-09-11 14:10:46,256533][I][components/metrics:437:log] step: 97  loss:  5.6043  grad_norm:  1.5532  memory:  8.44GiB(13.20%)  tps: 6,633  tflops: 61.56  mfu: 20.64%
  [2025-09-11 14:10:46,876233][I][components/metrics:437:log] step: 98  loss:  5.2414  grad_norm:  1.8785  memory:  8.44GiB(13.20%)  tps: 6,620  tflops: 61.44  mfu: 20.60%
  [2025-09-11 14:10:47,494307][I][components/metrics:437:log] step: 99  loss:  5.4408  grad_norm:  1.9023  memory:  8.44GiB(13.20%)  tps: 6,638  tflops: 61.60  mfu: 20.66%
  [2025-09-11 14:10:47,498379][I][tools/utils:63:collect] [GC] Performing periodical GC collection 0.00 seconds
  [2025-09-11 14:10:48,117495][I][components/metrics:437:log] step: 100  loss:  5.5676  grad_norm:  1.6557  memory:  8.44GiB(13.20%)  tps: 6,583  tflops: 61.09  mfu: 20.49%
  [2025-09-11 14:10:48,735546][I][components/metrics:437:log] step: 101  loss:  5.4260  grad_norm:  1.7622  memory:  8.44GiB(13.20%)  tps: 6,638  tflops: 61.60  mfu: 20.66%
  [2025-09-11 14:10:49,354915][I][components/metrics:437:log] step: 102  loss:  5.2493  grad_norm:  1.5373  memory:  8.44GiB(13.20%)  tps: 6,624  tflops: 61.47  mfu: 20.61%
  [2025-09-11 14:10:49,973188][I][components/metrics:437:log] step: 103  loss:  5.2459  grad_norm:  1.6686  memory:  8.44GiB(13.20%)  tps: 6,636  tflops: 61.58  mfu: 20.65%
  [2025-09-11 14:10:50,591512][I][components/metrics:437:log] step: 104  loss:  5.3994  grad_norm:  1.9303  memory:  8.44GiB(13.20%)  tps: 6,635  tflops: 61.57  mfu: 20.65%
  [2025-09-11 14:10:51,208789][I][components/metrics:437:log] step: 105  loss:  5.3256  grad_norm:  1.5555  memory:  8.44GiB(13.20%)  tps: 6,647  tflops: 61.68  mfu: 20.69%
  ^C
  [1]    169285 exit 1     ezpz-launch -m torchtitan.experiments.blendcorpus.train --job.config_file  |
         169290 interrupt  tee tt-bc-$(tstamp).log
  took: 0h:02m:06s
  ```

```
