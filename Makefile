PYTHON ?= python3

.PHONY: help bootstrap doctor smoke elementwise relu hgemm-smoke flash-attn-smoke hgemm-wheel hgemm-wheel-multi

help:
	@printf '%s\n' \
		'bootstrap          Initialize submodules and baseline Python deps' \
		'doctor             Inspect GPU / CUDA / PyTorch / submodule status' \
		'smoke              Run the recommended end-to-end smoke checks' \
		'elementwise        Build and run the elementwise teaching example' \
		'relu               Build and run the relu teaching example' \
		'hgemm-smoke        JIT-build and run a minimal HGEMM benchmark' \
		'flash-attn-smoke   JIT-build and run a minimal FlashAttention subset' \
		'hgemm-wheel        Build a native-arch HGEMM wheel for this machine' \
		'hgemm-wheel-multi  Build a multi-arch HGEMM wheel for distribution'

bootstrap:
	bash scripts/bootstrap_env.sh

doctor:
	$(PYTHON) scripts/doctor.py

smoke: doctor elementwise relu hgemm-smoke flash-attn-smoke

elementwise:
	$(PYTHON) scripts/run_example.py kernels/elementwise/elementwise.py

relu:
	$(PYTHON) scripts/run_example.py kernels/relu/relu.py

hgemm-smoke:
	$(PYTHON) scripts/run_example.py kernels/hgemm/hgemm.py --M 256 --N 256 --K 256 --mma --no-default --iters 1 --warmup 0

flash-attn-smoke:
	$(PYTHON) scripts/run_example.py kernels/flash-attn/flash_attn_mma.py --minimal-build --tag-hints split-kv,cute --B 1 --H 1 --N 128 --D 64 --iters 1 --warmup 0 --sdpa

hgemm-wheel:
	cd kernels/hgemm && $(PYTHON) setup.py bdist_wheel

hgemm-wheel-multi:
	cd kernels/hgemm && LEETCUDA_WHEEL_ARCHES=default $(PYTHON) setup.py bdist_wheel
