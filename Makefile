.PHONY: all rtsim clean

RTSIM_CXX ?= g++-11
RTSIM_DIR ?= RTSim
RTSIM_EXEC ?= $(RTSIM_DIR)/nvmain.fast

all: rtsim

rtsim:
	@echo "==== Building RTSim ===="
	@cd $(RTSIM_DIR) && scons CXX=$(RTSIM_CXX) --build-type=fast --verbose -j$$(nproc)

clean:
	@cd $(RTSIM_DIR) && scons -c
