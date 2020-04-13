# rtxTetAdvect - Collaboration Project for RTX-accelerated Tet-Mesh Particle Advection
==================================================================

# Building
==================================================================

## Dependencies

Optix 7, CUDA 10, and a recent NVIDIA driver.

## Checkout w/ Submodule

Mind this project uses a git submodule, so idealy clone with
`--recursive` flag:

    git clone --recursive http://github.com/owl-project/tetMeshQueries

If you did close without this flag, you can afterwards also do a

    git submodule init
	git submodule update
	
Building then via cmake and your favorite compiler toolchain.

## Building on Linux

For linux, I assume that `nvcc` is in the default path, and that there is environment variable names 'OptiX_INSTALL_DIR` that points to the optix SDK install directory

    cd tetMeshQueries
	mkdir bin
	cd bin
	cmake ..
	make

If you have any missing dependencies ccmake will tell you... just fix and re-run cmake


# License
==================================================================

Apache Licence 2.0 - pretty much do with it what you like; no warranties.

# Authors/Collaborators
==================================================================

Main Authors of this library:
- Ingo Wald (NVIDIA)
- Bin Wang (LSU)

In addition, many of the core algorithms were imported from a project
originally co-authored also by Will Usher and Nate Morrical, from the
SCI Insitute, University of Utah.

Reference to the original paper:

Wald, I., Usher, W., Morrical, N., Lediaev, L., & Pascucci,
V. (2019). RTX Beyond Ray Tracing: Exploring the Use of Hardware Ray
Tracing Cores for Tet-Mesh Point Location. Proceedings of High
Performance Graphics.

