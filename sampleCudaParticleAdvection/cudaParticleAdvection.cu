// ======================================================================== //
// Copyright 2019-2020 Ingo Wald (NVIDIA) and Bin Wang (LSU)                //
//                                                                          //
// Licensed under the Apache License, Version 2.0 (the "License");          //
// you may not use this file except in compliance with the License.         //
// You may obtain a copy of the License at                                  //
//                                                                          //
//     http://www.apache.org/licenses/LICENSE-2.0                           //
//                                                                          //
// Unless required by applicable law or agreed to in writing, software      //
// distributed under the License is distributed on an "AS IS" BASIS,        //
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. //
// See the License for the specific language governing permissions and      //
// limitations under the License.                                           //
// ======================================================================== //

#include "tetMeshQueryLib/OptixTetQuery.h"
#include "DeviceTetMesh.cuh"
#include "HostTetMesh.cuh"
#include <owl/common/math/random.h>
#include <fstream>

namespace advect {
  using namespace owl;
  using namespace owl::common;
  using namespace owl::tetQueries;
  
  std::string objTrajectoryFileName;
  std::string vtkStreamlineFileName;
  std::vector<std::vector<vec3f>> trajectories;
  
  typedef DoubleParticle Particle;
  //typedef FloatParticle Particle;

  typedef owl::common::box_t<vec3d> box3d;

  __global__ void initParticlesKernel(Particle *particles, int N,
                                      const box3d worldBounds)
  {
    int particleID = threadIdx.x + blockDim.x * blockIdx.x;
    if (particleID >= N) return;

    LCG<16> random;
    random.init(threadIdx.x,blockIdx.x);
    
    // *any* random position for now:
    vec3d randomPos
      = worldBounds.lower
      + vec3d(random(),random(),random())
      * worldBounds.size();
    
    (vec3d&)particles[particleID].pos = vec3d(randomPos);
	particles[particleID].isActive = true;
  }
  
  void cudaInitParticles(Particle *d_particles, int N,
                         const box3d &worldBounds)
  {
    int blockDims = 128;
    int gridDims  = divRoundUp(N,blockDims);
    initParticlesKernel<<<gridDims,blockDims>>>(d_particles, N, worldBounds);
  }


  inline __device__ double det(const vec3d A,
                               const vec3d B,
                               const vec3d C,
                               const vec3d D)
  {
    return dot(D-A,cross(B-A,C-A));
  }
                               

  __global__
  void particleAdvectKernel(Particle *d_particles,
							int      *d_tetIDs,
                            int       numParticles,
                            vec4i    *d_tetIndices,
                            vec3d    *d_vertexPositions,
                            vec3d    *d_vertexVelocities,
                            double    dt)
  {
    int particleID = threadIdx.x + blockDim.x * blockIdx.x;
    if (particleID >= numParticles) return;

    Particle &p = d_particles[particleID];
    if (!p.isActive) return;

    const int tetID = d_tetIDs[particleID];
    if (tetID < 0) {//tet=-1
		// this particle left the domain
		p.isActive = false;
      return;
    }

    vec4i index = d_tetIndices[tetID];
    const vec3d P = (const vec3d&)p.pos;
    const vec3d A = d_vertexPositions[index.x];
    const vec3d B = d_vertexPositions[index.y];
    const vec3d C = d_vertexPositions[index.z];
    const vec3d D = d_vertexPositions[index.w];

    const double den = det(A,B,C,D);
    if (den == 0.f) {//We are in a bad tet, set tetID=-2
		p.isActive = false;
      return;
    }

    const double wA = det(P,B,C,D) * (1./den);
    const double wB = det(A,P,C,D) * (1./den);
    const double wC = det(A,B,P,D) * (1./den);
    const double wD = det(A,B,C,P) * (1./den);

    const vec3d velA = d_vertexVelocities[index.x];
    const vec3d velB = d_vertexVelocities[index.y];
    const vec3d velC = d_vertexVelocities[index.z];
    const vec3d velD = d_vertexVelocities[index.w];

    const vec3d vel
      = wA * velA
      + wB * velB
      + wC * velC
      + wD * velD;

    ((vec3d &)p.pos) += dt * vel;
  }

  __global__
	  void particleAdvectKernelTetVel(Particle *d_particles,
		  int      *d_tetIDs,
		  int       numParticles,
		  vec4i    *d_tetIndices,
		  vec3d    *d_vertexPositions,
		  vec3d    *d_TetVelocities,
		  double    dt)
  {
	  int particleID = threadIdx.x + blockDim.x * blockIdx.x;
	  if (particleID >= numParticles) return;

	  Particle &p = d_particles[particleID];
	  if (!p.isActive) return;

	  const int tetID = d_tetIDs[particleID];
	  if (tetID < 0) {//tet=-1
		  // this particle left the domain
		  p.isActive = false;
		  return;
	  }

	  vec4i index = d_tetIndices[tetID];
	  const vec3d P = (const vec3d&)p.pos;
	  const vec3d A = d_vertexPositions[index.x];
	  const vec3d B = d_vertexPositions[index.y];
	  const vec3d C = d_vertexPositions[index.z];
	  const vec3d D = d_vertexPositions[index.w];

	  const double den = det(A, B, C, D);
	  if (den == 0.f) {//We are in a bad tet, set tetID=-2
		  p.isActive = false;
		  return;
	  }

	  const vec3d vel
		  = (vec3d &) d_TetVelocities[tetID];

	  ((vec3d &)p.pos) += dt * vel;
  }

  void cudaParticleAdvect(Particle *d_particles,
						  int      *d_tetIDs,
                          int numParticles,
                          vec4i    *d_tetIndices,
                          vec3d    *d_vertexPositions,
                          vec3d    *d_Velocities,
						  bool isTetVelocity,
                          double    dt)
  {
    int blockDims = 128;
    int gridDims  = divRoundUp(numParticles,blockDims);

	if(isTetVelocity)
		particleAdvectKernelTetVel<<<gridDims,blockDims>>> (d_particles,
															d_tetIDs,
															numParticles,
															d_tetIndices,
															d_vertexPositions,
															d_Velocities,
															dt);
	else
	particleAdvectKernel<<<gridDims,blockDims>>>(d_particles,
												 d_tetIDs,
                                                 numParticles,
                                                 d_tetIndices,
                                                 d_vertexPositions,
												 d_Velocities,
                                                 dt);
  }

  
  void addToTrajectories(Particle *d_particles,
                                int numParticles)
  {
    if (trajectories.empty())
      trajectories.resize(numParticles);

    cudaDeviceSynchronize();

    std::vector<Particle> hostParticles(numParticles);
    cudaCheck(cudaMemcpy(hostParticles.data(),
                         d_particles,
                         numParticles * sizeof(Particle),
                         cudaMemcpyDeviceToHost));
    for (int i=0;i<numParticles;i++) {
      if (!hostParticles[i].isActive) continue;
      
      vec3f p(hostParticles[i].pos[0],
              hostParticles[i].pos[1],
              hostParticles[i].pos[2]);
      trajectories[i].push_back(p);
    }
  }
  
  void saveTrajectories(const std::string &fileName)
  {
	  std::ofstream out(fileName);
	  int numVerticesWritten = 0;
	  for (auto &traj : trajectories) {
		  int firstVertexID = numVerticesWritten + 1; // +1 for obj format
		  if (traj.size() <= 1) continue;
		  for (auto &p : traj) {
			  out << "v " << p.x << " " << p.y << " " << p.z << std::endl;
			  numVerticesWritten++;
		  }
		  for (int i = 0; i < (traj.size() - 1); i++) {
			  out << "l " << (firstVertexID + i) << " " << (firstVertexID + i + 1) << std::endl;
		  }
	  }

	  out.close();
  }

  inline void writeStreamline2VTK(const std::string &fileName) {

	  std::ofstream out(fileName);
	  int NumSLs = 0;
	  int numVerticesWritten = 0;
	  for (auto &traj : trajectories) {
		  if (traj.size() <= 1) continue;
		  for (auto &p : traj) 
			  numVerticesWritten++;
		  NumSLs+=1;
	  }

	  out << "# vtk DataFile Version 4.1\n";
	  out << "vtk output\n";
	  out << "ASCII\n";
	  out << "DATASET POLYDATA\n";
	  out << "POINTS "<<numVerticesWritten<<" float\n";
	  for (auto &traj : trajectories) {
		  if (traj.size() <= 1) continue;
		  for (auto &p : traj) 
			  out << p.x << " " << p.y << " " << p.z << std::endl;
	  }
	  out << "\n";

	  out << "LINES " << NumSLs << " " << numVerticesWritten + NumSLs << "\n";
	  int vertexID = 0;
	  for (auto &traj : trajectories) {
		  if (traj.size() <= 1) continue;
		  out << traj.size();
		  for (int i = 0; i < traj.size(); i++) {
			  out << " " << vertexID;
			  vertexID+=1;
		  }
		  out << "\n";
	  }
	  out << "\n\n";

	  out << "CELL_DATA " << NumSLs << "\n";
	  out << "FIELD FieldData 1\n"; //Now we only set one field here

	  out << "StreamlineID 1 " << NumSLs <<" int\n"; 
	  for (int i = 0; i < NumSLs; ++i)
		  out << i << " " << "\n";;

	  out.close();
  }
  
  inline void writeParticles2VTU(unsigned int ti,
                                 Particle *d_particles,
								 int *d_tetIDs,
                                 int numParticles)
  {
    //Move data from GPU
    cudaDeviceSynchronize();
	
    std::vector<Particle> hostParticles(numParticles);
    cudaCheck(cudaMemcpy(hostParticles.data(),
                         d_particles,
                         numParticles * sizeof(Particle),
                         cudaMemcpyDeviceToHost));

	std::vector<int> h_tetIDs(numParticles);
	cudaCheck(cudaMemcpy(h_tetIDs.data(),
		d_tetIDs,
		numParticles * sizeof(int),
		cudaMemcpyDeviceToHost));

    //Output particle into VTU file
    int i;
    FILE *fp;
    char fileName[1024];

    sprintf(fileName, "particle_%04d.vtu", ti);
    if(ti % 500 ==0) printf("#adv: Write particles to file %s...\n",fileName);

    fp = fopen(fileName, "w");
    //fprintf(fp, "<?xml version='1.0' encoding='UTF-8'?>\n");
    //fprintf(fp, "<VTKFile xmlns='VTK' byte_order='LittleEndian' version='0.1' type='UnstructuredGrid'>\n");
    fprintf(fp, "<VTKFile type='UnstructuredGrid' version='1.0' byte_order='LittleEndian' header_type='UInt64'>\n");
    fprintf(fp, "<UnstructuredGrid>\n");
    fprintf(fp, "<Piece NumberOfCells='%d' NumberOfPoints='%d'>\n", numParticles, numParticles);
    fprintf(fp, "<Points>\n");
    fprintf(fp, "<DataArray NumberOfComponents='3' type='Float32' Name='Position' format='ascii'>\n");
    for (i = 0; i < numParticles; i++) {
      fprintf(fp, "%lf %lf %lf\n",
              hostParticles[i].pos[0],
              hostParticles[i].pos[1],
              hostParticles[i].pos[2]);
    }
    fprintf(fp, "</DataArray>\n");
    fprintf(fp, "</Points>\n");
    fprintf(fp, "<PointData>\n");
    fprintf(fp, "<DataArray NumberOfComponents='1' type='Int32' Name='ParticleType' format='ascii'>\n");
    for (i = 0; i < numParticles; i++) {
		fprintf(fp, "%d\n", hostParticles[i].isActive);
    }
    fprintf(fp, "</DataArray>\n");
    fprintf(fp, "<DataArray NumberOfComponents='1' type='Int32' Name='ParticleID' format='ascii'>\n");
    for (i = 0; i < numParticles; i++) {
      fprintf(fp, "%d\n", i);
    }
	fprintf(fp, "</DataArray>\n");
	fprintf(fp, "<DataArray NumberOfComponents='1' type='Int32' Name='ParticleTetID' format='ascii'>\n");
	for (i = 0; i < numParticles; i++) {
		fprintf(fp, "%d\n", h_tetIDs[i]);
	}

    fprintf(fp, "</DataArray>\n");
    fprintf(fp, "</PointData>\n");
    fprintf(fp, "<Cells>\n");
    fprintf(fp, "<DataArray type='Int32' Name='connectivity' format='ascii'>\n");
    for (i = 0; i < numParticles; i++) {
      fprintf(fp, "%d\n", i);
    }
    fprintf(fp, "</DataArray>\n");
    fprintf(fp, "<DataArray type='Int32' Name='offsets' format='ascii'>\n");
    for (i = 0; i < numParticles; i++) {
      fprintf(fp, "%d\n", i + 1);
    }
    fprintf(fp, "</DataArray>\n");
    fprintf(fp, "<DataArray type='UInt8' Name='types' format='ascii'>\n");
    for (i = 0; i < numParticles; i++) {
      fprintf(fp, "1\n");
    }
    fprintf(fp, "</DataArray>\n");
    fprintf(fp, "</Cells>\n");
    fprintf(fp, "</Piece>\n");
    fprintf(fp, "</UnstructuredGrid>\n");
    fprintf(fp, "</VTKFile>\n");
    fclose(fp);
  }
  

  extern "C" int main(int ac, char **av)
  {
    cudaTimer timer;
    timer.start();

    int numParticles = 2000;
    int testTetMeshGridSize = 8;
    int numSteps = 50000;
    double dt = 5e-4f;
	double SeedingBox[6] = { 0.0, 0.0, 0.0,
						     1.0, 1.0, 1.0 };
	bool usingSeedingBox = false;
	bool saveStreamlinetoFile = false;
	std::string vert_filename, tet_filename, velocity_vert_filename,velocity_tet_filename;
    
    for (int i=1;i<ac;i++) {
      const std::string arg = av[i];
      if (arg == "--num-particles")
        numParticles = std::atoi(av[++i]);
      else if (arg == "--num-steps")
        numSteps = std::atoi(av[++i]);
	  else if (arg == "--input_mesh") {
		  vert_filename = av[++i];
		  tet_filename = av[++i];
	  }
	  else if (arg == "--input_vertex_velocity_field") 
		  velocity_vert_filename= av[++i];
	  else if (arg == "--input_tet_velocity_field")
		  velocity_tet_filename = av[++i];
	  else if (arg == "--test-grid-size")
		  testTetMeshGridSize = std::atoi(av[++i]);
	  else if (arg == "-dt")
		  dt = std::atof(av[++i]);
	  else if (arg == "--seeding-box") {
		  for (int si = 0; si < 6; ++si)
			  SeedingBox[si] = std::atof(av[++i]);
		  usingSeedingBox = true;
	  }
	  else if (arg == "--save-streamline-to-obj") {
		  objTrajectoryFileName = av[++i];
		  saveStreamlinetoFile = true;
	  }
	  else if (arg == "--save-streamline-to-vtk") {
		  vtkStreamlineFileName = av[++i];
		  saveStreamlinetoFile = true;
	  }
      else
        throw std::runtime_error("unknown cmdline argument '"+arg+"'");
    }

    // ------------------------------------------------------------------
    // create a host-side model
    // ------------------------------------------------------------------
	bool isTetVelocity = false;
	HostTetMesh hostTetMesh;
	if (velocity_vert_filename.size() > 0) {
		hostTetMesh = HostTetMesh::readDataSet(vert_filename, tet_filename, velocity_vert_filename);
		isTetVelocity = false;
		printf("#adv: load vertex velocity field from file %s", velocity_vert_filename.c_str());
	}
	else if (velocity_tet_filename.size() > 0) {
		hostTetMesh = HostTetMesh::readDataSet(vert_filename, tet_filename, "", velocity_tet_filename);
		isTetVelocity = true;
		printf("#adv: load tet velocity field from file %s", velocity_vert_filename.c_str());
	}
	else {
		hostTetMesh = HostTetMesh::createTestDataSet(testTetMeshGridSize);
		isTetVelocity = false;
		printf("#adv: using synthetic velocity field");
	}

    // ------------------------------------------------------------------
    // build the query accelerator first, before the cuda kernels
    // allocate their memory.
    // ------------------------------------------------------------------
    OptixTetQuery tetQueryAccelerator(hostTetMesh.positions.data(),
                                      hostTetMesh.positions.size(),
                                      hostTetMesh.indices.data(),
                                      hostTetMesh.indices.size());
    // by now optix should have built all its data,and released
    // whatever temp memory it has used.


    // ------------------------------------------------------------------
    // upload our own cuda data
    // ------------------------------------------------------------------
    DeviceTetMesh devMesh;
    devMesh.upload(hostTetMesh);

    // ------------------------------------------------------------------
    // now run sample advection...
    // ------------------------------------------------------------------

    // alloc particles
    Particle *d_particles = nullptr;
    cudaCheck(cudaMalloc(&d_particles,numParticles*sizeof(Particle)));
	int *d_particles_tetIDs = nullptr;
	cudaCheck(cudaMalloc(&d_particles_tetIDs, numParticles * sizeof(int)));


    // initialize with random particles
	box3d initBox;
	if (usingSeedingBox) {
		initBox.extend(vec3d(SeedingBox[0], SeedingBox[1], SeedingBox[2]));
		initBox.extend(vec3d(SeedingBox[3], SeedingBox[4], SeedingBox[5]));
	}
	else {
		initBox = hostTetMesh.worldBounds;
	}
	std::cout << "Particle seeding bounding box = " << initBox.lower << " " <<initBox.lower<< std::endl;
	cudaInitParticles(d_particles, numParticles, initBox);

    cudaCheck(cudaDeviceSynchronize());
    
    printf("Init RunTime=%lf ms\n", timer.stop());

	//Check init state

	// and iterate
    timer.start();
    for (int i=0;i<numSteps;i++) {
	  // first, compute each particle's current tet
      tetQueryAccelerator.query_sync(d_particles, d_particles_tetIDs, numParticles);

      // ... and advect
      cudaParticleAdvect(d_particles,
						 d_particles_tetIDs,
                         numParticles,
                         devMesh.d_indices,
                         devMesh.d_positions,
                         devMesh.d_velocities,
						 isTetVelocity,
                         dt);

	  const int saveInterval = 100;
	  if (saveStreamlinetoFile)
		  if ((i % (saveInterval * 5)) == 0)
			  addToTrajectories(d_particles, numParticles);

	  if ((i % saveInterval) == 0) 
		  writeParticles2VTU(i, d_particles, d_particles_tetIDs, numParticles);
    }
    std::cout << "#adv: advection steps = " << numSteps << std::endl;
    std::cout << "done ... ignoring proper cleanup for now" << std::endl;

    printf("Particle advection RunTime=%lf ms\n", timer.stop());
    
	if (objTrajectoryFileName.size() > 0) 
		saveTrajectories(objTrajectoryFileName);
	if (vtkStreamlineFileName.size() > 0) 
		writeStreamline2VTK(vtkStreamlineFileName);
	

    return 0;
  }

}
