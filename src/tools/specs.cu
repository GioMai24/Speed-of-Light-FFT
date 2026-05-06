#include <iostream>

/** @file
 * @brief Print main system GPUs properties.
 */



int main(){
	int deviceCount;
	cudaGetDeviceCount(&deviceCount);

	if (deviceCount <= 0){
		std::cout << "No GPUs detected, be afraid." << std::endl;
		return 1;
	}

	cudaDeviceProp prop;
	for (int i=0; i<deviceCount; i++){
		cudaGetDeviceProperties(&prop, i);
		
		float glob = prop.totalGlobalMem >> 30;
		float constMem = prop.totalConstMem >> 10;
		float sharedMem = prop.sharedMemPerBlock >> 10; 
		float sharedMemSM = prop.sharedMemPerMultiprocessor >> 10;
		float busWidth = prop.memoryBusWidth / 8.f;
		float l2Cache = prop.l2CacheSize >> 20;
		char dim[] = {'x', 'y', 'z'};

		// getattr
		int *value = nullptr;
		int manAcc = cudaDeviceGetAttribute(value, cudaDevAttrConcurrentManagedAccess, 0);
		int memAcc = cudaDeviceGetAttribute(value, cudaDevAttrPageableMemoryAccess, 0);
		int pagAcc = cudaDeviceGetAttribute(value, cudaDevAttrPageableMemoryAccessUsesHostPageTables, 0);
		int streamAlloc = cudaDeviceGetAttribute(value, cudaDevAttrMemoryPoolsSupported, 0);
		int memPool = cudaDeviceGetAttribute(value, cudaDevAttrMemoryPoolSupportedHandleTypes, 0);
		
		// generalities
		std::cout << "Name: " << prop.name << '\n';
		std::cout << "Compute Capability: " << prop.major << '.' << prop.minor << '\n';
		std::cout << "Concurrent kernels? " << (prop.concurrentKernels ? "yes" : "no") << '\n';
		std::cout << "Warps: " << prop.warpSize << '\n';
		std::cout << "SM count: " << prop.multiProcessorCount << '\n';
		std::cout << "Max grid size: " << '\n';
		for(int i=0; i < 3; i++){
			std::cout << ' ' << dim[i] << ": " << prop.maxGridSize[i] << '\n';
		}
		std::cout << "Max blocks x SM: " << prop.maxBlocksPerMultiProcessor << '\n';
		std::cout << "Max threads x block: " << prop.maxThreadsPerBlock << '\n';
		std::cout << "Max threads x SM: " << prop.maxThreadsPerMultiProcessor << '\n';
		
		// memory
		std::cout << "Global Memory: " << glob  << " GiB\n";
		std::cout << "Constant Memory: " << constMem << " KiB\n";
		std::cout << "Shared Memory x SM: " << sharedMemSM << " KiB\n";
		std::cout << "Shared Memory x block: " << sharedMem << " KiB\n";
		std::cout << "Registers x block: " << prop.regsPerBlock << '\n';
		std::cout << "Registers x SM: " << prop.regsPerMultiprocessor << '\n';
		std::cout << "Bus width: " << busWidth << "\n";
		std::cout << "L2 size: " << l2Cache << " MiB\n";
		std::cout << "Stream-ordered memory allocator? " << (streamAlloc ? "yes" : "no") << '\n';
		std::cout << "Memory Pool? " << (memPool ? "yes" : "no") << '\n';


		// Unified memory
		std::cout << "Concurrent Managed Access? " << (manAcc ? "yes" : "no") << '\n';
		std::cout << "Pageable Memory? " << (memAcc ? "yes" : "no") << '\n';
		std::cout << "Host Page Tables? " << (pagAcc ? "Hardware" : "Software") << "\n\n";
	}
	return 0;
}
