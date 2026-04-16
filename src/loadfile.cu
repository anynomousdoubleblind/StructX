#include "loadfile.h"
#include <fstream>
#include <sstream>
#include <iostream>


uint8_t* loadXMLFile(const std::string& filePath, size_t& fileSize) {

#if defined(DEBUG_MODE) && DEBUG_MODE == 1
    std::cout << "\033[1;33m Load File started... \033[0m\n";
#endif

#if defined(DEBUG_MODE) && DEBUG_MODE == 2
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
#endif    

    // ______________________LOAD_FILE_____________________________
    std::ifstream file(filePath, std::ios::binary | std::ios::ate);                     // Open in binary mode, seek to end
    if (!file) {                                                                        // unable to open file
        std::cerr << "\033[1;31m Error: Unable to open file: \033[0m \n" << filePath << std::endl;
        return nullptr;
    }
    
    fileSize = file.tellg();                                                            // Get file size
    file.seekg(0, std::ios::beg);                                                       // Seek back to start


    // allocate pinned memory (Host Memory)
    uint8_t* h_buffer;                                                                  // the place that we store it    
    cudaHostAlloc((void**)&h_buffer, fileSize * sizeof(uint8_t), cudaHostAllocDefault); // allocate pinned memory

    if (!h_buffer) {                                                                    // Unable to allocate pinned memory!
        std::cerr << "\033[1;31m Error: Unable to allocate pinned memory! \033[0m \n" << std::endl;
        file.close();           
        return nullptr;
    }


    // Read file content into buffer
    file.read(reinterpret_cast<char*>(h_buffer), fileSize);                             // copy from memory to host
    file.close();                                   


#if defined(DEBUG_MODE) && DEBUG_MODE == 2
    // Measure the time taken for the kernel execution
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float time_ms = 0.0f;
    cudaEventElapsedTime(&time_ms, start, stop);

    uint64_t time_ns = static_cast<uint64_t>(time_ms * 1e6);
    std::cout << "⏱️ Load XML file: " << time_ns << " ns" << std::endl;
    std::cout << "⏱️ Load XML file: " << time_ns / 1e6 << " ms" << std::endl;

    // Cleanup
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
#endif

    return h_buffer;
}