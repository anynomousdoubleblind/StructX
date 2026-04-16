#include "standard_parse.h"         // Include the standard parse header
#include "stages/validation.h"      // Include the validator function
#include "stages/tokenization.h"    // Include the tokenization header
#include "stages/parser.h"          // Include the parser header
#include "stages/query.h"           // Include the query header
#include "stages/query_host.h"


void standard_parse(uint8_t* h_xmlContent, size_t length, string xpath_query) {
    if (length == 0 || h_xmlContent == nullptr) {
        std::cerr << "\033[1;31m Error: Invalid XML content. \033[0m" << std::endl;
        return;
    }

    // init - Calculate padding to align the buffer size to the nearest multiple of 4 bytes for optimal GPU performance.
    int remainder = length % 4;    
    int padding = (4 - remainder) & 3;                                                                  // Padding bytes needed.
    uint64_t padded_length = length + padding;
    uint64_t size_32 = padded_length / 4;

    // Allocate input memory on GPU
    uint8_t* d_xmlContent;
    cudaMalloc((void**)&d_xmlContent, (length + padding) * sizeof(uint8_t));
    cudaMemset(d_xmlContent, 0, (length + padding) * sizeof(uint8_t));
    #if defined(DEBUG_MODE) && DEBUG_MODE == 4
        cout << "#characters: " << length << " bytes" << endl;
    #endif
    // Transfer XML content to GPU
    #if defined(DEBUG_MODE) && DEBUG_MODE == 1
        std::cout << "\033[1;35m Transferring XML content to GPU... \033[0m\n";
        std::cout << "Padded Length: " << padded_length << " bytes" << std::endl;
        std::cout << "Original Length: " << length << " bytes" << std::endl;
        std::cout << "Padding: " << padding << " bytes" << std::endl;
        std::cout << "Size in 32-bit words: " << size_32 << " (total " << padded_length / 4 << " 32-bit words)" << std::endl;
    #endif
    #if defined(DEBUG_MODE) && DEBUG_MODE == 2
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);
    #endif
    #if defined(DEBUG_MODE) && DEBUG_MODE == 3
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);
    #endif
    
    cudaMemcpy(d_xmlContent, h_xmlContent, length * sizeof(uint8_t), cudaMemcpyHostToDevice);

    #if defined(DEBUG_MODE) && DEBUG_MODE == 2
        // Measure the time taken for the kernel execution
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float time_ms = 0.0f;
        cudaEventElapsedTime(&time_ms, start, stop);
        
        uint64_t time_ns = static_cast<uint64_t>(time_ms * 1e6);
        std::cout << "⏱️ Host to Device execution time: " << time_ns << " ns" << std::endl;
        std::cout << "⏱️ Host to Device execution time: " << time_ns / 1e6 << " ms" << std::endl;

        // Cleanup
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    #endif
    #if defined(DEBUG_MODE) && DEBUG_MODE == 3
        // Measure the time taken for the kernel execution
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float time_ms = 0.0f;
        cudaEventElapsedTime(&time_ms, start, stop);
        
        uint64_t time_ns_h2d = static_cast<uint64_t>(time_ms * 1e6);
        std::cout << "⏱️ Host to Device execution time: " << time_ns_h2d << " ns" << std::endl;
        std::cout << "⏱️ Host to Device execution time: " << time_ns_h2d / 1e6 << " ms" << std::endl;

        // Cleanup
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    #endif
    std::cout << "\033[1;35m Host to Device completed. \033[0m\n";
    #if defined(DEBUG_MODE) && DEBUG_MODE == 2
        // Measure the time taken for the kernel execution
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);
    #endif
    #if defined(DEBUG_MODE) && DEBUG_MODE == 3
        // Measure the time taken for the kernel execution
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);
    #endif

    // Stage 1 - Validator
    bool isValidUTF8 = stage1_UTF8Validator(reinterpret_cast<uint32_t *> (d_xmlContent), size_32);
    if (!isValidUTF8) {
        std::cerr << "\033[1;31m Error: Invalid UTF-8 encoding. \033[0m" << std::endl;
        return;
    }

    #if defined(DEBUG_MODE) && DEBUG_MODE == 3
        // Measure the time taken for the kernel execution
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        time_ms = 0.0f;
        cudaEventElapsedTime(&time_ms, start, stop);
        
        uint64_t time_ns_validation = static_cast<uint64_t>(time_ms * 1e6);
        std::cout << "⏱️ Validation execution time: " << time_ns_validation << " ns" << std::endl;
        std::cout << "⏱️ Validation execution time: " << time_ns_validation / 1e6 << " ms" << std::endl;

        // Cleanup
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    #endif
    #if defined(DEBUG_MODE) && DEBUG_MODE == 2
        // Measure the time taken for the kernel execution
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        time_ms = 0.0f;
        cudaEventElapsedTime(&time_ms, start, stop);
        
        time_ns = static_cast<uint64_t>(time_ms * 1e6);
        std::cout << "⏱️ Validation execution time: " << time_ns << " ns" << std::endl;
        std::cout << "⏱️ Validation execution time: " << time_ns / 1e6 << " ms" << std::endl;

        // Cleanup
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    #endif
    std::cout << "\033[1;32m Validation completed. \033[0m \n";
    #if defined(DEBUG_MODE) && DEBUG_MODE == 3
        // Measure the time taken for the kernel execution
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);
    #endif

    #if defined(DEBUG_MODE) && DEBUG_MODE == 5
        printGpuMemoryUsage("Before Tokenization");
    #endif
    
    // Stage 2 - Tokenization
    uint32_t* d_token_indices = nullptr;
    uint8_t* d_token_values = nullptr;
    size_t tokens_count = 0;
    stage2_tokenization(reinterpret_cast<uint32_t *> (d_xmlContent), 
                        padded_length,
                        &d_token_indices,
                        &d_token_values,
                        &tokens_count);  // Pass the required parameters for tokenization
             
    #if defined(DEBUG_MODE) && DEBUG_MODE == 5
        printGpuMemoryUsage("After Tokenization");
    #endif
    #if defined(DEBUG_MODE) && DEBUG_MODE == 3
        // Measure the time taken for the kernel execution
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        time_ms = 0.0f;
        cudaEventElapsedTime(&time_ms, start, stop);
        
        uint64_t time_ns_tokenization = static_cast<uint64_t>(time_ms * 1e6);
        std::cout << "⏱️ Tokenization execution time: " << time_ns_tokenization << " ns" << std::endl;
        std::cout << "⏱️ Tokenization execution time: " << time_ns_tokenization / 1e6 << " ms" << std::endl;
        std::cout << "⏱️ Tokens Count: " << tokens_count << std::endl;
        // Cleanup
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    #endif

    #if defined(DEBUG_MODE) && DEBUG_MODE == 4
        cout << "#tokens: " << tokens_count << " bytes" << endl;
    #endif
    std::cout << "\033[1;34m Tokenization completed. \033[0m\n";
    #if defined(DEBUG_MODE) && DEBUG_MODE == 3
        // Measure the time taken for the kernel execution
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);
    #endif
    if (tokens_count == 0 || d_token_indices == nullptr || d_token_values == nullptr) {
        std::cerr << "\033[1;31m Error: Tokenization failed or no tokens found. \033[0m" << std::endl;
        cudaFree(d_xmlContent);
        return;
    }

    // Stage 3 - Parser
    uint32_t* pair_pos = nullptr;
    uint32_t* depth = nullptr;
    int validation_error = 0;
    stage3_parse(
        d_token_indices,
        d_token_values,
        tokens_count,
        d_xmlContent,
        &depth,
        &pair_pos,
        &validation_error);


    #if defined(DEBUG_MODE) && DEBUG_MODE == 3
        // Measure the time taken for the kernel execution
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        time_ms = 0.0f;
        cudaEventElapsedTime(&time_ms, start, stop);
        
        uint64_t time_ns_parser = static_cast<uint64_t>(time_ms * 1e6);
        std::cout << "⏱️ Parser execution time: " << time_ns_parser << " ns" << std::endl;
        std::cout << "⏱️ Parser execution time: " << time_ns_parser / 1e6 << " ms" << std::endl;

        // Cleanup
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    #endif
    #if defined(DEBUG_MODE) && DEBUG_MODE == 5
        printGpuMemoryUsage("After Parser");
    #endif
    std::cout << "\033[1;36m Parsing completed. \033[0m \n";
    #if defined(DEBUG_MODE) && DEBUG_MODE == 3
        // Measure the time taken for the kernel execution
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);
    #endif
    #if defined(DEBUG_MODE) && DEBUG_MODE == 2
        // Measure the time taken for the kernel execution
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);
    #endif
    
    #if defined(DEBUG_MODE) && DEBUG_MODE == 3
        // Measure the time taken for the kernel execution
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);
    #endif

    if (validation_error != 0) {
        std::cerr << "\033[1;31m Error: Invalid XML input detected during parser validation. \033[0m\n";
        cudaFree(d_xmlContent);
        if (pair_pos != nullptr) cudaFree(pair_pos);
        if (depth != nullptr) cudaFree(depth);
        return;
    }

    // Stage 4 - Xpath Query
    uint32_t* d_matches = nullptr;
    size_t matched_tokens_size = 0;
    stage4_xpath(
        d_token_indices,
        d_token_values,
        tokens_count,
        length,
        d_xmlContent,
        &pair_pos,
        &depth,
        reinterpret_cast<const uint8_t*>(xpath_query.c_str()),  // Pass the XPath query as a byte array
        &d_matches,
        &matched_tokens_size
    );  // Pass the output array for matches

    #if defined(DEBUG_MODE) && DEBUG_MODE == 5
        printGpuMemoryUsage("After Query");
    #endif
    #if defined(DEBUG_MODE) && DEBUG_MODE == 3
        // Measure the time taken for the kernel execution
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        time_ms = 0.0f;
        cudaEventElapsedTime(&time_ms, start, stop);
        
        uint64_t time_ns_q = static_cast<uint64_t>(time_ms * 1e6);
        std::cout << "⏱️ Query Computation time: " << time_ns_q << " ns" << std::endl;
        std::cout << "⏱️ Query Computation time: " << time_ns_q / 1e6 << " ms" << std::endl;

        // Cleanup
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    #endif
    // Now h_structural and h_pair_pos contain the parsed tree data
    std::cout << "\033[1;32m Query completed successfully. \033[0m\n";
    #if defined(DEBUG_MODE) && DEBUG_MODE == 3
        // Measure the time taken for the kernel execution
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);
    #endif
    #if defined(DEBUG_MODE) && DEBUG_MODE == 2
        // Measure the time taken for the kernel execution
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);
    #endif

    // Process the output array to get the matches
    // return h_matches to cpu 
    uint32_t* h_matches = nullptr;
    cudaMallocHost((void**)&h_matches, sizeof(uint32_t) * matched_tokens_size); // Allocate pinned memory for structural array
    cudaMemcpy(h_matches, d_matches, sizeof(uint32_t) * matched_tokens_size, cudaMemcpyDeviceToHost);

    #if defined(DEBUG_MODE) && DEBUG_MODE == 2
        // Measure the time taken for the kernel execution
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        time_ms = 0.0f;
        cudaEventElapsedTime(&time_ms, start, stop);
        
        uint64_t time_ns_d2hq = static_cast<uint64_t>(time_ms * 1e6);
        std::cout << "⏱️ Query Device to Host execution time: " << time_ns_d2hq << " ns" << std::endl;
        std::cout << "⏱️ Query Device to Host execution time: " << time_ns_d2hq / 1e6 << " ms" << std::endl;

        // Cleanup
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    #endif
    #if defined(DEBUG_MODE) && DEBUG_MODE == 3
        // Measure the time taken for the kernel execution
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        time_ms = 0.0f;
        cudaEventElapsedTime(&time_ms, start, stop);
        
        uint64_t time_ns_d2hq = static_cast<uint64_t>(time_ms * 1e6);
        std::cout << "⏱️ Query Device to Host execution time: " << time_ns_d2hq << " ns" << std::endl;
        std::cout << "⏱️ Query Device to Host execution time: " << time_ns_d2hq / 1e6 << " ms" << std::endl;

        // Cleanup
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    #endif
    std::cout << "\033[1;32m D2H completed successfully. \033[0m\n";


    #if defined(DEBUG_MODE) && DEBUG_MODE == 3
        // Total Time
        std::cout << "⏱️ Total Parsing (including d2h) time: " << (time_ns_parser + time_ns_validation + time_ns_tokenization + time_ns_h2d + time_ns_d2hq + time_ns_q) / 1e6 << " ms" << std::endl;  
    #endif


    print_xml_matches(h_xmlContent, h_matches, matched_tokens_size);



    // Free GPU Memory
    cudaFree(d_xmlContent);

    #if defined(DEBUG_MODE) && DEBUG_MODE == 5
        printGpuMemoryUsage("End of Standard Parse");
    #endif

    // return Parsed Tree
    
}