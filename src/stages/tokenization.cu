#include "tokenization.h"
#include "parser.h"       // Parser Stage (new)
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/iterator/permutation_iterator.h>
#include <thrust/sequence.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/permutation_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/logical.h>
#include <thrust/functional.h>
#include <thrust/execution_policy.h>
#include <thrust/scan.h>

#include <cub/device/device_select.cuh>
#include <thrust/gather.h>

#include <cub/cub.cuh>

// d_flags  : uint8_t*   (0 / 1 for each position, size = length)
// length   : number of elements
// h_count  : host variable that will receive the total # of 1‑bytes
// ------------------------------------------------------------------

uint32_t cpu_vcmpeq4(uint32_t a, uint32_t b) {
    uint32_t result = 0;
    for (int i = 0; i < 4; ++i) {
        uint8_t byte_a = (a >> (i * 8)) & 0xFF;
        uint8_t byte_b = (b >> (i * 8)) & 0xFF;
        if (byte_a == byte_b) {
            result |= (1 << (i * 8));
        }
    }
    return result;
}

void compute_bytemaps_host(
    uint32_t w0,
    uint32_t w1,
    uint32_t w2,
    uint32_t* out_structural,
    uint32_t* out_cdata
) {
    uint32_t prev_4_bytes     = (w0 << 8) | (w1 >> 24);  // fallback to byte_perm
    uint32_t current_4_bytes  = w1;
    uint32_t next_4_bytes     = (w1 << 8) | (w2 >> 24);
    uint32_t next_next_4_bytes = (w1 << 16) | (w2 >> 16);

    uint32_t isLess = cpu_vcmpeq4(current_4_bytes, 0x3C3C3C3C) & 0x01010101 &
                     ~(cpu_vcmpeq4(next_4_bytes, 0x2F2F2F2F) & 0x01010101);

    uint32_t isAssign = cpu_vcmpeq4(current_4_bytes, 0x3D3D3D3D) & 0x01010101 &
                        cpu_vcmpeq4(next_4_bytes, 0x22222222) & 0x01010101;

    uint32_t isSlash = cpu_vcmpeq4(current_4_bytes, 0x2F2F2F2F) & 0x01010101 &
                      ((cpu_vcmpeq4(next_4_bytes, 0x3E3E3E3E) |
                        cpu_vcmpeq4(prev_4_bytes, 0x3C3C3C3C)) & 0x01010101);

    uint32_t isCDATA_open = cpu_vcmpeq4(current_4_bytes, 0x3C3C3C3C) & 0x01010101 &
                            cpu_vcmpeq4(next_4_bytes, 0x21212121) & 0x01010101 &
                            cpu_vcmpeq4(next_next_4_bytes, 0x5B5B5B5B) & 0x01010101;

    uint32_t isCDATA_close = cpu_vcmpeq4(current_4_bytes, 0x5D5D5D5D) & 0x0101010 &
                             cpu_vcmpeq4(next_4_bytes, 0x5D5D5D5D) & 0x03030303 &
                             cpu_vcmpeq4(next_next_4_bytes, 0x3E3E3E3E) & 0x03030303;

    *out_structural = isLess | isSlash | isAssign;
    *out_cdata = isCDATA_open | isCDATA_close;
}


struct BytemapFunctorSafe {
    __device__
    thrust::tuple<uint32_t, uint32_t> operator()(thrust::tuple<uint32_t, uint32_t, uint32_t> t) const {
        uint32_t w0 = thrust::get<0>(t);
        uint32_t w1 = thrust::get<1>(t);
        uint32_t w2 = thrust::get<2>(t);

        uint32_t prev_4_bytes = __byte_perm(w0, w1, 0x6543);
        uint32_t current_4_bytes = w1;
        uint32_t next_4_bytes = __byte_perm(w1, w2, 0x4321);
        uint32_t next_next_4_bytes = __byte_perm(w1, w2, 0x5432);

        uint32_t isLess = (__vcmpeq4(current_4_bytes, 0x3C3C3C3C) & 0x01010101) &
                          ((~__vcmpeq4(next_4_bytes, 0x2F2F2F2F)) & 0x01010101);

        uint32_t isAssign = (__vcmpeq4(current_4_bytes, 0x3D3D3D3D) & 0x01010101) &
                            (__vcmpeq4(next_4_bytes, 0x22222222) & 0x01010101);

        uint32_t isSlash = (__vcmpeq4(current_4_bytes, 0x2F2F2F2F) & 0x01010101) &
                           ((__vcmpeq4(next_4_bytes, 0x3E3E3E3E) |
                             __vcmpeq4(prev_4_bytes, 0x3C3C3C3C)) & 0x01010101);

        uint32_t isCDATASection = (
            (__vcmpeq4(current_4_bytes, 0x3C3C3C3C) & 0x01010101) &
            (__vcmpeq4(next_4_bytes, 0x21212121) & 0x01010101) &
            (__vcmpeq4(next_next_4_bytes, 0x5B5B5B5B) & 0x01010101)
        ) | (
            (__vcmpeq4(current_4_bytes, 0x5D5D5D5D) & 0x03030303) &
            (__vcmpeq4(next_4_bytes, 0x5D5D5D5D) & 0x03030303) &
            (__vcmpeq4(next_next_4_bytes, 0x3E3E3E3E) & 0x03030303)
        );

        return thrust::make_tuple(isLess | isSlash | isAssign, isCDATASection);
    }
};


void create_bytemaps_thrust_manual_edge(
    const uint32_t* d_xmlContent,
    size_t length_uint32,
    uint32_t* isStructural,
    uint32_t* isCDATA
) {
    // Handle edges manually: i = 0 and i = length_uint32 - 1
    uint32_t h_w0 = 0, h_w1 = 0, h_w2 = 0;

    // --------- i = 0 ----------
    cudaMemcpy(&h_w1, d_xmlContent, sizeof(uint32_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_w2, d_xmlContent + 1, sizeof(uint32_t), cudaMemcpyDeviceToHost);

    uint32_t s0, c0;
    compute_bytemaps_host(h_w0, h_w1, h_w2, &s0, &c0);

    cudaMemcpy(isStructural, &s0, sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(isCDATA, &c0, sizeof(uint32_t), cudaMemcpyHostToDevice);

    // --------- i = length-1 ----------
    cudaMemcpy(&h_w0, d_xmlContent + length_uint32 - 2, sizeof(uint32_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_w1, d_xmlContent + length_uint32 - 1, sizeof(uint32_t), cudaMemcpyDeviceToHost);
    h_w2 = 0;

    uint32_t sN, cN;
    compute_bytemaps_host(h_w0, h_w1, h_w2, &sN, &cN);

    cudaMemcpy(isStructural + length_uint32 - 1, &sN, sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(isCDATA + length_uint32 - 1, &cN, sizeof(uint32_t), cudaMemcpyHostToDevice);

    // ---------- middle range ----------
    auto w0_iter = thrust::device_pointer_cast((uint32_t*)d_xmlContent + 0);
    auto w1_iter = thrust::device_pointer_cast((uint32_t*)d_xmlContent + 1);
    auto w2_iter = thrust::device_pointer_cast((uint32_t*)d_xmlContent + 2);

    auto input_zip = thrust::make_zip_iterator(thrust::make_tuple(w0_iter, w1_iter, w2_iter));
    auto output_zip = thrust::make_zip_iterator(thrust::make_tuple(
        thrust::device_pointer_cast(isStructural + 1),
        thrust::device_pointer_cast(isCDATA + 1)
    ));

    thrust::transform(
        thrust::cuda::par,
        input_zip, input_zip + (length_uint32 - 2),
        output_zip,
        BytemapFunctorSafe()
    );
}



// Example kernel (placeholder)
__global__ void create_bytemaps(const uint32_t* d_xmlContent, size_t length, size_t length_uint32, uint32_t* isStructural, uint32_t* isCDATA) {
    // Calculate global thread index
    size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;

    for(uint32_t i=index; i<length_uint32 && i < length; i+=stride) {
        // Tokenization logic here
        // For example, identify start and end tags, attributes, etc.
        // This is a placeholder for the actual tokenization logic.
        // You can use d_xmlContent[i] to access the XML content.
        uint32_t w0 = (i > 0) ? d_xmlContent[i - 1] : 0;
        uint32_t w1 = d_xmlContent[i];
        uint32_t w2 = (i + 1 < length_uint32 && i + 1 < length) ? d_xmlContent[i + 1] : 0;

        // Sliding windows using byte_perm
        uint32_t prev_4_bytes = __byte_perm(w0, w1, 0x6543);    // Permute the bytes to create a 4-byte chunk
        uint32_t current_4_bytes = w1;                          // Current 4-byte chunk
        uint32_t next_4_bytes = __byte_perm(w1, w2, 0x4321);    // Permute the bytes to create a 4-byte chunk
        uint32_t next_next_4_bytes = __byte_perm(w1, w2, 0x5432);// Permute the bytes to create a 4-byte chunk

        // to-do: handle CDATA [[CDATA[]]] and comments <!-- comment -->
        // Process the 4-byte chunk
        // Check for less than sign (<)
        // Skip less than sign before slashesh (/)
        uint32_t isLess = (__vcmpeq4(current_4_bytes, 0x3C3C3C3C) & 0x01010101)& ((~__vcmpeq4(next_4_bytes, 0x2F2F2F2F)) & 0x01010101);

        // Check for greater than sign (>)
        // uint32_t isGreater = __vcmpeq4(current_4_bytes, 0x3E3E3E3E) & 0x01010101; 
        
        
        // Check for assignment operator. 
        // Tag values cannot contain double quotes.
        // Valid assignments should occur within tags and must precede any double quotes to maintain structural correctness.
        uint32_t isAssign = (__vcmpeq4(current_4_bytes, 0x3D3D3D3D) & 0x01010101) & ((__vcmpeq4(next_4_bytes, 0x22222222)) & 0x01010101);


        // Check for slash (/)
        // Keep slash if it is after a less than sign (<)
        // Keep slash if it is before a greater than sign (>)
        uint32_t isSlash = (__vcmpeq4(current_4_bytes, 0x2F2F2F2F) & 0x01010101) & 
                            (   
                                (
                                    (__vcmpeq4(next_4_bytes, 0x3E3E3E3E)) | 
                                    (__vcmpeq4(prev_4_bytes, 0x3C3C3C3C))
                                ) & 0x01010101
                            );


        // Check for CDATA section
        uint32_t isCDATASection = (
                                (__vcmpeq4(current_4_bytes,   0x3C3C3C3C) & 0x01010101) &  // < < < <
                                (__vcmpeq4(next_4_bytes,      0x21212121) & 0x01010101) &  // ! ! ! !
                                (__vcmpeq4(next_next_4_bytes, 0x5B5B5B5B) & 0x01010101)    // [ [ [ [ 
                                ) | ( 
                                (__vcmpeq4(current_4_bytes,   0x5D5D5D5D) & 0x01010101) &  // ] ] ] ]
                                (__vcmpeq4(next_4_bytes,      0x5D5D5D5D) & 0x01010101) &  // ] ] ] ]
                                (__vcmpeq4(next_next_4_bytes, 0x3E3E3E3E) & 0x01010101)    // > > > > 
                                );

        // isGreaterSlash[i] = isGreater | isSlash;
        isCDATA[i] = isCDATASection; // Store CDATA section flag
        isStructural[i] = isLess | isSlash | isAssign;
    }
}


__global__ void mask_structural_inside_cdata(
    uint32_t* isStructural,  // isLessGreaterSlashTokens
    uint32_t* isCDATA,       // inclusive XOR scan of CDATA open/close flags (per-byte parity)
    size_t length,           // length of the input data
    size_t length_uint32     // length in uint32_t units
) {
    size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;

    for (size_t i = index; i < length_uint32 && i < length; i += stride) {
        uint32_t c = isCDATA[i];
        uint32_t s = isStructural[i];
        #pragma unroll
        for (int lane = 0; lane < 4; ++lane) {
            uint8_t parity = (uint8_t)((c >> (lane * 8)) & 0xFFu);
            if (parity & 1u) {
                s &= ~(0xFFu << (lane * 8));
            }
        }
        isStructural[i] = s;
    }
}

// __global__ void scatter_tokens_delta(
//     const uint32_t* __restrict__ scanned_flags,  // after inclusive scan
//     const uint8_t* __restrict__ content,        // xml data
//     uint32_t* __restrict__ out_indices,         // output: indices
//     uint8_t* __restrict__ out_values,           // output: actual characters
//     size_t length                               // length of the input data
// ) {
//     uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
//     uint32_t stride = blockDim.x * gridDim.x;
//     for (size_t i=tid; i<length; i+=stride) {
//         uint32_t curr = scanned_flags[i];
//         uint32_t prev = (i > 0) ? scanned_flags[i - 1] : 0;
//         if (curr - prev == 1) {                 // This was originally a 1 in flags
//             uint32_t pos = curr - 1;            // because scan[i] == total before i+1
//             out_indices[pos] = i;
//             out_values[pos] = content[i];
//         }
//     }   
// }


// Extract token indices and values from the flags
// void scatter(
//     uint8_t* d_flags,               // isLessGreaterSlashTokens
//     uint8_t* d_content,             // d_xmlContent
//     uint32_t* d_token_indices,      // output: indices
//     uint8_t* d_token_values,        // output: actual characters
//     size_t length
// ) {
//     thrust::inclusive_scan(
//         thrust::cuda::par,
//         thrust::device_pointer_cast(d_flags),
//         thrust::device_pointer_cast(d_flags + length),
//         thrust::device_pointer_cast(d_flags)  // in-place
//     );
//     // Launch Kernel to scatter tokens 
//     size_t gridSize = (length + BLOCKSIZE - 1) / BLOCKSIZE;
//     scatter_tokens_delta<<<gridSize, BLOCKSIZE>>>(
//         d_flags,
//         d_content,
//         d_token_indices,
//         d_token_values,
//         length
//     );
//     cudaDeviceSynchronize();
// }


void scatter_cub_flagged(
    uint8_t*   d_flags,                     // 0 / 1 bytes (length = `length`)
    uint8_t*   d_content,                   // XML text (bytes)
    uint32_t*  d_token_indices,             // OUT: matching indices  (size >= length)
    uint8_t*   d_token_values,              // OUT: matching bytes    (size >= length)
    size_t     length,                      // total bytes
    uint32_t*  d_selected_cnt,              // OUT: device pointer to uint32_t counter
    void**     d_temp_storage,              // IN/OUT: temp buffer (may be nullptr first)
    size_t*    temp_bytes )                 // IN/OUT: size of temp buffer
{
    // Step 1: Generate [0,1,2,…] virtually   (no device allocation needed)
    auto counting_begin = thrust::counting_iterator<uint32_t>(0);

    // Step 2 - A:  Query CUB temp buffer size - Run CUB DeviceSelect::Flagged on the indices
    /*
        - CUB (pronounced “cube”) is CUDA Utilities Backend
        - DeviceSelect::Flagged is CUB’s GPU‑parallel routine that selects (copies) the items whose associated flag is “true/one”.
        - The algorithm avoids branch divergence and global‑memory holes, giving significantly higher throughput than a naïve thrust::copy_if for large arrays (hundreds of MB – GB).
    */
    /*
        What happens under the hood?
            CUB launches a well‑tuned kernel that
            – loads data into shared memory in tiles
            – performs a warp‑wide ballot on the flags
            – writes only the selected items to global memory using prefix‑sum offsets.
    */
    cub::DeviceSelect::Flagged(
        *d_temp_storage,                /*temp_storage   =*/ 
        *temp_bytes,                    /*temp_storage_bytes=*/ 
        counting_begin,                 /*d_in           =*/        // virtual indices              --> pointer / iterator to the items you want to sieve
        d_flags,                        /*d_flags        =*/        // 0 / 1                        --> pointer to the flags (0 / 1)
        d_token_indices,                /*d_out          =*/        // keepers                      --> pointer to the output buffer (indices)  
        d_selected_cnt,                 /*d_num_selected =*/        // count of selected items      --> pointer to the output buffer (count of selected items)
        length,                         /*num_items      =*/        // total number of items        --> number of items in the input buffer
        0 );                            /*stream         =*/ 



    // Step 2 - B:  Allocate temporary storage if not already allocated
    // If this was the “query” call, allocate the buffer and launch again
    if (*d_temp_storage == nullptr) cudaMalloc(&(*d_temp_storage), *temp_bytes);

    // Step 2 - C:  Actual CUB selection - Perform selection
    cub::DeviceSelect::Flagged(
        *d_temp_storage, *temp_bytes,
        counting_begin, d_flags,
        d_token_indices, d_selected_cnt,  // again real device pointer
        length
    );
    
    

    //------------------------------------------------------------------
    // Step 3:  Gather the corresponding characters
    /*
        ‑ the index list (d_token_indices) is already compact (N_selected)
        ‑ copy d_content[index[i]] -> d_token_values[i]
    */
    uint32_t h_selected = 0;
    cudaMemcpy(&h_selected, d_selected_cnt, sizeof(uint32_t), cudaMemcpyDeviceToHost);

    thrust::gather(
        thrust::device_pointer_cast(d_token_indices),
        thrust::device_pointer_cast(d_token_indices) + h_selected,
        thrust::device_pointer_cast(d_content),
        thrust::device_pointer_cast(d_token_values) );
}

// found:5668287
// Extract token indices and values from the flags
void scatter(
    uint8_t* d_flags,               // isLessGreaterSlashTokens
    uint8_t* d_content,             // d_xmlContent
    uint32_t* d_token_indices,      // output: indices
    uint8_t* d_token_values,        // output: actual characters
    size_t length
) {

    void*  d_temp   = nullptr;
    size_t temp_sz  = 0;
    uint32_t* d_sel_cnt;
    cudaMalloc(&d_sel_cnt, sizeof(uint32_t));

    // scatter
    scatter_cub_flagged(d_flags, d_content,                 // input: flags and XML content
                        d_token_indices, d_token_values,    // output: indices and values
                        length,                             // total bytes in XML
                        d_sel_cnt,                       // output: selected count   
                        &d_temp, &temp_sz);
    
}

void stage2_tokenization(
    uint32_t* d_xmlContent,
    size_t padded_length,
    uint32_t** d_token_indices,     // Output: token positions
    uint8_t** d_token_values,       // Output: actual token characters
    size_t* tokens_count            // Output: how many tokens
) {
    // Launch kernel
    #if defined(DEBUG_MODE) && DEBUG_MODE == 1
        std::cout << "\033[1;34m Tokenization started... \033[0m\n";
    #endif

    /*_______________________________________STEP1_Create_Byte_Maps_______________________________________*/
    #if defined(DEBUG_MODE) && DEBUG_MODE == 1
        std::cout << "\033[1;34m Tokenization step 1 - start... \033[0m\n";
    #endif
    #if defined(DEBUG_MODE) && DEBUG_MODE == 2
        // Measure the time taken for the kernel execution
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);
    #endif
    
    // Define the number of threads per block
    const uint32_t length_uint32 = (padded_length + 3) / 4;
    const uint32_t number_blocks_uint32 = (length_uint32 + BLOCKSIZE - 1) / BLOCKSIZE;
 

    // Allocate memory for tokens on the device
    uint8_t* isStructural;
    cudaMallocAsync((void**)&isStructural, padded_length * sizeof(uint8_t) * 2, 0);
    uint8_t* isCDATA = isStructural + padded_length; // Use the second half of the allocated memory for CDATA
    // cudaMallocAsync((void**)&isCDATA, padded_length * sizeof(uint8_t), 0);

    create_bytemaps<<<number_blocks_uint32, BLOCKSIZE>>>(d_xmlContent, padded_length, length_uint32, (uint32_t*) isStructural, (uint32_t*) isCDATA);
    cudaDeviceSynchronize();

    // it is a bit faster than the kenrel implementation (13.8958 ms vs 13.2153 ms)
    // create_bytemaps_thrust_manual_edge(
    //     d_xmlContent,     // const uint32_t* → your input XML buffer in uint32_t*
    //     length_uint32,    // size_t → number of uint32_t elements in d_xmlContent
    //     (uint32_t*) isStructural,  // uint32_t* → output buffer for structural flags
    //     (uint32_t*) isCDATA        // uint32_t* → output buffer for CDATA flags
    // );

    #if defined(DEBUG_MODE) && DEBUG_MODE == 5
        printGpuMemoryUsage("After Bytemaps Creation");
    #endif
    #if defined(DEBUG_MODE) && DEBUG_MODE == 2
        // Measure the time taken for the kernel execution
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float time_ms = 0.0f;
        cudaEventElapsedTime(&time_ms, start, stop);
        
        uint64_t time_ns = static_cast<uint64_t>(time_ms * 1e6);
        std::cout << "⏱️ STEP1 execution time (create_bytemaps): " << time_ns << " ns" << std::endl;
        std::cout << "⏱️ STEP1 execution time (create_bytemaps): " << time_ns / 1e6 << " ms" << std::endl;

        // Cleanup
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    #endif
    #if defined(DEBUG_MODE) && DEBUG_MODE == 1
        // Print Results (for debugging)
        print_token_array_as_bytes("Structural",   (u_int32_t*) isStructural, length_uint32);
        print_token_array_as_bytes("cdata",   (u_int32_t*) isCDATA, length_uint32);
    #endif
    #if defined(DEBUG_MODE) && DEBUG_MODE == 2
        // Measure the time taken for the kernel execution
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);
    #endif

    /* _____________________________________STEP2_CDATA_______________________________________*/
    // Wrap raw device pointer into a thrust device pointer
    bool has_cdata = count_ones_cub(isCDATA, padded_length); //much faster than thrust::count
    // bool has_cdata = thrust::count(
    //     thrust::cuda::par,
    //     cdata_ptr,
    //     cdata_ptr + padded_length,
    //     1 // Count the number of '1's in the isStructural array
    // );
    #if defined(DEBUG_MODE) && DEBUG_MODE == 4
        std::cout << "has_cdata: " << has_cdata << std::endl;
    #endif

    if (has_cdata) {
        thrust::device_ptr<uint8_t> cdata_ptr(isCDATA);
        thrust::inclusive_scan(
            thrust::cuda::par,
            cdata_ptr,
            cdata_ptr + padded_length,
            cdata_ptr,
            thrust::bit_xor<uint8_t>());
    
        // Run kernel function to mask structural tokens inside CDATA
        mask_structural_inside_cdata<<<number_blocks_uint32, BLOCKSIZE>>>(
            (uint32_t*) isStructural,
            (uint32_t*) isCDATA,
            padded_length,
            length_uint32
        );

        cudaDeviceSynchronize();
        #if defined(DEBUG_MODE) && DEBUG_MODE == 1
            // Print Results (for debugging)
            print_token_array_as_bytes("cdata (scanned)",   (u_int32_t*) isCDATA, length_uint32);
            print_token_array_as_bytes("isStructural (masked)",   (u_int32_t*) isStructural, length_uint32);
        #endif

    } 
    
    #if defined(DEBUG_MODE) && DEBUG_MODE == 2
        // Measure the time taken for the kernel execution
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        time_ms = 0.0f;
        cudaEventElapsedTime(&time_ms, start, stop);
        
        time_ns = static_cast<uint64_t>(time_ms * 1e6);
        std::cout << "⏱️ STEP2(CDATA) execution time (thrust::reduce): " << time_ns << " ns" << std::endl;
        std::cout << "⏱️ STEP2(CDATA) execution time (thrust::reduce): " << time_ns / 1e6 << " ms" << std::endl;

        // Cleanup
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    #endif

    #if defined(DEBUG_MODE) && DEBUG_MODE == 5
        printGpuMemoryUsage("After CDATA Creation");
    #endif

    /*_______________________________________STEP2_Gather_______________________________________*/
    /*_______________________________________STEP2(A)_reduce_______________________________________*/
    #if defined(DEBUG_MODE) && DEBUG_MODE == 1 
        std::cout << "\033[1;34m Tokenization step 2 - start... \033[0m\n"; 
    #endif
    #if defined(DEBUG_MODE) && DEBUG_MODE == 2
        // Measure the time taken for the kernel execution
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);
    #endif
    
    // Sum all the 1s in the vector
    // *tokens_count = thrust::count(
    //     thrust::cuda::par,
    //     thrust::device_pointer_cast(isStructural),
    //     thrust::device_pointer_cast(isStructural + padded_length),
    //     1 // Count the number of '1's in the isStructural array
    // );

    *tokens_count = count_ones_cub(isStructural, padded_length); //much faster than thrust::count


    // cout << "tokens_count: " << *tokens_count << endl;
    // Validation
    if (*tokens_count == 0 || *tokens_count == 1) {
        std::cerr << "Error: No (or 1) tokens found in the XML content." << std::endl;
        exit(0);
    }


    #if defined(DEBUG_MODE) && DEBUG_MODE == 2
        // Measure the time taken for the kernel execution
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        time_ms = 0.0f;
        cudaEventElapsedTime(&time_ms, start, stop);
        
        time_ns = static_cast<uint64_t>(time_ms * 1e6);
        std::cout << "⏱️ count execution time (thrust::reduce): " << time_ns << " ns" << std::endl;
        std::cout << "⏱️ count execution time (thrust::reduce): " << time_ns / 1e6 << " ms" << std::endl;

        // Cleanup
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    #endif
    #if defined(DEBUG_MODE) && DEBUG_MODE == 1
        std::cout << "Count of Less/Slash tokens: " << *tokens_count << std::endl;
    #endif

    /*_______________________________________STEP2(B):_Extract_Structural_Indices_&_Values_______________________________________*/

    #if defined(DEBUG_MODE) && DEBUG_MODE == 1
        std::cout << "\033[1;34m Tokenization step 2b - start... \033[0m\n";
    #endif
    #if defined(DEBUG_MODE) && DEBUG_MODE == 2
        // Measure the time taken for the kernel execution
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);
    #endif

    // Allocate memory for token indices and values
    cudaMalloc((void**)d_token_indices, (*tokens_count) * sizeof(uint32_t));
    cudaMalloc((void**)d_token_values,  (*tokens_count) * sizeof(uint8_t));


    // Launch Kernel to scatter tokens 
    scatter(isStructural, (uint8_t*) d_xmlContent, *d_token_indices, *d_token_values, padded_length);
    #if defined(DEBUG_MODE) && DEBUG_MODE == 5
        printGpuMemoryUsage("After Scatter");
    #endif


    cudaFreeAsync(isStructural, 0); // Free the allocated memory for isStructural
    
    #if defined(DEBUG_MODE) && DEBUG_MODE == 2
        // Measure the time taken for the kernel execution
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        time_ms = 0.0f;
        cudaEventElapsedTime(&time_ms, start, stop);
        
        time_ns = static_cast<uint64_t>(time_ms * 1e6);
        std::cout << "⏱️ scatter execution time (scatter): " << time_ns << " ns" << std::endl;
        std::cout << "⏱️ scatter execution time (scatter): " << time_ns / 1e6 << " ms" << std::endl;

        // Cleanup
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    #endif
    #if defined(DEBUG_MODE) && DEBUG_MODE == 1
        // Print Results (for debugging)
        print_uint32_indices("Final Tokens", *d_token_indices, *tokens_count);
        print_token_info("Final Tokens", *d_token_indices, *d_token_values, *tokens_count);
        std::cout << "\033[1;34m Tokenization step 2b - end... \033[0m\n";
    #endif
    // TODO: Copy back to CPU if needed

    // cudaFree(isStructural); // Free the allocated memory for isStructural
}
