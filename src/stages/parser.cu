#include "parser.h"
#include <iostream>
#include <cuda_runtime.h>
#include <cub/device/device_scan.cuh>                
#include <cub/device/device_radix_sort.cuh>
#include <cub/iterator/transform_input_iterator.cuh> 
#include <thrust/copy.h>
#include <thrust/count.h>
#include <thrust/device_ptr.h>
#include <thrust/equal.h>
#include <thrust/execution_policy.h>
#include <thrust/gather.h>
#include <thrust/sequence.h>


// Functor:  '<' → +1   '/' → −1   else 0
struct TagSignFunctor {
    __device__ int operator()(uint8_t c) const {
        return (c == '<') ?  1 :
               (c == '/') ? -1 : 0;
    }
};

// Generic wrapper: inclusive sum on the GPU
template <typename InputIt, typename OutputT>  
void cub_inclusive_sum(InputIt  d_in_begin, OutputT* d_out, size_t   num_items, cudaStream_t stream = 0)
{
    void*  d_temp      = nullptr;
    size_t temp_bytes  = 0;

    // size‑query pass
    cub::DeviceScan::InclusiveSum(d_temp, temp_bytes, d_in_begin, d_out, num_items, stream);

    // allocate temp buffer exactly once
    cudaMalloc(&d_temp, temp_bytes);

    // real work
    cub::DeviceScan::InclusiveSum(d_temp, temp_bytes, d_in_begin, d_out, num_items, stream);

    cudaFree(d_temp);
}


struct decrease_uint32 {
    __host__ __device__
    uint32_t operator()(uint32_t x) const {
        return x - 1;
    }
};

struct is_opening_char_uint8 {
    __host__ __device__
    bool operator()(uint8_t x) const {
        return x == '<';
    }
};


// struct is_true {
//     __host__ __device__
//     bool operator()(uint8_t x) const {
//         return x != 0;
//     }
// };
// struct abs_from_pre_depth {
//     __host__ __device__
//     int8_t operator()(int8_t val) const {
//         return abs(val) > 0;
//     }
// };


__global__ void build_pair_pos_from_sorted_adjacent_kernel(
    const uint8_t* sorted_token_values,
    const uint32_t* sorted_token_offsets,
    uint32_t* pair_pos,
    uint32_t tokens_count,
    int* d_error_flag
) {
    size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;

    for (uint32_t i = index; i < tokens_count; i += stride) {
        if (__ldg(d_error_flag) != 0) return;
        const uint8_t token = __ldg(&sorted_token_values[i]);
        if (token != '<') continue;

        const uint32_t partner_i = i + 1;
        // if (partner_i >= tokens_count) {
        //     atomicExch(d_error_flag, 1);
        //     continue;
        // }

        const uint8_t partner_token = __ldg(&sorted_token_values[partner_i]);
        // if (partner_token != '/') {
        //     atomicExch(d_error_flag, 1);
        //     continue;
        // }

        const uint32_t open_offset = __ldg(&sorted_token_offsets[i]);
        const uint32_t close_offset = __ldg(&sorted_token_offsets[partner_i]);
        pair_pos[open_offset] = close_offset;
    }
}

__global__ void build_pair_pos_from_sorted_adjacent_kernel_simd(
    const uint8_t* sorted_token_values,
    const uint32_t* sorted_token_offsets,
    uint32_t* pair_pos,
    uint32_t tokens_count,
    int* d_error_flag
) {
    /*
    Performance wise doesn't matter much in comparison to the regular kernel, but it's a good example of how to use SIMD instructions to process data.
    It's a good example of how to use SIMD instructions to process data.
    */
    const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t threads = blockDim.x * gridDim.x;

    // Each thread processes a 4-wide chunk to increase ILP/work per thread.
    const uint32_t base = tid * 4u;
    const uint32_t chunk_stride = threads * 4u;
    constexpr uint32_t LT4 = 0x3C3C3C3Cu; // '<' replicated in all bytes.

    for (uint32_t chunk = base; chunk < tokens_count; chunk += chunk_stride) {
        if (__ldg(d_error_flag) != 0) return;

        // Load 4 token bytes and compare all lanes using one SIMD compare instruction.
        const uint8_t t0 = (chunk + 0u < tokens_count) ? __ldg(&sorted_token_values[chunk + 0u]) : 0u;
        const uint8_t t1 = (chunk + 1u < tokens_count) ? __ldg(&sorted_token_values[chunk + 1u]) : 0u;
        const uint8_t t2 = (chunk + 2u < tokens_count) ? __ldg(&sorted_token_values[chunk + 2u]) : 0u;
        const uint8_t t3 = (chunk + 3u < tokens_count) ? __ldg(&sorted_token_values[chunk + 3u]) : 0u;

        const uint32_t packed_tokens =
            static_cast<uint32_t>(t0) |
            (static_cast<uint32_t>(t1) << 8) |
            (static_cast<uint32_t>(t2) << 16) |
            (static_cast<uint32_t>(t3) << 24);

        const uint32_t lane_eq_mask = __vcmpeq4(packed_tokens, LT4);

        #pragma unroll
        for (uint32_t lane = 0; lane < 4u; ++lane) {
            const uint32_t i = chunk + lane;
            if (i >= tokens_count) continue;

            const uint32_t partner_i = i + 1u;
            const uint32_t is_open = ((lane_eq_mask >> (lane * 8u)) & 0xFFu) ? 1u : 0u;
            const uint32_t has_partner = (partner_i < tokens_count) ? 1u : 0u;
            const uint32_t write_mask = is_open & has_partner;

            // Keep store path branch-minimized: write only when lane predicate is active.
            if (__builtin_expect(write_mask != 0u, 0)) {
                const uint32_t open_offset = __ldg(&sorted_token_offsets[i]);
                const uint32_t close_offset = __ldg(&sorted_token_offsets[partner_i]);
                pair_pos[open_offset] = close_offset;
            }
        }
    }
}

__device__ __forceinline__ bool is_tag_name_delim(uint8_t c) {
    return (c == '>') || (c == ' ') || (c == '/') || (c == '\t') || (c == '\n') || (c == '\r');
}

__global__ void validate_depth_and_pairs_kernel(
    const uint32_t* d_depth,
    const uint32_t* d_sorted_depth,
    const uint32_t* d_sorted_token_offsets,
    const uint32_t* d_sorted_token_indices,
    const uint8_t* d_sorted_token_values,
    const uint32_t* d_pair_pos,
    const uint8_t* d_xmlContent,
    uint32_t tokens_count,
    int* d_error_flag
) {
    const uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t stride = blockDim.x * gridDim.x;
    constexpr int MAX_TAG_COMPARE = 256;

    for (uint32_t k = index; k < tokens_count; k += stride) {
        const uint32_t current_depth = __ldg(&d_depth[k]);
        if (static_cast<int32_t>(current_depth) < 0) {
            atomicExch(d_error_flag, 1);
        }
    }

    for (uint32_t i = index; i < tokens_count; i += stride) {
        if (__ldg(d_error_flag) != 0) return;

        const uint8_t current_token = __ldg(&d_sorted_token_values[i]);
        if (current_token != '<') continue;

        const uint32_t partner_i = i + 1;
        if (partner_i >= tokens_count) {
            atomicExch(d_error_flag, 1);
            continue;
        }

        const uint8_t partner_token = __ldg(&d_sorted_token_values[partner_i]);
        if (partner_token != '/') {
            atomicExch(d_error_flag, 1);
            continue;
        }

        const uint32_t open_depth = __ldg(&d_sorted_depth[i]);
        const uint32_t close_depth = __ldg(&d_sorted_depth[partner_i]);
        if (static_cast<int32_t>(open_depth) < 0 || static_cast<int32_t>(close_depth) < 0 || open_depth != close_depth) {
            atomicExch(d_error_flag, 1);
            continue;
        }

        const uint32_t open_offset = __ldg(&d_sorted_token_offsets[i]);
        const uint32_t open_xml_idx = __ldg(&d_sorted_token_indices[i]);
        const uint32_t close_offset = __ldg(&d_sorted_token_offsets[partner_i]);
        const uint32_t close_xml_idx = __ldg(&d_sorted_token_indices[partner_i]);

        const uint32_t pair_pos_close_offset = __ldg(&d_pair_pos[open_offset]);
        if (pair_pos_close_offset != close_offset) {
            atomicExch(d_error_flag, 1);
            continue;
        }

        if (__ldg(&d_xmlContent[open_xml_idx]) != '<' || __ldg(&d_xmlContent[close_xml_idx]) != '/') {
            atomicExch(d_error_flag, 1);
            continue;
        }

        // User-defined self-closing rule: partner is '/' and next XML char is '>'.
        if (__ldg(&d_xmlContent[close_xml_idx + 1]) == '>') {
            continue;
        }

        bool ended = false;
        bool mismatch = false;
        #pragma unroll 4
        for (int n = 0; n < MAX_TAG_COMPARE; ++n) {
            const uint8_t open_c = __ldg(&d_xmlContent[open_xml_idx + 1 + n]);
            const uint8_t close_c = __ldg(&d_xmlContent[close_xml_idx + 1 + n]);
            const bool open_end = is_tag_name_delim(open_c);
            const bool close_end = is_tag_name_delim(close_c);

            if (open_end || close_end) {
                mismatch = (open_end != close_end);
                ended = true;
                break;
            }
            if (open_c != close_c) {
                mismatch = true;
                ended = true;
                break;
            }
        }

        if (!ended || mismatch) {
            atomicExch(d_error_flag, 1);
        }
    }
}

struct is_open_or_close_token {
    __host__ __device__
    bool operator()(uint8_t c) const {
        return (c == '<') || (c == '/');
    }
};

template <typename ValueT>
static inline void stable_sort_pairs_by_depth_cub(
    const uint32_t* d_keys_in,
    uint32_t* d_keys_out,
    const ValueT* d_values_in,
    ValueT* d_values_out,
    uint32_t num_items
) {
    void* d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;

    cub::DeviceRadixSort::SortPairs(
        d_temp_storage,
        temp_storage_bytes,
        d_keys_in,
        d_keys_out,
        d_values_in,
        d_values_out,
        num_items
    );

    cudaMalloc(&d_temp_storage, temp_storage_bytes);

    cub::DeviceRadixSort::SortPairs(
        d_temp_storage,
        temp_storage_bytes,
        d_keys_in,
        d_keys_out,
        d_values_in,
        d_values_out,
        num_items
    );

    cudaFree(d_temp_storage);
}

template <typename T>
static inline bool device_arrays_equal(const T* lhs, const T* rhs, uint32_t count) {
    return thrust::equal(
        thrust::cuda::par,
        thrust::device_pointer_cast(lhs),
        thrust::device_pointer_cast(lhs + count),
        thrust::device_pointer_cast(rhs)
    );
}


void stage3_parse(
    const uint32_t* d_token_indices,
    const uint8_t* d_token_values,
    const uint32_t tokens_count,
    const uint8_t* d_xmlContent,
    uint32_t** d_DEPTH, // Output: depth of each token
    uint32_t** d_pair_pos, // Output: pair positions for open-close tags
    int* h_validation_error // Output: host-side validation flag
    ) 
    {
    if (h_validation_error != nullptr) {
        *h_validation_error = 0;
    }
    if (tokens_count <= 2) {
        std::cout << "No (less than two) tokens to parse.\n";
        return;
    }

    #if defined(DEBUG_MODE) && DEBUG_MODE == 2
        // Measure the time taken for the kernel execution
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);
    #endif
    
    using SignIter = cub::TransformInputIterator<int ,TagSignFunctor, const uint8_t*>;      // value type produced (int32), functor, underlying pointer type
    SignIter sign_begin(d_token_values, TagSignFunctor{});                                  // virtual stream of  ‑1 / 0 / +1  generated on the fly 

    cudaMalloc((void**)d_DEPTH, tokens_count * sizeof(uint32_t));                           // allocate DEPTH output once 
    cub_inclusive_sum(sign_begin, *d_DEPTH, tokens_count);                                  // single‑line scan

    #if defined(DEBUG_MODE) && DEBUG_MODE == 5
        printGpuMemoryUsage("Parser Bytemap Creation");
    #endif


    #if defined(DEBUG_MODE) && DEBUG_MODE == 2
        // Measure the time taken for the kernel execution
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float time_ms = 0.0f;
        cudaEventElapsedTime(&time_ms, start, stop);
        
        double time_ns = static_cast<uint64_t>(time_ms * 1e6);
        std::cout << "⏱️ PRE DEPTH Transform execution time: " << time_ns << " ns" << std::endl;
        std::cout << "⏱️ PRE DEPTH Transform execution time: " << time_ns / 1e6 << " ms" << std::endl;

        // Cleanup
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    #endif
    #if defined(DEBUG_MODE) && DEBUG_MODE == 1
        print_uint32_array("Token Indices", d_token_indices, tokens_count);
        print_uint32_array("Depth", *d_DEPTH, tokens_count);
    #endif
    #if defined(DEBUG_MODE) && DEBUG_MODE == 2
        // Measure the time taken for the kernel execution
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);
    #endif

    // Now we need to handle the open-close bytemap to adjust the depth.
    thrust::transform_if(
        thrust::cuda::par,
        thrust::device_pointer_cast(*d_DEPTH),
        thrust::device_pointer_cast(*d_DEPTH + tokens_count),
        thrust::device_pointer_cast(d_token_values),
        thrust::device_pointer_cast(*d_DEPTH),
        decrease_uint32(),
        is_opening_char_uint8()
    );

    #if defined(DEBUG_MODE) && DEBUG_MODE == 5
        printGpuMemoryUsage("Parser Transform If");
    #endif

    #if defined(DEBUG_MODE) && DEBUG_MODE == 2
        // Measure the time taken for the kernel execution
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        time_ms = 0.0f;
        cudaEventElapsedTime(&time_ms, start, stop);
        
        time_ns = static_cast<uint64_t>(time_ms * 1e6);
        std::cout << "⏱️ Transform if execution time: " << time_ns << " ns" << std::endl;
        std::cout << "⏱️ Transform if Transform execution time: " << time_ns / 1e6 << " ms" << std::endl;

        // Cleanup
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    #endif
    #if defined(DEBUG_MODE) && DEBUG_MODE == 1
        print_uint32_array("Depth (after transform if)", *d_DEPTH, tokens_count);
    #endif

    #if defined(DEBUG_MODE) && DEBUG_MODE == 4
        uint32_t max_depth = thrust::reduce(
            thrust::cuda::par,
            thrust::device_pointer_cast(*d_DEPTH),
            thrust::device_pointer_cast(*d_DEPTH + tokens_count),
            static_cast<uint32_t>(0),
            thrust::maximum<uint32_t>()
        );
        cout << "#depth: " << max_depth << endl;
    #endif

    #if defined(DEBUG_MODE) && DEBUG_MODE == 2
        cudaEvent_t pair_build_start, pair_build_stop;
        cudaEventCreate(&pair_build_start);
        cudaEventCreate(&pair_build_stop);
        cudaEventRecord(pair_build_start);
    #endif


    // ---- Pair-building prep (GPU-only) ----
    // We keep the original parser outputs unchanged and build all sort/pair artifacts
    // in temporary buffers.
    uint32_t* d_u32_slab = nullptr;
    uint8_t* d_u8_slab = nullptr;
    uint32_t* d_depth_keys_in = nullptr;
    uint32_t* d_depth_keys_out = nullptr;
    // These hold token offsets [0..tokens_count-1]. We sort offsets by depth, then
    // gather indices/values in that sorted order.
    uint32_t* d_token_offsets_in = nullptr;
    uint32_t* d_token_offsets_out = nullptr;
    // Sorted views (by depth) of token XML positions and token chars ('<', '/', '=', ...).
    uint32_t* d_sorted_token_indices = nullptr;
    uint8_t* d_sorted_token_values = nullptr;

    // DEBUG_MODE=1 snapshots to verify original arrays are not modified.
    uint32_t* d_depth_original = nullptr;
    uint32_t* d_token_indices_original = nullptr;
    uint8_t* d_token_values_original = nullptr;

    // Allocate 2 slabs (u32 + u8) and point each "row" into its segment.
    // This reduces allocator overhead versus many small cudaMalloc calls.
    uint32_t u32_rows = 5; // depth_in, depth_out, offsets_in, offsets_out, sorted_indices
    uint32_t u8_rows = 1;  // sorted_values
    #if defined(DEBUG_MODE) && DEBUG_MODE == 1
        u32_rows += 2; // depth_original, token_indices_original
        u8_rows += 1;  // token_values_original
    #endif

    cudaMalloc((void**)&d_u32_slab, tokens_count * u32_rows * sizeof(uint32_t));
    cudaMalloc((void**)&d_u8_slab, tokens_count * u8_rows * sizeof(uint8_t));

    uint32_t* u32_row = d_u32_slab;
    d_depth_keys_in = u32_row;                 u32_row += tokens_count;
    d_depth_keys_out = u32_row;                u32_row += tokens_count;
    d_token_offsets_in = u32_row;              u32_row += tokens_count;
    d_token_offsets_out = u32_row;             u32_row += tokens_count;
    d_sorted_token_indices = u32_row;          u32_row += tokens_count;

    uint8_t* u8_row = d_u8_slab;
    d_sorted_token_values = u8_row;            u8_row += tokens_count;

    // Copy depth into sortable key buffer so original d_DEPTH stays untouched.
    cudaMemcpy(d_depth_keys_in, *d_DEPTH, tokens_count * sizeof(uint32_t), cudaMemcpyDeviceToDevice);

    #if defined(DEBUG_MODE) && DEBUG_MODE == 1
        // Preserve originals for post-checks (sanity: sort path must be non-destructive).
        d_depth_original = u32_row;            u32_row += tokens_count;
        d_token_indices_original = u32_row;    u32_row += tokens_count;
        d_token_values_original = u8_row;      u8_row += tokens_count;

        cudaMemcpy(d_depth_original, *d_DEPTH, tokens_count * sizeof(uint32_t), cudaMemcpyDeviceToDevice);
        cudaMemcpy(d_token_indices_original, d_token_indices, tokens_count * sizeof(uint32_t), cudaMemcpyDeviceToDevice);
        cudaMemcpy(d_token_values_original, d_token_values, tokens_count * sizeof(uint8_t), cudaMemcpyDeviceToDevice);
    #endif

    const uint32_t sort_blocks = (tokens_count + BLOCKSIZE - 1) / BLOCKSIZE;
    // Build identity offsets: each element points to its original token slot.
    thrust::sequence(
        thrust::cuda::par,
        thrust::device_pointer_cast(d_token_offsets_in),
        thrust::device_pointer_cast(d_token_offsets_in + tokens_count),
        0u
    );

    // Stable sort offsets by depth key.
    // Result: d_token_offsets_out is an ordering of token slots grouped by depth.
    stable_sort_pairs_by_depth_cub(
        d_depth_keys_in,
        d_depth_keys_out,
        d_token_offsets_in,
        d_token_offsets_out,
        tokens_count
    );

    // Reconstruct sorted token arrays using the sorted offsets.
    // d_sorted_token_indices: XML byte positions in sorted-by-depth order.
    // d_sorted_token_indices = d_token_offsets_out;
    // d_sorted_token_values: token chars in sorted-by-depth order.
    thrust::gather(
        thrust::cuda::par,
        thrust::device_pointer_cast(d_token_offsets_out),
        thrust::device_pointer_cast(d_token_offsets_out + tokens_count),
        thrust::device_pointer_cast(d_token_indices),
        thrust::device_pointer_cast(d_sorted_token_indices)
    );

    thrust::gather(
        thrust::cuda::par,
        thrust::device_pointer_cast(d_token_offsets_out),
        thrust::device_pointer_cast(d_token_offsets_out + tokens_count),
        thrust::device_pointer_cast(d_token_values),
        thrust::device_pointer_cast(d_sorted_token_values)
    );

    // Global GPU error flag:
    // 0 = valid so far, 1 = any pairing/validation rule violated.
    int* d_validation_error_flag = nullptr;
    cudaMalloc((void**)&d_validation_error_flag, sizeof(int));
    cudaMemset(d_validation_error_flag, 0, sizeof(int));

    // pair_pos maps "open token offset" -> "closing token offset".
    // Initialize to 0xFFFFFFFF (invalid/unpaired sentinel).
    cudaMalloc((void**)d_pair_pos, tokens_count * sizeof(uint32_t));
    cudaMemset(*d_pair_pos, 0xFF, tokens_count * sizeof(uint32_t));

    // Build pair_pos directly on GPU from adjacent entries in sorted arrays:
    // if sorted[i] is '<', partner is expected at sorted[i+1] and must be '/'.
    // Previous scalar launch is kept (commented) per request.
    // build_pair_pos_from_sorted_adjacent_kernel<<<sort_blocks, BLOCKSIZE>>>(
    //     d_sorted_token_values,
    //     d_token_offsets_out,
    //     *d_pair_pos,
    //     tokens_count,
    //     d_validation_error_flag
    // );

    // SIMD launch: each thread processes 4 sorted indices.
    build_pair_pos_from_sorted_adjacent_kernel_simd<<<sort_blocks, BLOCKSIZE>>>(
        d_sorted_token_values,
        d_token_offsets_out, // instead of sending the index, we are sending the offsets 
        *d_pair_pos,
        tokens_count,
        d_validation_error_flag
    );

    #if defined(DEBUG_MODE) && DEBUG_MODE == 2
        cudaEventRecord(pair_build_stop);
        cudaEventSynchronize(pair_build_stop);
        time_ms = 0.0f;
        cudaEventElapsedTime(&time_ms, pair_build_start, pair_build_stop);
        time_ns = static_cast<uint64_t>(time_ms * 1e6);
        std::cout << "⏱️ Post-transform_if to pair_pos build time: " << time_ns << " ns" << std::endl;
        std::cout << "⏱️ Post-transform_if to pair_pos build time: " << time_ns / 1e6 << " ms" << std::endl;
        cudaEventDestroy(pair_build_start);
        cudaEventDestroy(pair_build_stop);
    #endif

    #if defined(DEBUG_MODE) && DEBUG_MODE == 1
        print_uint32_array("Pair Pos", *d_pair_pos, tokens_count);
    #endif

    #if defined(DEBUG_MODE) && DEBUG_MODE == 2
        cudaEvent_t validation_kernel_start, validation_kernel_stop;
        cudaEventCreate(&validation_kernel_start);
        cudaEventCreate(&validation_kernel_stop);
        cudaEventRecord(validation_kernel_start);
    #endif

    validate_depth_and_pairs_kernel<<<sort_blocks, BLOCKSIZE>>>(
        *d_DEPTH,
        d_depth_keys_out,
        d_token_offsets_out,
        d_sorted_token_indices,
        d_sorted_token_values,
        *d_pair_pos,
        d_xmlContent,
        tokens_count,
        d_validation_error_flag
    );

    #if defined(DEBUG_MODE) && DEBUG_MODE == 2
        cudaEventRecord(validation_kernel_stop);
        cudaEventSynchronize(validation_kernel_stop);
        time_ms = 0.0f;
        cudaEventElapsedTime(&time_ms, validation_kernel_start, validation_kernel_stop);
        time_ns = static_cast<uint64_t>(time_ms * 1e6);
        std::cout << "⏱️ validate_depth_and_pairs_kernel time: " << time_ns << " ns" << std::endl;
        std::cout << "⏱️ validate_depth_and_pairs_kernel time: " << time_ns / 1e6 << " ms" << std::endl;
        cudaEventDestroy(validation_kernel_start);
        cudaEventDestroy(validation_kernel_stop);
    #endif

    int h_error_flag = 0;
    cudaMemcpy(&h_error_flag, d_validation_error_flag, sizeof(int), cudaMemcpyDeviceToHost);
    if (h_validation_error != nullptr) {
        *h_validation_error = h_error_flag;
    }

    #if defined(DEBUG_MODE) && DEBUG_MODE == 1
        print_sorted_tokens_after_depth_sort(d_sorted_token_indices, d_sorted_token_values, tokens_count);
        print_uint32_array("Depth (sorted copy)", d_depth_keys_out, tokens_count);

        const bool depth_unchanged = device_arrays_equal(*d_DEPTH, d_depth_original, tokens_count);
        const bool indices_unchanged = device_arrays_equal(d_token_indices, d_token_indices_original, tokens_count);
        const bool values_unchanged = device_arrays_equal(d_token_values, d_token_values_original, tokens_count);
        std::cout << "[Parser Check] d_DEPTH unchanged: " << (depth_unchanged ? "YES" : "NO") << std::endl;
        std::cout << "[Parser Check] d_token_indices unchanged: " << (indices_unchanged ? "YES" : "NO") << std::endl;
        std::cout << "[Parser Check] d_token_values unchanged: " << (values_unchanged ? "YES" : "NO") << std::endl;
        std::cout << "[Parser Validation] validation_error=" << h_error_flag << std::endl;
    #endif

    #if defined(DEBUG_MODE) && DEBUG_MODE == 5
        printGpuMemoryUsage("before free in structure recognition");
    #endif
    

    if (d_u32_slab != nullptr) {
        cudaFree(d_u32_slab);
    }
    if (d_u8_slab != nullptr) {
        cudaFree(d_u8_slab);
    }
    if (d_validation_error_flag != nullptr) {
        cudaFree(d_validation_error_flag);
    }

}
