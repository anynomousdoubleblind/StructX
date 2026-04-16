#include "validation.h"

__device__ __forceinline__
void vectorizedClassification(uint32_t block_compressed, uint32_t prev1, uint32_t& result, uint64_t size, int total_padded_32){
    constexpr const uint8_t TOO_SHORT   = 1<<0; // 00000001
                                                // The leading byte must be followed by N-1 continuation bytes, 
                                                // where N is the UTF-8 character length.
                                                // 11______ 0_______
                                                // 11______ 11______

    constexpr const uint8_t TOO_LONG    = 1<<1; // The leading byte must not be a continuation byte.
                                                // 0_______ 10______

    constexpr const uint8_t OVERLONG_2  = 1<<5; // Above U+7F for two-byte characters,
                                                // 1100000_ 10______
    constexpr const uint8_t OVERLONG_3  = 1<<2; // Above U+7FF for three-byte characters,
                                                // 11100000 100_____
    constexpr const uint8_t OVERLONG_4  = 1<<6; // Above U+7FFF for three-byte characters,
                                                // 11110000 1000____

    constexpr const uint8_t SURROGATE   = 1<<4; // The decoded character must be not be in U+D800...DFFF
                                                // 11101101 101_____

    constexpr const uint8_t TWO_CONTS   = 1<<7; // Two continious bit after each other
                                                // 10______ 10______

    constexpr const uint8_t TOO_LARGE   = 1<<3; // The decoded character must be less than or equal to U+10FFFF
                                                // 11110100 1001____
                                                // 11110100 101_____
                                                // 11110101 1001____
                                                // 11110101 101_____
                                                // 1111011_ 1001____
                                                // 1111011_ 101_____
                                                // 11111___ 1001____
                                                // 11111___ 101_____

    constexpr const uint8_t TOO_LARGE_1000 = 1<<6;
                                                // Out of the range, it must be maximum 100 if you see 0101, 011_, or 1___
                                                // 11110101 1000____
                                                // 1111011_ 1000____
                                                // 11111___ 1000____


    constexpr const uint8_t CARRY = TOO_SHORT | TOO_LONG | TWO_CONTS; 
                                                // These all have ____ in byte 1 . 10000011


    
    // SIMDJSON use table in CPU, but in GPU Table is very slow
    // we check 4 character in a single time by this:
    constexpr const uint32_t TOO_SHORT_32 = (
        ((uint32_t)TOO_SHORT)       | 
        ((uint32_t)TOO_SHORT) << 8  | 
        ((uint32_t)TOO_SHORT) << 16 | 
        ((uint32_t)TOO_SHORT) << 24
    );
    constexpr const uint32_t TOO_LONG_32 = (
        ((uint32_t)TOO_LONG)        | 
        ((uint32_t)TOO_LONG) << 8   |
        ((uint32_t)TOO_LONG) << 16  |
        ((uint32_t)TOO_LONG) << 24
    );
    constexpr const uint32_t OVERLONG_2_32 = (
        ((uint32_t)OVERLONG_2)       | 
        ((uint32_t)OVERLONG_2) << 8  | 
        ((uint32_t)OVERLONG_2) << 16 | 
        ((uint32_t)OVERLONG_2) << 24
    );
    constexpr const uint32_t OVERLONG_3_32 = (
        ((uint32_t)OVERLONG_3)       | 
        ((uint32_t)OVERLONG_3) << 8  | 
        ((uint32_t)OVERLONG_3) << 16 | 
        ((uint32_t)OVERLONG_3) << 24
    );
    constexpr const uint32_t OVERLONG_4_32 = (
        ((uint32_t)OVERLONG_4)       | 
        ((uint32_t)OVERLONG_4) << 8  | 
        ((uint32_t)OVERLONG_4) << 16 | 
        ((uint32_t)OVERLONG_4) << 24
    );
    constexpr const uint32_t SURROGATE_32 = (
        ((uint32_t)SURROGATE)       | 
        ((uint32_t)SURROGATE) << 8  | 
        ((uint32_t)SURROGATE) << 16 | 
        ((uint32_t)SURROGATE) << 24
    );
    constexpr const uint32_t TWO_CONTS_32 = (
        ((uint32_t)TWO_CONTS)       | 
        ((uint32_t)TWO_CONTS) << 8  | 
        ((uint32_t)TWO_CONTS) << 16 | 
        ((uint32_t)TWO_CONTS) << 24
    );
    constexpr const uint32_t TOO_LARGE_32 = (
        ((uint32_t)TOO_LARGE)       | 
        ((uint32_t)TOO_LARGE) << 8  | 
        ((uint32_t)TOO_LARGE) << 16 | 
        ((uint32_t)TOO_LARGE) << 24
    );
    constexpr const uint32_t TOO_LARGE_1000_32 = (
        ((uint32_t)TOO_LARGE_1000)       | 
        ((uint32_t)TOO_LARGE_1000) << 8  | 
        ((uint32_t)TOO_LARGE_1000) << 16 | 
        ((uint32_t)TOO_LARGE_1000) << 24
    );
    constexpr const uint32_t CARRY_32 = (
        ((uint32_t)CARRY)       | 
        ((uint32_t)CARRY) << 8  | 
        ((uint32_t)CARRY) << 16 |
        ((uint32_t)CARRY) << 24
    );
    


    uint32_t prev1_current = prev1;
    uint32_t byte_1 = 
        (__vcmpltu4(prev1_current, 0x80808080) & TOO_LONG_32) |
        (__vcmpgeu4(prev1_current, 0xC0C0C0C0) & TOO_SHORT_32) | 
        ( (__vcmpeq4(prev1_current, 0xC0C0C0C0) | __vcmpeq4(prev1_current, 0xC1C1C1C1)) & OVERLONG_2_32) | 
        (__vcmpeq4(prev1_current, 0xEDEDEDED) & (SURROGATE_32)) | 
        (__vcmpeq4(prev1_current, 0xE0E0E0E0) & (OVERLONG_3_32)) | 
        (__vcmpeq4(prev1_current, 0xF0F0F0F0) & (OVERLONG_4_32)) | 
        (__vcmpgtu4(prev1_current, 0xF4F4F4F4) & TOO_LARGE_1000_32) | 
        (__vcmpgtu4(prev1_current, 0xF3F3F3F3) & TOO_LARGE_32);

    byte_1 = (__vcmpeq4(byte_1, 0x00000000) & TWO_CONTS_32);
    

    uint32_t block_compressed_high = (block_compressed >> 4) & 0x0F0F0F0F; 
    // 4 khune bala ro brdshti 
    // baraye moqaysee adadi bordim daste rast k rahat tr bashe

    // to make it more easier than before, save it and use it multiple time
    uint32_t less_than_12 = __vcmpltu4(block_compressed_high, 0x0C0C0C0C);
    uint32_t byte_2_high = 
        ((__vcmpltu4(block_compressed_high, 0x08080808) | __vcmpgtu4(block_compressed_high, 0x0B0B0B0B)) & TOO_SHORT_32) |
        (less_than_12 & __vcmpgeu4(block_compressed_high, 0x08080808) & (TOO_LONG_32 | OVERLONG_2_32 | TWO_CONTS_32)) | 
        (less_than_12 & __vcmpgtu4(block_compressed_high, 0x08080808) & TOO_LARGE_32) | 
        (__vcmpeq4(block_compressed_high, 0x08080808) & (TOO_LARGE_1000_32 | OVERLONG_4_32)) | 
        (__vcmpgtu4(block_compressed_high, 0x09090909) & less_than_12 & SURROGATE_32); 


    result =   (byte_1 & byte_2_high);  
    // 0 --> okay and return secussfuly
}

// make sure it has 2 or 3 continuation
// for 3,4 Byte
__device__ __forceinline__
void continuationBytes(uint32_t prev2, uint32_t prev3, uint32_t sc, uint32_t& must32Upper_sc, uint64_t size, int total_padded_32){
    static const uint32_t third_subtract_byte =  
    // 11100000 - 1 --> 11011111 --> This is the maximum of 2 Byte, So if it’s more than this, we have 3 
        (0b11100000u-1)       | 
        (0b11100000u-1) << 8  | 
        (0b11100000u-1) << 16 | 
        (0b11100000u-1) << 24;

    static const uint32_t fourth_subtract_byte = 
        (0b11110000u-1)       | 
        (0b11110000u-1) << 8  |
        (0b11110000u-1) << 16 | 
        (0b11110000u-1) << 24;


    int index = blockIdx.x * blockDim.x + threadIdx.x;

    // the latest byte in our UTF8Bytes (character) is third or fourth
    // subtract prev2 and prev3 from third_subtract_byte and fourth_subtract_byte
    // must be 0 
    // unsign saturated subtraction 4 Byte --> 4 Byte ro parallel az ham kam mikone ya 0 mishe ya 1 
    // ma mikhaym prev2 az third_subtract_byte va prev3 az forth_subtract_byte kochak tr bashe k javab 0 bashe
    uint32_t is_third_byte  = __vsubus4(prev2, third_subtract_byte);
    uint32_t is_fourth_byte = __vsubus4(prev3, fourth_subtract_byte);


    uint32_t gt = ( __vsubss4((int32_t)(is_third_byte | is_fourth_byte), int32_t(0)) ) & 0xFFFFFFFF; 
    
    // because we are working in 32 bit, we need do this for all 4 characters
    uint32_t must32 = __vcmpgtu4(gt, 0); // gt --> hamin must32 hast o mitonim hazfesh knim

    must32Upper_sc = (must32 & 0x80808080) ^ sc;            //  sc --> output of 32 bit check
    // upper bit of each 4 character
} 

__global__ 
void checkAscii(uint32_t* blockCompressed_GPU, uint64_t size, int total_padded_32, bool* hastUTF8, int WORDS){
    int threadId = threadIdx.x;
    __shared__ uint32_t shared_flag;
    
    if(threadId == 0) shared_flag = 0;
    __syncthreads();

    int index = blockIdx.x * blockDim.x + threadId;
    int stride = blockDim.x * gridDim.x;

    for(long i = index; i< total_padded_32; i+=stride){
        int start = i*WORDS;
        #pragma unroll
        for(int j=start; j<size && j<start+WORDS; j++){
            if((blockCompressed_GPU[j] & 0x80808080) != 0) atomicOr(&shared_flag, 1); 
            // check the upper bit
            // atomic or because it works in parallel
        }
        __syncthreads();
    }
    if(threadId == 0 && shared_flag) *hastUTF8 = true;
}

__global__
void checkUTF8(uint32_t* blockCompressed_GPU, uint32_t* error_GPU, uint64_t size, int total_padded_32, int WORDS){
    /*
    - blockCompressed_GPU is a pointer to the compressed data block in GPU memory, 
    - error_GPU is a pointer to a location in GPU memory where the function will store an error code if it detects invalid UTF-8, 
    - size is the size of the data block, 
    - total_padded_32 is the total number of 32-bit words in the padded data block, and 
    - WORDS is the number of words processed by each thread in each iteration of the loop
    */
    static const uint32_t max_val = 
        (uint32_t)(0b11000000u-1 << 24) | 
        (uint32_t)(0b11100000u-1 << 16) | 
        (uint32_t)(0b11110000u-1 << 8)  | 
        (uint32_t)(255); 

    int threadId = threadIdx.x;
    __shared__ uint32_t shared_error;
    if(threadId == 0) shared_error = 0;

    __syncthreads();
    int index = blockIdx.x * blockDim.x + threadId;
    int stride = blockDim.x * gridDim.x;

    for(long i = index; i< total_padded_32; i+=stride){
        int start = i*WORDS;
        #pragma unroll
        for(int j=start; j<size && j<start+WORDS; j++){
            uint32_t current = blockCompressed_GPU[j];
            uint32_t previous = j>0 ? blockCompressed_GPU[j-1] : 0;
            uint32_t prev_incomplete = __vsubus4(previous, max_val);
            
            if((current & 0x80808080) == 0) {
                atomicExch(&shared_error, prev_incomplete);
            }else{
                uint32_t prev1, prev2, prev3;
                uint32_t sc;
                uint32_t must32Upper_sc;

                uint64_t dist = ( ((uint64_t)current) << 32) | (uint64_t) previous;
                prev1 = (uint32_t)(dist >> 3*8); // shifted by 3 byte (3 * 8 bits)
                prev2 = (uint32_t)(dist >> 2*8); // shifted by 2 byte (2 * 8 bits)
                prev3 = (uint32_t)(dist >> 1*8); // shifted by 1 byte (1 * 8 bits)

                vectorizedClassification(current, prev1, sc, size, total_padded_32); // check 1,2 Byte 
                continuationBytes(prev2, prev3, sc, must32Upper_sc, size, total_padded_32); // Check 3,4 byte

                atomicExch(&shared_error, must32Upper_sc); // return error
            }
        }
    }
    __syncthreads();
    if(threadId==0 && shared_error) *error_GPU = shared_error;
}

inline bool stage1_UTF8Validator(uint32_t * block_GPU, uint64_t size){
    // _________________INIT_________________________
    int total_padded_32 = size;

    uint32_t* general_ptr;
    cudaMallocAsync(&general_ptr, sizeof(uint32_t), 0);
    uint32_t* error_GPU = general_ptr;
    cudaMemsetAsync(error_GPU, 0, sizeof(uint32_t), 0);

  
    int total_padded_16B = (size+3)/4;
    int WORDS = 4;
    int numBlock_16B = (total_padded_16B+BLOCKSIZE-1) / BLOCKSIZE;


    bool hastUTF8 = false;
    bool* hastUTF8_GPU;
    cudaMallocAsync(&hastUTF8_GPU, sizeof(bool), 0);                  //  Allocates Memory on the Device and Returns a Pointer to the Allocated Memory.
    cudaMemsetAsync(hastUTF8_GPU, 0, sizeof(bool), 0);                //  Initializes a Block of Memory on the Device with a Specified Value


    // _________________PART_1_______________________
    checkAscii<<<numBlock_16B, BLOCKSIZE>>>(block_GPU, size, total_padded_16B, hastUTF8_GPU, WORDS);
    cudaStreamSynchronize(0);
    
    cudaMemcpyAsync(&hastUTF8, hastUTF8_GPU, sizeof(bool), cudaMemcpyDeviceToHost, 0);
    //cudaFreeAsync(hastUTF8_GPU, 0);

    if(!hastUTF8){ 
        cudaFreeAsync(general_ptr, 0);
        return true;
    }

    // _________________PART_2_______________________
    checkUTF8<<<numBlock_16B, BLOCKSIZE>>>(block_GPU, error_GPU, size, total_padded_16B, WORDS);
    cudaStreamSynchronize(0);

    // _________________RESULT_______________________
    uint32_t error = 0;
    cudaMemcpyAsync(&error, error_GPU, sizeof(uint32_t), cudaMemcpyDeviceToHost, 0);
    cudaFreeAsync(general_ptr, 0);
    if(error != 0){ 
        printf("Incomplete ASCII!\n"); 
        return false;
    }

            
    return true;

}
