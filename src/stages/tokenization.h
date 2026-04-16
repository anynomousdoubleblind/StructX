#ifndef TOKENIZATION_H
#define TOKENIZATION_H

#include <cstdint>

// Host function to launch tokenization kernel
void stage2_tokenization(
    uint32_t* d_xmlContent,
    size_t padded_length,
    uint32_t** d_token_indices,  // Output: token positions
    uint8_t** d_token_values,    // Output: actual token characters
    size_t* tokens_count       // Output: how many tokens
);

#endif  // TOKENIZATION_H
