#ifndef PARSER_H
#define PARSER_H

#include <cstdint>

void stage3_parse(
    const uint32_t* d_token_indices,
    const uint8_t* d_token_values,
    const uint32_t tokens_count,
    const uint8_t* d_xmlContent,
    uint32_t** depth,
    uint32_t** pair_pos, // output: open token offset -> close token offset
    int* h_validation_error // output: host-side validation flag (0 valid, 1 invalid)
);

#endif
