#ifndef QUERY_H
#define QUERY_H

#include <cstdint>

void stage4_xpath(
    const uint32_t* d_token_indices,
    const uint8_t* d_token_values,
    const uint32_t tokens_count,
    uint32_t xml_length,
    const uint8_t* d_xmlContent,
    uint32_t** pair_pos, // open token offset -> close token offset
    uint32_t** transfored_depth,
    const uint8_t* xpath_query,  
    uint32_t** d_selected_token_indices,          // output: matches indices
    size_t* matched_tokens       // Output: how many tokens
);

#endif

