#pragma once  

#include <cstdint>

// Function to print matches from XML content
void print_xml_matches(
    const uint8_t* h_xmlContent, 
    const uint32_t* h_matches, 
    int matched_tokens_size
);