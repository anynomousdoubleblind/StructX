#include <iostream>
#include <cstdint>
#include <string>
#include <vector>

// Function to print matches from XML content
void print_xml_matches(
    const uint8_t* h_xmlContent, 
    const uint32_t* h_matches, 
    int matched_tokens_size
) {
    std::cout << "Matches found:" << matched_tokens_size/2 << std::endl;
    for (int i = 0; i < matched_tokens_size && i < 20; i += 2) {
    
        uint32_t start_idx = h_matches[i];
        uint32_t end_idx   = h_matches[i + 1];
        if(h_xmlContent[end_idx-1] == '<') {
            end_idx = end_idx-2;
            while(h_xmlContent[start_idx] != '>') {
                start_idx++;
            }
            start_idx++;
        }
        else {
            end_idx = end_idx+1;
        }

        std::cout << "Match " << (i / 2) << ": ";

        // Print content from start to end
        for (uint32_t j = start_idx; j <= end_idx; ++j) {
            std::cout << static_cast<char>(h_xmlContent[j]);
        }
        std::cout << std::endl;
    }
} 