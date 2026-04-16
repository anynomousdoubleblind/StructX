#include "query.h"
#include <iostream>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <string>
#include <sstream>
#include <regex>

using namespace std;    

#define MAX_ATTR_DEPTHS 8  // adjust as needed
#define MAX_TAG_COND_DEPTHS 8  // adjust as needed
#define MAX_STEPS 16        // Max XPath depth (adjust if needed): root/child/grandchild
#define MAX_STR_LEN 32      // Max tag/attr name length: <name> <attr> <val>
#define MAX_OP_LEN 4        // Max operator length : < = > != =

__constant__ uint8_t c_pure_tag_names_flat[MAX_STEPS * MAX_STR_LEN];
__constant__ uint8_t c_pure_tag_lengths[MAX_STEPS];


struct abs_one_mapper {
    __host__ __device__
    int8_t operator()(int8_t x) const {
        return (x == -1 || x == 1) ? 1 : 0;
    }
};

struct abs_one_mapper_int {
    __host__ __device__
    int operator()(int8_t x) const {
        return (x == -1 || x == 1) ? 1 : 0;
    }
};

struct abs_one_mapper_int_optimized {
    __host__ __device__ 
    int operator()(int8_t x) const {
        return (x | -x) >> 7 & 1;   // 0 → 0, anything else → 1
    }
};


struct abs_one_mapper_uint8 {
    __host__ __device__
    uint8_t operator()(int8_t x) const {
        return (x == -1 || x == 1) ? 1 : 0;
    }
};

struct is_reset {
    __device__ int operator()(int8_t x) const {
        return (x == -1) ? 1 : 0;
    }
};

struct is_reset_tag_cond {
    __device__ int operator()(int8_t x) const {
        return (x == 2) ? 1 : 0;
    }
};

struct remove_neg_one {
    __device__ int8_t operator()(int8_t x) const {
        return (x == -1) ? 0 : x;
    }
};

struct is_curr_row {
    __device__ int8_t operator()(int8_t x) const {
        return x;
    }
};

struct mask_attr {
    __host__ __device__
    int8_t operator()(const thrust::tuple<thrust::tuple<int8_t, int8_t, int8_t, int8_t>, int8_t>& t) const {
        int8_t prev = thrust::get<0>(thrust::get<0>(t));
        int8_t curr = thrust::get<1>(thrust::get<0>(t));
        int8_t next = thrust::get<2>(thrust::get<0>(t));
        int8_t isStructural = thrust::get<3>(thrust::get<0>(t));
        int8_t final_val = thrust::get<1>(t);

        // Apply mask logic
        // int8_t cancel1 = (prev == 1 && next == 0);       // 1
        // int8_t cancel2 = (curr == 1 && next == 0);       // 1
        int8_t res1    = (prev == 1 && curr == 2);       // 1
        int8_t cancel = (curr == 1 && isStructural != 0); // when we have two consecutive 1s,


        // int8_t update = res1 * 1 + cancel1 * (-1) + cancel2 * 1 ;
        int8_t update = res1 * 1 + cancel * isStructural * (-1) ;
        // return isStructural;  // if case3, return 0, else return final_val + update
        return final_val + update;  // if case3, return 0, else return final_val + update
    }
};

struct index_mask_update_final {
    int k;   // target = 2*index_val + 1

    index_mask_update_final(int index_val) : k(2 * index_val + 1) {}

    __host__ __device__
    int8_t operator()(const thrust::tuple<int, int, int8_t>& t) const {
        int prev = thrust::get<0>(t);
        int curr = thrust::get<1>(t);
        int8_t final_val = thrust::get<2>(t);

        bool case1 = (prev == curr - 1) && (curr != k) && (curr % 2 ==  1);
        bool case2 = (prev == curr - 1) && (curr != k + 1) && (curr % 2 == 0);

        int8_t update = (case2 * 1) + (case1 * -1);  // +1 or -1
        return final_val + update;

    }
};

struct prev_index_lookup
{
    __host__ __device__
    int operator()(const int& x) const {
        return max(x - 1, 0);  // previous index or 0
    }
};

struct next_index_lookup_clamped {
    int max_index;
    next_index_lookup_clamped(int max_idx) : max_index(max_idx) {}

    __host__ __device__
    int operator()(const int& i) const {
        return (i + 1 < max_index) ? (i + 1) : i;  // clamp at last index
    }
};

struct edge_from_rtl_ltr {
    int count_2s;
    edge_from_rtl_ltr(int count_2s) : count_2s(count_2s) {}

    __host__ __device__
    int8_t operator()(const thrust::tuple<int, int, int, int, int, int8_t, int8_t>& t) const {
        int ltr_prev = thrust::get<0>(t);
        int ltr_curr = thrust::get<1>(t);

        int rtl_curr = thrust::get<2>(t);
        int rtl_next = thrust::get<3>(t);
        
        int segment_key = thrust::get<4>(t);  
        int8_t current_val = thrust::get<5>(t);
        int8_t final_val = thrust::get<6>(t);
        // int final_val = thrust::get<5>(t);

        bool rtl_case = (rtl_curr == 1 && rtl_next != 1) && (segment_key != count_2s);  // skip last chunk
        // bool rtl_case = (rtl_curr == 1 && rtl_next == 0) && (segment_key != count_2s);  // skip last chunk
        bool ltr_case = (ltr_prev != 1 && ltr_curr == 1) && (segment_key != 0);         // skip first chunk
        // bool ltr_case = (ltr_prev == 0 && ltr_curr == 1) && (segment_key != 0);         // skip first chunk

        // ltr
        bool case1 = !rtl_case && !ltr_case && (current_val == 1);
        bool case2 = !rtl_case && !ltr_case && (current_val == -1);

        int8_t val = (case1 * -1) + case2;
        return final_val + val;
    }
};

struct FlattenedQuery {
    int  depth[MAX_STEPS];


    char tag_names[MAX_STEPS][MAX_STR_LEN];
    char tag_ops[MAX_STEPS][MAX_OP_LEN];
    char tag_vals[MAX_STEPS][MAX_STR_LEN];
    char tag_cond_names[MAX_STEPS][MAX_STR_LEN];

    char attr_names[MAX_STEPS][MAX_STR_LEN];
    char attr_ops[MAX_STEPS][MAX_OP_LEN];
    char attr_vals[MAX_STEPS][MAX_STR_LEN];
    char attr_cond_names[MAX_STEPS][MAX_STR_LEN];

    char index_names[MAX_STEPS][MAX_STR_LEN];
    char index_ops[MAX_STEPS][MAX_OP_LEN];
    int  index_vals[MAX_STEPS];

    int  num_steps;  // actual number of steps
};

// Function to preprocess the XPath query
vector<vector<string>> preprocess_xpath(const string& xpath) {

    vector<vector<string>> result(12); // 12 rows

    /*
    Row Index | Meaning
            0 | Tag Name
            1 | Attribute Name
            2 | Attribute Condition Op (=)
            3 | Tag Condition Op (>, <)
            4 | Tag Condition Value
            5 | Attribute Condition Value
            6 | Depth
            7 | Attribute Cond Name
            8 | Tag Cond Name
            9 | index name
            10| index cond op
            11| index cond val
    */

    /* __________________________________________Step 1: Split by '/'__________________________________________ */
    /*
        Input: "/data/loc[area>500]/state"
        Result: ["data", "loc[area>500]", "state"]  
    */
    vector<string> tokens;
    stringstream ss(xpath);
    string item;
    while (getline(ss, item, '/')) {
        if (!item.empty()) tokens.push_back(item);
    }

#if defined(DEBUG_MODE) && DEBUG_MODE == 1
    // function for print tokens
    auto print_tokens = [](const vector<string>& tokens) {
        for (const auto& token : tokens) {
            std::cout << token << " ";
        }
        std::cout << std::endl;
    };

    std::cout << "Tokens after splitting by '/': ";
    print_tokens(tokens);
#endif
    /* __________________________________________Step 2: Process each token_____________________________________ */
    size_t query_token_size = tokens.size();

    // Looping through each token (a tag step like data, loc[area>500], or state)
    for (size_t i = 0; i < query_token_size; ++i) {
        const string& current_token = tokens[i];



        smatch match;

        // Match tag with optional predicate
        regex full_expr(R"((\w+)(\[(.*?)\])?)"); // Example: "loc[area>500]" → match[1]="loc" and match[3]="area>500"

        // Match pure index expression
        // regex pure_index_expr(R"(^\d+$)"); // Example: "loc[0]" → match[1]="loc" and match[3]="0"
        regex pure_index_expr(R"(^\d+$)");

        // Match index expression with position condition
        regex position_expr(R"(^(position)\(\)\s*([<>=!]+)\s*(\d+)$)");

        // Match last function with conditio
        regex last_expr(R"(^(last)\(\)\s*-\s*(\d+)$)");
        

        if (regex_match(current_token, match, full_expr)) {
            string tag_name = match[1];
            string predicate = match[3];
            result[0].push_back(tag_name);
            /* 
            Match[0]: data,     Match[1]:X,             Match[2]: X
            Match[0]: loc,      Match[1]:[area>500],    Match[2]: area>500
            Match[0]: state,    Match[1]:X,             Match[2]: X 
            */
            // cout << "Tag Name: " << tag_name << endl;
            // cout << "Predicate: " << predicate << endl;

            // Now parse predicate if exists
            if (!predicate.empty()) {
                // Match attribute
                regex attr_expr(R"(@(\w+)\s*([=])\s*'([^']*)')");

                // Match Condition
                // regex cond_expr(R"((\w+)\s*([<>=!]+)\s*([\w\-\+]+))");
                regex cond_expr(R"((\w+)\s*([<>=!]+)\s*(\w+))");

                // Match position or last function
                // regex pos_expr(R"((position|last)\(\)\s*([<>=!]+)\s*(\d+))");

                // Match attribute condition
                if (regex_match(predicate, match, attr_expr)) {
                    /*
                    Example: [@size='small'] --> Captures: attr name: size | op: = | val: small
                    */
                    // cout << "Attribute Name[1]: " << match[1] << endl;
                    // cout << "Attribute Name[2]: " << match[2] << endl;
                    // cout << "Attribute Name[3]: " << match[3] << endl;
                    result[1].push_back(match[1]);               // attr name
                    result[2].push_back(match[2]);               // attr op
                    result[5].push_back(match[3]);               // attr val
                    result[7].push_back(match[1]);               // attr condition name
                    result[3].push_back("");
                    result[4].push_back("");
                    result[8].push_back("");

                } else if (regex_match(predicate, match, cond_expr)) {
                    /*
                    Example: [area>500]  --> Captures: name: area | op: > | val: 500
                    */
                    // cout << "Tag Condition Name: " << match[1] << endl;
                    result[3].push_back(match[2]);               // tag cond op
                    result[4].push_back(match[3]);               // tag cond val
                    result[8].push_back(match[1]);               // tag cond name
                    result[1].push_back("");
                    result[2].push_back("");
                    result[5].push_back("");
                    result[7].push_back("");

                } 
                if (regex_match(predicate, match, pure_index_expr)) {
                    result[1].push_back("");
                    result[2].push_back("");
                    result[3].push_back("");               
                    result[4].push_back("");               
                    result[5].push_back("");
                    result[7].push_back("");
                    result[8].push_back("");               
                    result[9].push_back("");
                    result[10].push_back("");
                    result[11].push_back(predicate);  // the number itself
                } 
                else if (regex_match(predicate, match, position_expr)) {
                    result[1].push_back("");
                    result[2].push_back("");
                    result[3].push_back("");               
                    result[4].push_back("");               
                    result[5].push_back("");
                    result[7].push_back("");
                    result[8].push_back(""); 
                    result[9].push_back(match[1]);        // position
                    result[10].push_back(match[2]);     // >
                    result[11].push_back(match[3]);    // i
                } 
                else if (regex_match(predicate, match, last_expr)) {
                    result[1].push_back("");
                    result[2].push_back("");
                    result[3].push_back("");               
                    result[4].push_back("");               
                    result[5].push_back("");
                    result[7].push_back("");
                    result[8].push_back(""); 
                    result[9].push_back(match[1]);        // last
                    result[10].push_back("");          // -
                    result[11].push_back(match[2]);    // i
                }else {
                    // unknown format
                    // cout << "Unknown format: " << predicate << endl;
                    for (int r = 1; r <= 11; ++r) if(r!=6) result[r].push_back("");
                }
            } else {
                for (int r = 1; r <= 11; ++r) if(r!=6) result[r].push_back("");
            }

            result[6].push_back(to_string(i));  // depth: Stores the depth (0 for root, 1 for next step, etc.)      
        }
    }
    return result;
}

template<typename T>
void copy_str(const T& src, char* dest, size_t len) {
    strncpy(dest, src.c_str(), len - 1);
    dest[len - 1] = '\0';
}

FlattenedQuery flatten_query(const vector<vector<string>>& parsed) {
    // Function to flatten the parsed query into a GPU-friendly format
    // This function will be executed on the CPU
    // The parsed query is a 12-row matrix, and we will flatten it into a single structure
    // The flattened structure will be passed to the GPU for processing
    // The function takes the parsed query as input and returns a FlattenedQuery structure
    // The FlattenedQuery structure contains arrays for each field, with a maximum number of steps defined by MAX_STEPS
    FlattenedQuery fq{};
    size_t num_steps = parsed[0].size();
    fq.num_steps = num_steps;

    for (size_t i = 0; i < num_steps; ++i) {
        auto copy_str = [](const string& src, char* dest, size_t len) {
            strncpy(dest, src.c_str(), len-1);
            dest[len-1] = '\0';  // ensure null-termination
        };

        copy_str(parsed[0][i], fq.tag_names[i], MAX_STR_LEN);
        copy_str(parsed[1][i], fq.attr_names[i], MAX_STR_LEN);
        copy_str(parsed[2][i], fq.attr_ops[i], MAX_OP_LEN);
        copy_str(parsed[3][i], fq.tag_ops[i], MAX_OP_LEN);
        copy_str(parsed[4][i], fq.tag_vals[i], MAX_STR_LEN);
        copy_str(parsed[5][i], fq.attr_vals[i], MAX_STR_LEN);
        fq.depth[i] = stoi(parsed[6][i]);
        copy_str(parsed[7][i], fq.attr_cond_names[i], MAX_STR_LEN);
        copy_str(parsed[8][i], fq.tag_cond_names[i], MAX_STR_LEN);
        copy_str(parsed[9][i], fq.index_names[i], MAX_STR_LEN);
        copy_str(parsed[10][i], fq.index_ops[i], MAX_OP_LEN);
        fq.index_vals[i] = parsed[11][i].empty() ? -1 : stoi(parsed[11][i]);
    }

    return fq;
}

void analyze_query_conditions(const FlattenedQuery& fq, bool* has_attr_name, bool* has_tag_cond_name, bool* has_index_cond_val, bool* has_index_val) {
    *has_attr_name = false;
    *has_tag_cond_name = false;
    *has_index_cond_val = false;

    for (int i = 0; i < fq.num_steps; ++i) {
        if (fq.attr_names[i][0] != '\0') {
            *has_attr_name = true;
        }
        if (fq.tag_cond_names[i][0] != '\0') {
            *has_tag_cond_name = true;
        }
        if (fq.index_vals[i] != -1) {
            *has_index_val = true;
        }
        if(fq.index_names[i][0] != '\0') {
            *has_index_cond_val = true;
        }
    }
} 

// Kernel function for tag matching
__global__ void pure_tag_matching(
    const uint8_t* __restrict__ d_xmlContent,
    uint32_t xml_length,
    const uint32_t* __restrict__ d_token_indices,
    const uint8_t* __restrict__ d_token_values,
    const uint32_t* __restrict__ pair_pos,
    const uint32_t* __restrict__ transformed_depth,
    const uint32_t tokens_count,
    const FlattenedQuery* d_flattened_query,
    const int num_steps,
    const int* d_tag_lengths,                           // Precomputed tag lengths
    int8_t* d_output
) {
    size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;
    

    for(uint32_t k = index; k < tokens_count; k+=stride) {
        uint32_t current_depth     = __ldg(&transformed_depth[k]);
        // uint32_t current_depth = transformed_depth[k];
        if (current_depth >= num_steps) continue; // Skip if the current depth exceeds the number of steps
        
        // Opening-only fast path: close tokens are written via pair_pos.
        const uint8_t token_type = __ldg(&d_token_values[k]);
        if (token_type != '<') continue;
    
    
        const uint32_t pos = __ldg(&d_token_indices[k]) + 1;        // skip '<'
        const uint8_t tag_len = c_pure_tag_lengths[current_depth];   // Load tag length from constant cache, which is precomputed in the host and better than using shm 
        
        // Compare token in XML content agains the query tag name
        bool match = true;
        #pragma unroll 16
        for (uint8_t j = 0; j < tag_len; ++j) {
            // uint8_t query_char = s_tag_names[current_depth][j];
            uint8_t query_char = c_pure_tag_names_flat[current_depth * MAX_STR_LEN + j];
            uint8_t xml_char = ((pos + j) < xml_length) ? d_xmlContent[pos + j] : 0;

            // match &= (query_char == xml_char);  // Compare only within tag length
            if (query_char != xml_char) {
                match = false;
                break;
            }
        }
        uint8_t final_char = (pos + tag_len < xml_length) ? d_xmlContent[pos + tag_len] : 0;
        bool match_last_char = (final_char == '>' || final_char == ' ' || final_char == '/');
        match &= match_last_char;  // Ensure the tag ends with '>' or ' '

        if (!match) continue;

        d_output[k] = 1;
        const uint32_t paired_close_k = __ldg(&pair_pos[k]);
        if (paired_close_k < tokens_count) {
            // doing inline if statement to avoid branch divergence will cause performance degradation because of the more memory access
            d_output[paired_close_k] = -1;
        }

    }
}

// 4-byte vectorized tag compare version for the no-condition pure-tag path.
__global__ void pure_tag_matching_simd(
    const uint8_t* __restrict__ d_xmlContent,
    uint32_t xml_length,
    const uint32_t* __restrict__ d_token_indices,
    const uint8_t* __restrict__ d_token_values,
    const uint32_t* __restrict__ pair_pos,
    const uint32_t* __restrict__ transformed_depth,
    const uint32_t tokens_count,
    const FlattenedQuery* d_flattened_query,
    const int num_steps,
    const int* d_tag_lengths,
    int8_t* d_output
) {
    size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;

    for (uint32_t k = index; k < tokens_count; k += stride) {
        const uint32_t current_depth = __ldg(&transformed_depth[k]);
        if (current_depth >= static_cast<uint32_t>(num_steps)) continue;

        const uint8_t token_type = __ldg(&d_token_values[k]);
        if (token_type != '<') continue;

        const uint32_t pos = __ldg(&d_token_indices[k]) + 1u;
        const uint8_t tag_len = c_pure_tag_lengths[current_depth];
        const uint32_t q_base = current_depth * MAX_STR_LEN;

        // Guard once so vector and scalar compares can read safely without per-byte bounds checks.
        if (static_cast<uint64_t>(pos) + static_cast<uint64_t>(tag_len) >= static_cast<uint64_t>(xml_length)) continue;

        bool match = true;
        uint8_t j = 0;

        #pragma unroll 8
        for (; j + 4u <= tag_len; j += 4u) {
            const uint32_t q4 =
                static_cast<uint32_t>(c_pure_tag_names_flat[q_base + j + 0u]) |
                (static_cast<uint32_t>(c_pure_tag_names_flat[q_base + j + 1u]) << 8) |
                (static_cast<uint32_t>(c_pure_tag_names_flat[q_base + j + 2u]) << 16) |
                (static_cast<uint32_t>(c_pure_tag_names_flat[q_base + j + 3u]) << 24);
            const uint32_t x4 =
                static_cast<uint32_t>(__ldg(&d_xmlContent[pos + j + 0u])) |
                (static_cast<uint32_t>(__ldg(&d_xmlContent[pos + j + 1u])) << 8) |
                (static_cast<uint32_t>(__ldg(&d_xmlContent[pos + j + 2u])) << 16) |
                (static_cast<uint32_t>(__ldg(&d_xmlContent[pos + j + 3u])) << 24);

            if (__vcmpeq4(q4, x4) != 0xFFFFFFFFu) {
                match = false;
                break;
            }
        }

        #pragma unroll
        for (; match && j < tag_len; ++j) {
            if (c_pure_tag_names_flat[q_base + j] != __ldg(&d_xmlContent[pos + j])) {
                match = false;
            }
        }

        if (!match) continue;

        const uint8_t final_char = __ldg(&d_xmlContent[pos + tag_len]);
        if (!(final_char == '>' || final_char == ' ' || final_char == '/')) continue;

        d_output[k] = 1;
        const uint32_t paired_close_k = __ldg(&pair_pos[k]);
        if (paired_close_k < tokens_count) {
            d_output[paired_close_k] = -1;
        }
    }
}


// Reference tag kernel (conditional XPath path): constant-memory query tags, opening-only,
// paired close via pair_pos; writes per-depth row and OR row (same layout as legacy).
__global__ void tag_matching(
    const uint8_t* __restrict__ d_xmlContent,
    uint32_t xml_length,
    const uint32_t* __restrict__ d_token_indices,
    const uint8_t* __restrict__ d_token_values,
    const uint32_t* __restrict__ pair_pos,
    const uint32_t* __restrict__ transformed_depth,
    const uint32_t tokens_count,
    const FlattenedQuery* d_flattened_query,
    const int num_steps,
    const int* d_tag_lengths,
    int8_t* d_output
) {
    size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;
    const size_t or_row_base = static_cast<size_t>(num_steps) * tokens_count;

    for (uint32_t k = index; k < tokens_count; k += stride) {
        const uint32_t current_depth = __ldg(&transformed_depth[k]);
        if (current_depth >= static_cast<uint32_t>(num_steps)) continue;

        const uint8_t token_type = __ldg(&d_token_values[k]);
        if (token_type != '<') continue;

        const uint32_t pos = __ldg(&d_token_indices[k]) + 1u;
        const uint8_t tag_len = c_pure_tag_lengths[current_depth];

        bool match = true;
        #pragma unroll 16
        for (uint8_t j = 0; j < tag_len; ++j) {
            const uint8_t query_char = c_pure_tag_names_flat[current_depth * MAX_STR_LEN + j];
            const uint8_t xml_char = ((pos + j) < xml_length) ? __ldg(&d_xmlContent[pos + j]) : 0;
            if (query_char != xml_char) {
                match = false;
                break;
            }
        }
        const uint8_t final_char = (pos + tag_len < xml_length) ? __ldg(&d_xmlContent[pos + tag_len]) : 0;
        const bool match_last_char = (final_char == '>' || final_char == ' ' || final_char == '/');
        match &= match_last_char;

        if (!match) continue;

        const size_t row_base = static_cast<size_t>(current_depth) * tokens_count;
        d_output[row_base + k] = 1;
        d_output[or_row_base + k] = 1;

        const uint32_t close_k = __ldg(&pair_pos[k]);
        if (close_k < tokens_count) {
            d_output[row_base + close_k] = -1;
            d_output[or_row_base + close_k] = -1;
        }
    }
}

// Production tag kernel: same outputs as tag_matching, 4-byte vectorized name compare.
__global__ void tag_matching_simd(
    const uint8_t* __restrict__ d_xmlContent,
    uint32_t xml_length,
    const uint32_t* __restrict__ d_token_indices,
    const uint8_t* __restrict__ d_token_values,
    const uint32_t* __restrict__ pair_pos,
    const uint32_t* __restrict__ transformed_depth,
    const uint32_t tokens_count,
    const FlattenedQuery* d_flattened_query,
    const int num_steps,
    const int* d_tag_lengths,
    int8_t* d_output
) {
    size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;
    const size_t or_row_base = static_cast<size_t>(num_steps) * tokens_count;

    for (uint32_t k = index; k < tokens_count; k += stride) {
        const uint32_t current_depth = __ldg(&transformed_depth[k]);
        if (current_depth >= static_cast<uint32_t>(num_steps)) continue;

        const uint8_t token_type = __ldg(&d_token_values[k]);
        if (token_type != '<') continue;

        const uint32_t pos = __ldg(&d_token_indices[k]) + 1u;
        const uint8_t tag_len = c_pure_tag_lengths[current_depth];
        const uint32_t q_base = current_depth * MAX_STR_LEN;

        if (static_cast<uint64_t>(pos) + static_cast<uint64_t>(tag_len) >= static_cast<uint64_t>(xml_length)) continue;

        bool match = true;
        uint8_t j = 0;

        #pragma unroll 8
        for (; j + 4u <= tag_len; j += 4u) {
            const uint32_t q4 =
                static_cast<uint32_t>(c_pure_tag_names_flat[q_base + j + 0u]) |
                (static_cast<uint32_t>(c_pure_tag_names_flat[q_base + j + 1u]) << 8) |
                (static_cast<uint32_t>(c_pure_tag_names_flat[q_base + j + 2u]) << 16) |
                (static_cast<uint32_t>(c_pure_tag_names_flat[q_base + j + 3u]) << 24);
            const uint32_t x4 =
                static_cast<uint32_t>(__ldg(&d_xmlContent[pos + j + 0u])) |
                (static_cast<uint32_t>(__ldg(&d_xmlContent[pos + j + 1u])) << 8) |
                (static_cast<uint32_t>(__ldg(&d_xmlContent[pos + j + 2u])) << 16) |
                (static_cast<uint32_t>(__ldg(&d_xmlContent[pos + j + 3u])) << 24);

            if (__vcmpeq4(q4, x4) != 0xFFFFFFFFu) {
                match = false;
                break;
            }
        }

        #pragma unroll
        for (; match && j < tag_len; ++j) {
            if (c_pure_tag_names_flat[q_base + j] != __ldg(&d_xmlContent[pos + j])) {
                match = false;
            }
        }

        if (!match) continue;

        const uint8_t final_char = __ldg(&d_xmlContent[pos + tag_len]);
        if (!(final_char == '>' || final_char == ' ' || final_char == '/')) continue;

        const size_t row_base = static_cast<size_t>(current_depth) * tokens_count;
        d_output[row_base + k] = 1;
        d_output[or_row_base + k] = 1;

        const uint32_t close_k = __ldg(&pair_pos[k]);
        if (close_k < tokens_count) {
            d_output[row_base + close_k] = -1;
            d_output[or_row_base + close_k] = -1;
        }
    }
}


__global__ void attribute_matching(
    const uint8_t* d_xmlContent,
    uint32_t xml_length,
    const uint32_t* d_token_indices,
    const uint8_t* d_token_values,
    uint32_t* transformed_depth,
    const uint32_t tokens_count,
    const FlattenedQuery* d_flattened_query,
    const uint8_t* d_attr_lengths,             // Attribute lengths (query-side)
    const uint8_t* d_attr_depth_lookup,        // Lookup table for attribute depths
    const uint8_t* d_attr_val_lengths,         // Attribute value lengths (query-side)
    uint8_t num_attr_depths,                   // Number of attribute depths
    int8_t* d_attribute_output             // Output: attr_idx * tokens_count
) {
    size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;

    __shared__ uint8_t s_attr_depth_lookup[MAX_ATTR_DEPTHS];
    __shared__ uint8_t s_attr_lengths[MAX_ATTR_DEPTHS];
    __shared__ uint8_t s_attr_val_lengths[MAX_ATTR_DEPTHS];
    if (threadIdx.x == 0) {
        for (size_t i = 0; i < num_attr_depths; ++i) {
            s_attr_depth_lookup[i] = d_attr_depth_lookup[i];
            s_attr_lengths[i] = d_attr_lengths[i];              // copy precomputed lengths
            s_attr_val_lengths[i] = d_attr_val_lengths[i];      // copy precomputed lengths
        }
    }
    __syncthreads();

    for(uint32_t k = index; k < tokens_count; k+=stride) {
        if (d_token_values[k] != '=') continue; // Only process '=' tokens (attributes)

        // Find if current_depth is in the attribute depth lookup
        uint32_t current_depth = transformed_depth[k] - 1;
        int8_t attr_idx = -1;
        for (uint8_t i = 0; i < num_attr_depths; ++i) {
            if (s_attr_depth_lookup[i] == current_depth) {
                attr_idx = (int8_t) i;
                break;
            }
        }
        if (attr_idx == -1) continue; // Not an attribute depth


        // Step 1: Match Attribute Key
        bool attr_key_match = true;
        uint8_t attr_key_len = s_attr_lengths[attr_idx];
        // printf("attr_len = %d\n", attr_key_len);

        uint32_t current_token_indices = d_token_indices[k]; // Look back before '='
        #pragma unroll
        for (uint8_t j = 0; j < attr_key_len; ++j) {
            uint32_t pos = (current_token_indices >= attr_key_len - j) ? current_token_indices - attr_key_len + j : 0;
            uint8_t xml_char = (pos < xml_length) ? d_xmlContent[pos] : 0;
            uint8_t query_char = d_flattened_query->attr_names[current_depth][j];
            attr_key_match &= (xml_char == query_char);
        }
        if (!attr_key_match) continue;

        // Step 2: Match Attribute Value
        bool attr_value_match = false;

        
        uint32_t value_start_pos = current_token_indices + 2;               // Start reading after '=' and quote -> skip '=' and '"'
        uint8_t query_value_len = s_attr_val_lengths[attr_idx];            // Compute query value length
        // printf("query_value_len = %d\n", query_value_len);
        // Compare extracted value with query value
        int compare_result = 0;                                             // 0 = equal, <0 = less, >0 = greater
        #pragma unroll
        for (uint8_t j = 0; j < query_value_len; ++j) {
            uint8_t xml_char = (value_start_pos + j < xml_length) ? d_xmlContent[value_start_pos + j] : 0;
            uint8_t query_char = d_flattened_query->attr_vals[current_depth][j]; // changed
            if (xml_char != query_char) {
                compare_result = (xml_char < query_char) ? -1 : 1;
                break;
            }
        }



        // Handle different operators
        uint8_t op0 = d_flattened_query->attr_ops[current_depth][0];
        uint8_t op1 = d_flattened_query->attr_ops[current_depth][1];
        // All possible comparisons
        int8_t eq_case  = (op0 == '=' && op1 == '\0') * (compare_result == 0);
        int8_t neq_case = (op0 == '!' && op1 == '=') * (compare_result != 0);
        int8_t lt_case  = (op0 == '<' && op1 == '\0') * (compare_result < 0);
        int8_t gt_case  = (op0 == '>' && op1 == '\0') * (compare_result > 0);

        // Final OR
        attr_value_match = eq_case | neq_case | lt_case | gt_case;
        d_attribute_output[current_depth * tokens_count + k] = attr_value_match ? 1 : 0;

        // Step 3: Write final result
        // if(attr_value_match){
        //     // d_attribute_output[current_depth * tokens_count + k] = 1;
        //     printf("ThredIdx: %d, attr_idx: %d, k: %d, current_depth: %d, attr_key_len: %d, query_value_len: %d\n", 
        //         threadIdx.x, attr_idx, k, current_depth, attr_key_len, query_value_len);
        //     printf("XML Content: %.*s\n", query_value_len, d_xmlContent + value_start_pos);
        //     printf("Query Value: %.*s\n", query_value_len, d_flattened_query->attr_vals[current_depth]);
        //     printf("Operator: %c%c\n", op0, op1);
        //     printf("Compare Result: %d\n", compare_result);

        // }
        
    }
}

__global__ void tag_condition_matching(
    const uint8_t* d_xmlContent,
    uint32_t xml_length,
    const uint32_t* d_token_indices,
    const uint8_t* d_token_values,
    uint32_t* transformed_depth,
    const uint32_t tokens_count,
    const FlattenedQuery* d_flattened_query,
    const uint8_t* d_tag_cond_lengths,              // Attribute lengths (query-side)
    const uint8_t* d_tag_cond_depth_lookup,         // Lookup table for attribute depths
    const int* d_tag_cond_val_lengths,          // Attribute value lengths (query-side)
    const uint8_t num_tag_cond_depths,                    // Number of attribute depths
    int8_t* d_tag_cond_output                   // Output: attr_idx * tokens_count
) {
    size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;

    __shared__ uint8_t s_depth_lookup[MAX_TAG_COND_DEPTHS];
    __shared__ uint8_t s_key_lengths[MAX_TAG_COND_DEPTHS];
    __shared__ int s_val_lengths[MAX_TAG_COND_DEPTHS];

    if (threadIdx.x == 0) {
        for (size_t i = 0; i < num_tag_cond_depths; ++i) {
            s_depth_lookup[i] = d_tag_cond_depth_lookup[i];
            s_key_lengths[i] = d_tag_cond_lengths[i];
            s_val_lengths[i] = d_tag_cond_val_lengths[i];
        }
    }
    __syncthreads();

    for(uint32_t k = index; k < tokens_count; k+=stride) {
        if (d_token_values[k] != '/') continue;  // Only process closing tags

        // Find if current_depth is in the attribute depth lookup
        uint32_t current_depth = transformed_depth[k] - 1;
        
        int8_t cond_idx = -1;
        for (size_t i = 0; i < num_tag_cond_depths; ++i) {
            if (s_depth_lookup[i] == current_depth) {
                cond_idx = (int8_t) i;
                break;
            }
        }
        if (cond_idx == -1) continue;



        // Step 1: Match Attribute Key
        // Step 1: Match tag name
        bool tag_match = true;
        uint8_t tag_len = s_key_lengths[cond_idx];
        uint32_t tag_pos = d_token_indices[k] + 1;  // skip '</' //?

        #pragma unroll
        for (uint8_t j = 0; j < tag_len; ++j) {
            if ((tag_pos + j) >= xml_length) {
                tag_match = false;
                break;
            }
            uint8_t xml_char = d_xmlContent[tag_pos + j];
            uint8_t query_char = d_flattened_query->tag_cond_names[current_depth][j];
            if (xml_char != query_char) {
                tag_match = false;
                break;
            }
        }
        if (!tag_match) continue;


        // Step 2: Compare value before closing tag
        int value_len = s_val_lengths[cond_idx];
        uint32_t value_start = (d_token_indices[k] >= value_len) ? d_token_indices[k] - 1 - value_len : 0;

        int cmp = 0;  // strcmp-style result
        for (int j = 0; j < value_len; ++j) {
            uint8_t xml_char = (value_start + j < xml_length) ? d_xmlContent[value_start + j] : 0;
            uint8_t query_char = d_flattened_query->tag_vals[current_depth][j];
            if (xml_char != query_char) {
                cmp = (xml_char < query_char) ? -1 : 1;
                break;
            }
        }

        // Step 3: Operator logic
        uint8_t op0 = d_flattened_query->tag_ops[current_depth][0];
        uint8_t op1 = d_flattened_query->tag_ops[current_depth][1];

        bool match = false;
        match |= (op0 == '=' && op1 == '\0' && cmp == 0);
        match |= (op0 == '!' && op1 == '=' && cmp != 0);
        match |= (op0 == '<' && op1 == '\0' && cmp < 0);
        match |= (op0 == '>' && op1 == '\0' && cmp > 0);

        if (match) {
            d_tag_cond_output[current_depth * tokens_count + k] = 2;
        }
    }
}

struct is_query_output {
    int target_val;

    is_query_output(int _target) : target_val(_target) {}

    __host__ __device__
    uint8_t operator()(thrust::tuple<int8_t, int8_t> t) const {
        int8_t prev_val = thrust::get<0>(t);
        int8_t curr_val = thrust::get<1>(t);

        return (curr_val == target_val) && (prev_val != target_val) || (curr_val == target_val - 1) && (prev_val == target_val);
    }
};


// Function to generate output flags for chain boundaries
void generate_output_flags(
    const int8_t* d_structural_output,
    int tokens_count,
    int target_val,
    uint8_t** d_output_flag_ptr
) {
    // Allocate output flag array
    cudaMalloc((void**)d_output_flag_ptr, tokens_count * sizeof(uint8_t));
    uint8_t* d_output_flag = *d_output_flag_ptr;

    // Define iterators
    auto prev_iter = thrust::device_pointer_cast(d_structural_output);
    auto curr_iter = thrust::device_pointer_cast(d_structural_output + 1);

    auto zipped_begin = thrust::make_zip_iterator(thrust::make_tuple(prev_iter, curr_iter));
    auto zipped_end = zipped_begin + (tokens_count - 2);

    cout << "target_val: " << target_val << endl;
    // Apply transform (skip first and last elements)
    thrust::transform(
        thrust::cuda::par,
        zipped_begin,
        zipped_end,
        thrust::device_pointer_cast(d_output_flag + 1),  // write from second element
        is_query_output(target_val)
    );

    // Handle first element (index 0)
    int8_t first_val;
    cudaMemcpy(&first_val, d_structural_output, sizeof(int8_t), cudaMemcpyDeviceToHost);
    uint8_t first_flag = (first_val == target_val);
    cudaMemcpy(d_output_flag, &first_flag, sizeof(uint8_t), cudaMemcpyHostToDevice);

    // Handle last element (index tokens_count - 1)
    int8_t last_val;
    cudaMemcpy(&last_val, d_structural_output + (tokens_count - 1), sizeof(int8_t), cudaMemcpyDeviceToHost);
    uint8_t last_flag = (last_val == target_val);
    cudaMemcpy(d_output_flag + (tokens_count - 1), &last_flag, sizeof(uint8_t), cudaMemcpyHostToDevice);
}

void stage4_xpath(
    const uint32_t* d_token_indices,
    const uint8_t* d_token_values,
    const uint32_t tokens_count,
    uint32_t xml_length,
    const uint8_t* d_xmlContent,
    uint32_t** pair_pos, 
    uint32_t** transformed_depth, 
    const uint8_t* xpath_query,  
    uint32_t** d_selected_token_indices,                // output: matches indices
    size_t* matched_tokens                                // Output: how many tokens matched
){


    // _____________________________Xpath Preprocessing_____________________________
    // This function will be executed on the CPU
    // Implement the logic for preprocessing the XPath query here
    // For example, you might want to tokenize the XPath query and prepare it for processing
    // For now, we will just print the XPath query
    // The 12-row matrix format is super flexible and GPU-friendly.
    #if defined(DEBUG_MODE) && DEBUG_MODE == 1
        std::cout << "PreProcessing XPath query: " << xpath_query << std::endl;
    #endif

    #if defined(DEBUG_MODE) && DEBUG_MODE == 2
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);
    #endif
    
    std::string xpath(reinterpret_cast<const char*>(xpath_query));
    auto parsed_query = preprocess_xpath(xpath);

    // _____________________________Vector_Flatten_____________________________
    // let's flatten our 12-row matrix into a fixed-size, GPU-friendly format!
    // target GPU Data Layout
    /*
        ________Field________   ___Type___	    _____Example_Values_____
        tag_names[N][16]	    char array	    "data", "loc", "state"
        attr_names[N][16]	    char array	    "size", ...
        attr_ops[N][4]	        char array	    "=", ...
        tag_ops[N][4]	        char array	    ">", ...
        tag_vals[N][16]	        char array	    "500", ...
        attr_vals[N][16]	    char array	    "small", ...
        depth[N]	            int	            0, 1, 2
        attr_cond_names[N][16]	char array	    "size"
        tag_cond_names[N][16]	char array	    "area"
        index_names[N][16]	    char array	    "position", "last", ...
        index_ops[N][4]	        char array	    ">", "-"
        index_vals[N]	        int	            3, 1, ...
    */

    #if defined(DEBUG_MODE) && DEBUG_MODE == 2
        // Measure the time taken for the kernel execution
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float time_ms = 0.0f;
        cudaEventElapsedTime(&time_ms, start, stop);
        
        double time_ns = static_cast<uint64_t>(time_ms * 1e6);
        std::cout << "⏱️ preprocess_xpath execution time: " << time_ns << " ns" << std::endl;
        std::cout << "⏱️ preprocess_xpath Transform execution time: " << time_ns / 1e6 << " ms" << std::endl;

        // Cleanup
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    #endif

    #if defined(DEBUG_MODE) && DEBUG_MODE == 2
        // Measure the time taken for the kernel execution
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);
    #endif

    FlattenedQuery fq = flatten_query(parsed_query);

    bool has_attr_name = false;
    bool has_tag_cond_name = false;
    bool has_index_cond_val = false;
    bool has_index_val = false;
    analyze_query_conditions(fq, &has_attr_name, &has_tag_cond_name, &has_index_cond_val, &has_index_val);


    #if defined(DEBUG_MODE) && DEBUG_MODE == 2
            // Measure the time taken for the kernel execution
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);

            time_ms = 0.0f;
            cudaEventElapsedTime(&time_ms, start, stop);

            time_ns = static_cast<uint64_t>(time_ms * 1e6);
            std::cout << "⏱️ flatten execution time: " << time_ns << " ns" << std::endl;
            std::cout << "⏱️ flatten execution time: " << time_ns / 1e6 << " ms" << std::endl;

            // Cleanup
            cudaEventDestroy(start);
            cudaEventDestroy(stop);
    #endif
    #if defined(DEBUG_MODE) && DEBUG_MODE == 1
        print_parsed_query(parsed_query);
        // Print flattened result
        std::cout << "Flattened Query (Test Output):\n";
        for (int i = 0; i < fq.num_steps; ++i) {
            std::cout << "Step " << i << ": ";
            std::cout << "Tag=" << fq.tag_names[i] << ", ";
            std::cout << "Attr=" << fq.attr_names[i] << ", ";
            std::cout << "AttrOp=" << fq.attr_ops[i] << ", ";
            std::cout << "TagOp=" << fq.tag_ops[i] << ", ";
            std::cout << "TagVal=" << fq.tag_vals[i] << ", ";
            std::cout << "Depth=" << fq.depth[i] << ", ";
            std::cout << "TagCondName=" << fq.tag_cond_names[i] << ", ";
            std::cout << "IndexName=" << fq.index_names[i] << ", ";
            std::cout << "IndexOp=" << fq.index_ops[i] << ", ";
            std::cout << "IndexVal=" << fq.index_vals[i] << std::endl;
        }
        std::cout << "Num Steps: " << fq.num_steps << std::endl;
        std::cout << "Flattening completed." << std::endl;
    #endif

    // _____________________________GPU Memory Allocation_____________________________
    // Allocate memory on the GPU for the flattened query
    FlattenedQuery* d_flattened_query;
    cudaMalloc((void**)&d_flattened_query, sizeof(FlattenedQuery));
    cudaMemcpy(d_flattened_query, &fq, sizeof(FlattenedQuery), cudaMemcpyHostToDevice);

    // Allocate memory for the output
    int8_t* d_output;
    const uint32_t tokens_count_uint8 = ( tokens_count + BLOCKSIZE - 1) / BLOCKSIZE;
    int8_t* d_structural_output; 

    
    #if defined(DEBUG_MODE) && DEBUG_MODE == 5
        printGpuMemoryUsage("After Query Parser");
    #endif

    const bool has_any_condition = (has_attr_name || has_tag_cond_name || has_index_cond_val || has_index_val);
    const int debug_output_rows = has_any_condition ? (fq.num_steps + 1) : 1;

    if(!has_any_condition){
        // No conditions to match, just return the original token indices
        cudaMalloc((void**)&d_output, sizeof(int8_t) * tokens_count);    // fq.num_steps-> rows: +1 for the OR of all of the previous rows 
        cudaMemset(d_output, 0, sizeof(int8_t) * tokens_count);          // Initialize output to 0

        // Tag metadata for pure_tag_matching goes to constant memory (shared by all blocks).
        uint8_t h_pure_tag_lengths[MAX_STEPS] = {0};
        uint8_t h_pure_tag_names_flat[MAX_STEPS * MAX_STR_LEN] = {0};
        for (int i = 0; i < fq.num_steps; ++i) {
            h_pure_tag_lengths[i] = static_cast<uint8_t>(strlen(fq.tag_names[i]));
            for (int j = 0; j < MAX_STR_LEN; ++j) {
                h_pure_tag_names_flat[i * MAX_STR_LEN + j] = static_cast<uint8_t>(fq.tag_names[i][j]);
            }
        }
        cudaMemcpyToSymbol(c_pure_tag_lengths, h_pure_tag_lengths, sizeof(h_pure_tag_lengths));
        cudaMemcpyToSymbol(c_pure_tag_names_flat, h_pure_tag_names_flat, sizeof(h_pure_tag_names_flat));

        #if defined(DEBUG_MODE) && DEBUG_MODE == 2
            float pure_tag_simd_ms = 0.0f;
            cudaEvent_t simd_start, simd_stop;
            cudaEventCreate(&simd_start);
            cudaEventCreate(&simd_stop);
            cudaEventRecord(simd_start);
        #endif

        // _____________________________Tag_Matching (SIMD timing)_____________________________

        // Run and time SIMD kernel.
        cudaMemset(d_output, 0, sizeof(int8_t) * tokens_count);
        pure_tag_matching_simd<<<tokens_count_uint8, BLOCKSIZE>>>(
            d_xmlContent, xml_length, d_token_indices, (uint8_t*) d_token_values, *pair_pos, *transformed_depth, tokens_count,
            d_flattened_query, fq.num_steps, nullptr, d_output);
        cudaDeviceSynchronize();
            
        d_structural_output  = d_output; // No need to allocate a separate output for structural matching
        #if defined(DEBUG_MODE) && DEBUG_MODE == 2
            cudaEventRecord(simd_stop);
            cudaEventSynchronize(simd_stop);
            cudaEventElapsedTime(&pure_tag_simd_ms, simd_start, simd_stop);
            cudaEventDestroy(simd_start);
            cudaEventDestroy(simd_stop);
        #endif

        #if defined(DEBUG_MODE) && DEBUG_MODE == 5
            printGpuMemoryUsage("After Tag Matching");
        #endif
        #if defined(DEBUG_MODE) && DEBUG_MODE == 1
            std::cout << "Kernel execution completed_0." << std::endl;
            print_query_output(d_structural_output, 1, tokens_count, tokens_count);
        #endif
        #if defined(DEBUG_MODE) && DEBUG_MODE == 2
            time_ns = static_cast<uint64_t>(pure_tag_simd_ms * 1e6);
            std::cout << "⏱️ pure_tag_matching_simd execution time: " << time_ns << " ns" << std::endl;
            std::cout << "⏱️ pure_tag_matching_simd execution time: " << time_ns / 1e6 << " ms" << std::endl;
        #endif

    }else{
        cudaMalloc((void**)&d_output, sizeof(int8_t) * tokens_count * (fq.num_steps+1));    // fq.num_steps-> rows: +1 for the OR of all of the previous rows 
        cudaMemset(d_output, 0, sizeof(int8_t) * tokens_count * (fq.num_steps+1));          // Initialize output to 0

        // Tag Matching
        int h_tag_lengths[MAX_STEPS];

        // Attribute matching
        std::vector<uint8_t> h_attr_depths;
        std::vector<uint8_t> h_attr_lengths;
        std::vector<uint8_t> h_attr_val_lengths;

        std::vector<int> h_index_depths;
        std::vector<int> h_index_vals;

        std::vector<int> h_tag_cond_depths;
        std::vector<int> h_tag_cond_lengths;
        std::vector<int> h_tag_cond_vals_lengths;
    
    
        for (int i = 0; i < fq.num_steps; ++i) {
            if (fq.attr_names[i][0] != '\0') {                              // Check if attribute exists at this step
                h_attr_depths.push_back((uint8_t)i);                                 // Store the depth index
                h_attr_lengths.push_back((uint8_t) strlen(fq.attr_names[i]));         // Length of the attribute name
                // cout << "attr value: " << strlen(fq.attr_vals[i]) << endl;
                h_attr_val_lengths.push_back((uint8_t) strlen(fq.attr_vals[i]));      // attribute value length
            }
            if(fq.index_vals[i] != -1 && fq.index_names[i][0] == '\0'){
                h_index_depths.push_back(i);
                h_index_vals.push_back(fq.index_vals[i]);
            }
            if(fq.tag_cond_names[i][0] != '\0') {                              // Check if tag condition exists at this step
                h_tag_cond_depths.push_back((uint8_t) i);                                 // Store the depth index
                h_tag_cond_lengths.push_back((uint8_t) strlen(fq.tag_cond_names[i]));     // Length of the tag condition name
                h_tag_cond_vals_lengths.push_back(strlen(fq.tag_vals[i]));         // tag condition value length
            }
            h_tag_lengths[i] = strlen(fq.tag_names[i]);                     // compute tag length 

        }
        uint8_t num_attr_depths = h_attr_depths.size();
        int num_index_depths = h_index_depths.size();
        uint8_t num_tag_cond_depths = h_tag_cond_depths.size();

        // Copy to device: tag names (device buffer kept for compatibility; kernels use constant memory)
        int* d_tag_lengths;
        cudaMalloc(&d_tag_lengths, sizeof(int) * fq.num_steps);
        cudaMemcpy(d_tag_lengths, h_tag_lengths, sizeof(int) * fq.num_steps, cudaMemcpyHostToDevice);

        uint8_t h_pure_tag_lengths[MAX_STEPS] = {0};
        uint8_t h_pure_tag_names_flat[MAX_STEPS * MAX_STR_LEN] = {0};
        for (int ti = 0; ti < fq.num_steps; ++ti) {
            h_pure_tag_lengths[ti] = static_cast<uint8_t>(strlen(fq.tag_names[ti]));
            for (int tj = 0; tj < MAX_STR_LEN; ++tj) {
                h_pure_tag_names_flat[ti * MAX_STR_LEN + tj] = static_cast<uint8_t>(fq.tag_names[ti][tj]);
            }
        }
        cudaMemcpyToSymbol(c_pure_tag_lengths, h_pure_tag_lengths, sizeof(h_pure_tag_lengths));
        cudaMemcpyToSymbol(c_pure_tag_names_flat, h_pure_tag_names_flat, sizeof(h_pure_tag_names_flat));

        // Copy to device: attribute 
        uint8_t* d_attr_depth_lookup;
        cudaMalloc(&d_attr_depth_lookup, sizeof(uint8_t) * num_attr_depths);
        cudaMemcpy(d_attr_depth_lookup, h_attr_depths.data(), sizeof(uint8_t) * num_attr_depths, cudaMemcpyHostToDevice);

        uint8_t* d_attr_lengths;
        cudaMalloc(&d_attr_lengths, sizeof(uint8_t) * num_attr_depths);
        cudaMemcpy(d_attr_lengths, h_attr_lengths.data(), sizeof(uint8_t) * num_attr_depths, cudaMemcpyHostToDevice);
        // cout << "h_attr_lengths.size() = " << h_attr_lengths.size() << endl;
        
        uint8_t* d_attr_val_lengths;
        cudaMalloc(&d_attr_val_lengths, sizeof(uint8_t) * num_attr_depths);
        cudaMemcpy(d_attr_val_lengths, h_attr_val_lengths.data(), sizeof(uint8_t) * num_attr_depths, cudaMemcpyHostToDevice);

        // Copy to device: index
        int* d_index_depth_lookup;
        cudaMalloc(&d_index_depth_lookup, sizeof(int) * num_index_depths);
        cudaMemcpy(d_index_depth_lookup, h_index_depths.data(), sizeof(int) * num_index_depths, cudaMemcpyHostToDevice);

        // Copy to device: Tag condition
        uint8_t* d_tag_cond_depth_lookup;
        cudaMalloc(&d_tag_cond_depth_lookup, sizeof(uint8_t) * h_tag_cond_depths.size());
        cudaMemcpy(d_tag_cond_depth_lookup, h_tag_cond_depths.data(), sizeof(uint8_t) * h_tag_cond_depths.size(), cudaMemcpyHostToDevice);
        
        int* d_tag_cond_val_lengths;
        cudaMalloc(&d_tag_cond_val_lengths, sizeof(int) * h_tag_cond_depths.size());
        cudaMemcpy(d_tag_cond_val_lengths, h_tag_cond_vals_lengths.data(), sizeof(int) * h_tag_cond_depths.size(), cudaMemcpyHostToDevice);

        uint8_t* d_tag_cond_lengths; 
        cudaMalloc(&d_tag_cond_lengths, sizeof(uint8_t) * h_tag_cond_depths.size());
        cudaMemcpy(d_tag_cond_lengths, h_tag_cond_lengths.data(), sizeof(uint8_t) * h_tag_cond_depths.size(), cudaMemcpyHostToDevice);
        
        #if defined(DEBUG_MODE) && DEBUG_MODE == 2
                // Measure the time taken for the kernel execution
            cudaEventCreate(&start);
            cudaEventCreate(&stop);
            cudaEventRecord(start);
        #endif
        
        // _____________________________Tag_Matching (SIMD)_____________________________
        tag_matching_simd<<<tokens_count_uint8, BLOCKSIZE>>>(
            d_xmlContent, xml_length, d_token_indices, (uint8_t*) d_token_values, *pair_pos, *transformed_depth, tokens_count,
            d_flattened_query, fq.num_steps, nullptr, d_output);
        cudaDeviceSynchronize();


        #if defined(DEBUG_MODE) && DEBUG_MODE == 5
            printGpuMemoryUsage("After Tag Matching");
        #endif

        #if defined(DEBUG_MODE) && DEBUG_MODE == 2
            // Measure the time taken for the kernel execution
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);

            time_ms = 0.0f;
            cudaEventElapsedTime(&time_ms, start, stop);

            time_ns = static_cast<uint64_t>(time_ms * 1e6);
            std::cout << "⏱️ tag_matching_simd (else) execution time: " << time_ns << " ns" << std::endl;
            std::cout << "⏱️ tag_matching_simd (else) execution time: " << time_ns / 1e6 << " ms" << std::endl;

            // Cleanup
            cudaEventDestroy(start);
            cudaEventDestroy(stop);
        #endif
        #if defined(DEBUG_MODE) && DEBUG_MODE == 1
            std::cout << "Kernel attribute matching execution completed." << std::endl;
            print_query_output(d_output, fq.num_steps + 1, tokens_count, tokens_count);    
            
            std::cout << "=== Attribute Extraction Debug ===" << std::endl;
            for (int i = 0; i < fq.num_steps; ++i) {
                std::cout << "Step " << i << " — attr_name: \"" << fq.attr_names[i] << "\""
                        << ", attr_val: \"" << fq.attr_vals[i] << "\""
                        << ", strlen(attr_name): " << strlen(fq.attr_names[i])
                        << ", strlen(attr_val): " << strlen(fq.attr_vals[i]) << std::endl;
            }
            std::cout << "=== End Debug ===\n\n";

            std::cout << "===== Host-side Attribute Matching Info =====" << std::endl;
            std::cout << "Number of attribute depths: " << h_attr_depths.size() << std::endl;

            for (size_t i = 0; i < h_attr_depths.size(); ++i) {
                std::cout << "Attr Depth[" << i << "] = " << static_cast<int>(h_attr_depths[i])
                        << ", Attr Length = " << static_cast<int>(h_attr_lengths[i])
                        << ", Attr Val Length = " << static_cast<int>(h_attr_val_lengths[i])
                        << std::endl;
            }
            std::cout << "=============================================" << std::endl;
        #endif


        // DEBUG_MODE 6–10 (tag-condition branch only):
        //   6 = after tag_condition_matching kernel + full output matrix print
        //   7 = per-depth metadata + output_row for that tag-cond depth
        //   8 = reset_flags, d_segment_keys, abs_mask
        //   9 = d_ltr_output and d_rtl_output after segment scans
        //   10 = final_row after edge_from_rtl_ltr
        // DEBUG_MODE 1 runs steps 6–10 in order (same prints as 6|7|8|9|10 combined).

        // _____________________________Tag_Condition_Matching_____________________________
        if(has_tag_cond_name) {
            #if defined(DEBUG_MODE) && (DEBUG_MODE == 1 || DEBUG_MODE == 6)
                    std::cout << "Tag conditions found in the query." << std::endl;
            #endif
            

            #if defined(DEBUG_MODE) && DEBUG_MODE == 2
                // Measure the time taken for the kernel execution
                cudaEventCreate(&start);
                cudaEventCreate(&stop);
                cudaEventRecord(start);
            #endif
            tag_condition_matching<<<tokens_count_uint8, BLOCKSIZE>>>(
                d_xmlContent, xml_length, d_token_indices, (uint8_t*) d_token_values, *transformed_depth, tokens_count,
                d_flattened_query, d_tag_cond_lengths, d_tag_cond_depth_lookup, d_tag_cond_val_lengths, num_tag_cond_depths, d_output);
            cudaDeviceSynchronize();


            #if defined(DEBUG_MODE) && DEBUG_MODE == 5
                printGpuMemoryUsage("After Tag Cond Matching");
            #endif
            #if defined(DEBUG_MODE) && DEBUG_MODE == 2
                // Measure the time taken for the kernel execution
                cudaEventRecord(stop);
                cudaEventSynchronize(stop);

                time_ms = 0.0f;
                cudaEventElapsedTime(&time_ms, start, stop);
                
                time_ns = static_cast<uint64_t>(time_ms * 1e6);
                std::cout << "⏱️ tag_condition_matching execution time: " << time_ns << " ns" << std::endl;
                std::cout << "⏱️ tag_condition_matching execution time: " << time_ns / 1e6 << " ms" << std::endl;

                // Cleanup
                cudaEventDestroy(start);
                cudaEventDestroy(stop);
            #endif

            #if defined(DEBUG_MODE) && (DEBUG_MODE == 1 || DEBUG_MODE == 6)
                print_query_output(d_output, fq.num_steps + 1, tokens_count, tokens_count);
            #endif


            #if defined(DEBUG_MODE) && DEBUG_MODE == 2
                // Measure the time taken for the kernel execution
                cudaEventCreate(&start);
                cudaEventCreate(&stop);
                cudaEventRecord(start);
            #endif

            int* d_segment_keys;
            cudaMalloc(&d_segment_keys, sizeof(int) * tokens_count * 3);
            int* d_ltr_output = d_segment_keys + tokens_count;
            int* d_rtl_output = d_segment_keys + tokens_count * 2;
            

            for (uint8_t i = 0; i < num_tag_cond_depths; i++){
                /* code */
                uint8_t depth_idx = h_tag_cond_depths[i];  // the current depth
                int tag_cond_val_len = h_tag_cond_vals_lengths[i]; // the value to filter for


                int8_t* output_row_ptr = d_output + (depth_idx * tokens_count);
                int8_t* final_row_ptr = d_output + tokens_count * fq.num_steps;

                // Step 0: Count the number of 2s in the output_row_ptr
                // uint32_t count_ones_cub(uint8_t* d_flags, size_t length){
                uint32_t count_2s = reduce_cub_int(output_row_ptr, tokens_count) / 2; // Divide by 2 because we are counting pairs of 2s

                // Step 1: compute reset flags (1 where value==2 for tag cond)
                auto reset_flags_iter = thrust::make_transform_iterator(
                    thrust::device_pointer_cast(output_row_ptr),
                    is_reset_tag_cond()
                );

                #if defined(DEBUG_MODE) && (DEBUG_MODE == 1 || DEBUG_MODE == 7)
                    cudaDeviceSynchronize();
                    std::cout << "[tag_cond] iter=" << static_cast<int>(i)
                              << " depth_idx=" << static_cast<int>(depth_idx)
                              << " count_2s=" << count_2s
                              << " tag_cond_val_len=" << tag_cond_val_len << std::endl;
                    print_device_array(output_row_ptr, tokens_count, "tag_cond output_row (depth row)");
                #endif

                #if defined(DEBUG_MODE) && (DEBUG_MODE == 1 || DEBUG_MODE == 8)
                    cudaDeviceSynchronize();
                    print_thrust_iterator(reset_flags_iter, tokens_count, "reset_flags (is_reset_tag_cond)");
                #endif

                // Step 2: exclusive scan → segment keys
                thrust::exclusive_scan(
                    reset_flags_iter,
                    reset_flags_iter + tokens_count,
                    thrust::device_pointer_cast(d_segment_keys),
                    0
                );

                #if defined(DEBUG_MODE) && (DEBUG_MODE == 1 || DEBUG_MODE == 8)
                    cudaDeviceSynchronize();
                    print_device_array(d_segment_keys, tokens_count, "d_segment_keys");
                #endif

                // Step 3: create mask over abs values (1 for -1 or 1, else 0)
                auto abs_mask_iter = thrust::make_transform_iterator(
                    thrust::device_pointer_cast(output_row_ptr),
                    abs_one_mapper_int()
                );

                #if defined(DEBUG_MODE) && (DEBUG_MODE == 1 || DEBUG_MODE == 8)
                    cudaDeviceSynchronize();
                    print_thrust_iterator(abs_mask_iter, tokens_count, "abs_mask");
                #endif

                // Step 4: left-to-right scan by segment
                thrust::inclusive_scan_by_key(
                    thrust::device_pointer_cast(d_segment_keys),
                    thrust::device_pointer_cast(d_segment_keys + tokens_count),
                    abs_mask_iter,
                    thrust::device_pointer_cast(d_ltr_output)
                );

                // Step 5: right-to-left scan by segment
                auto rev_keys = thrust::make_reverse_iterator(thrust::device_pointer_cast(d_segment_keys + tokens_count));
                auto rev_vals = thrust::make_reverse_iterator(abs_mask_iter + tokens_count);
                auto rev_out  = thrust::make_reverse_iterator(thrust::device_pointer_cast(d_rtl_output + tokens_count));

                thrust::inclusive_scan_by_key(
                    rev_keys, rev_keys + tokens_count,
                    rev_vals,
                    rev_out
                );

                #if defined(DEBUG_MODE) && (DEBUG_MODE == 1 || DEBUG_MODE == 9)
                    cudaDeviceSynchronize();
                    print_device_array(d_ltr_output, tokens_count, "d_ltr_output");
                    print_device_array(d_rtl_output, tokens_count, "d_rtl_output");
                #endif

                // Step 6: combine left and right scans
                auto ltr_curr = thrust::device_pointer_cast(d_ltr_output);
                auto ltr_prev = thrust::make_permutation_iterator(
                    ltr_curr,
                    thrust::make_transform_iterator(thrust::counting_iterator<int>(0), prev_index_lookup())
                );

                auto rtl_curr = thrust::device_pointer_cast(d_rtl_output);
                auto rtl_next = thrust::make_permutation_iterator(
                    rtl_curr,
                    thrust::make_transform_iterator(thrust::counting_iterator<int>(0), next_index_lookup_clamped(tokens_count))
                );

                auto current_row = thrust::device_pointer_cast(output_row_ptr);
                auto final_iter = thrust::device_pointer_cast(final_row_ptr);

                auto zipped = thrust::make_zip_iterator(
                    thrust::make_tuple(ltr_prev, ltr_curr, rtl_curr, rtl_next, thrust::device_pointer_cast(d_segment_keys), current_row, final_iter)
                );


                cout << "count: " << count_2s << endl;
                // Run transform
                thrust::transform(
                    zipped, zipped + tokens_count,
                    final_iter,
                    edge_from_rtl_ltr(count_2s)
                );

                #if defined(DEBUG_MODE) && (DEBUG_MODE == 1 || DEBUG_MODE == 10)
                    cudaDeviceSynchronize();
                    print_device_array(final_row_ptr, tokens_count, "final_row after edge_from_rtl_ltr");
                #endif

                #if defined(DEBUG_MODE) && DEBUG_MODE == 5
                    printGpuMemoryUsage("After Tag Cond Matching 2");
                #endif

            }

            cudaFree(d_segment_keys);

            #if defined(DEBUG_MODE) && DEBUG_MODE == 5
                printGpuMemoryUsage("After Tag Cond Matching 3");
            #endif

            #if defined(DEBUG_MODE) && DEBUG_MODE == 2
                // Measure the time taken for the kernel execution
                cudaEventRecord(stop);
                cudaEventSynchronize(stop);

                time_ms = 0.0f;
                cudaEventElapsedTime(&time_ms, start, stop);
                
                time_ns = static_cast<uint64_t>(time_ms * 1e6);
                std::cout << "⏱️ tag_condition_masking execution time: " << time_ns << " ns" << std::endl;
                std::cout << "⏱️ tag_condition_masking execution time: " << time_ns / 1e6 << " ms" << std::endl;

                // Cleanup
                cudaEventDestroy(start);
                cudaEventDestroy(stop);
            #endif
            
        }
        
        // _____________________________Index_Matching_____________________________
        if(has_index_val) {
            #if defined(DEBUG_MODE) && DEBUG_MODE == 1
                std::cout << "Index conditions found in the query." << std::endl;
            #endif
            #if defined(DEBUG_MODE) && DEBUG_MODE == 2
                // Measure the time taken for the kernel execution
                cudaEventCreate(&start);
                cudaEventCreate(&stop);
                cudaEventRecord(start);
            #endif

            // Pre-allocate before loop:
            int* d_segment_keys;
            cudaMalloc(&d_segment_keys, sizeof(int) * tokens_count * 2);
            int* d_scan_output = d_segment_keys + tokens_count;             

            // std::cout << "here 2" << std::endl;
            for(int i = 0; i < num_index_depths; ++i) {
                int depth_idx = h_index_depths[i];     // the current depth
                int index_val = h_index_vals[i];                // the value to filter for
                // std::cout << "here 3" << std::endl;

                if(depth_idx == 0) {
                    std::cout << "not implemented yet." << std::endl;
                    exit(0);
                    // normal scan:
                    // to-do: add a check for the first row
                }else{
                    // std::cout << "here 5" << std::endl;
                    int8_t* output_row_ptr = d_output + (depth_idx * tokens_count);
                    int8_t* final_row_ptr = d_output + tokens_count * fq.num_steps;
                    int8_t* prev_row_ptr = d_output + (depth_idx - 1) * tokens_count;

                    #if defined(DEBUG_MODE) && DEBUG_MODE == 1
                        print_device_array(output_row_ptr, tokens_count, "output_row_ptr");
                        print_device_array(prev_row_ptr, tokens_count, "prev_row_ptr");
                    #endif

                    thrust::device_ptr<int> segment_keys_ptr(d_segment_keys);
                    thrust::device_ptr<int> scan_output_ptr(d_scan_output);

                    
                    // step 1: make parent ranges (1 to -1)
                    auto prev_abs_begin = thrust::make_transform_iterator(
                        thrust::device_pointer_cast(prev_row_ptr),
                        abs_one_mapper_int_optimized()
                    );


                    // step 2: exclusive scan by key to find the segments
                    thrust::exclusive_scan(
                        thrust::cuda::par,
                        prev_abs_begin,
                        prev_abs_begin + tokens_count,
                        d_segment_keys,  // your pre-allocated buffer
                        0
                    );

                    #if defined(DEBUG_MODE) && DEBUG_MODE == 1
                        print_device_array(d_segment_keys, tokens_count, "d_segment_keys");
                        print_thrust_iterator(prev_abs_begin, tokens_count, "prev_abs_begin (abs(prev_row_ptr))");
                    #endif

                    // abs(output_row) 
                    auto transformed_current_row_ptr = thrust::make_transform_iterator(
                        thrust::device_pointer_cast(output_row_ptr),
                        abs_one_mapper_int_optimized()
                    );

                    // Step 3: inclusive scan by key
                    thrust::inclusive_scan_by_key(
                        thrust::cuda::par,
                        segment_keys_ptr,
                        segment_keys_ptr + tokens_count,
                        transformed_current_row_ptr,
                        scan_output_ptr,
                        thrust::equal_to<int>(),
                        thrust::plus<int>()
                    );

                    #if defined(DEBUG_MODE) && DEBUG_MODE == 1
                        print_thrust_iterator(transformed_current_row_ptr, tokens_count, "transformed_current_row_ptr (abs(output_row_ptr))");
                    #endif


                    // Step 4: generate output flags for chain boundaries and update the output
                    auto prev_iter = thrust::make_permutation_iterator(
                        scan_output_ptr, thrust::make_transform_iterator(thrust::counting_iterator<int>(0), prev_index_lookup())
                    );
                    auto curr_iter = scan_output_ptr;
                    auto final_iter = thrust::device_pointer_cast(final_row_ptr);
            
                    auto zipped = thrust::make_zip_iterator(
                        thrust::make_tuple(prev_iter, curr_iter, final_iter)
                    );
            
                    thrust::transform(thrust::cuda::par, zipped, zipped + tokens_count, final_iter, index_mask_update_final(index_val));

                    #if defined(DEBUG_MODE) && DEBUG_MODE == 1
                        print_device_array(d_scan_output, tokens_count, "scan_output_ptr");
                        print_device_array(final_row_ptr, tokens_count, "final_row_ptr");
                    #endif


                    #if defined(DEBUG_MODE) && DEBUG_MODE == 5
                        printGpuMemoryUsage("After idx Matching 2");
                    #endif
                }
            }

            cudaFree(d_segment_keys);

            #if defined(DEBUG_MODE) && DEBUG_MODE == 5
                printGpuMemoryUsage("After idx Matching 3");
            #endif

            #if defined(DEBUG_MODE) && DEBUG_MODE == 2
                // Measure the time taken for the kernel execution
                cudaEventRecord(stop);
                cudaEventSynchronize(stop);

                time_ms = 0.0f;
                cudaEventElapsedTime(&time_ms, start, stop);

                time_ns = static_cast<uint64_t>(time_ms * 1e6);
                std::cout << "⏱️ has_index execution time: " << time_ns << " ns" << std::endl;
                std::cout << "⏱️ has_index execution time: " << time_ns / 1e6 << " ms" << std::endl;

                // Cleanup
                cudaEventDestroy(start);
                cudaEventDestroy(stop);
            #endif

            if(has_index_cond_val){
                std::cout << "Not Implemented: Index conditions found in the query." << std::endl;
                exit(0);
            }


        }

        // _____________________________Attribute_Matching_____________________________
        if(has_attr_name) {
            #if defined(DEBUG_MODE) && DEBUG_MODE == 1
                std::cout << "Attribute conditions found in the query." << std::endl;
            #endif
            


            #if defined(DEBUG_MODE) && DEBUG_MODE == 2
                // Measure the time taken for the kernel execution
                cudaEventCreate(&start);
                cudaEventCreate(&stop);
                cudaEventRecord(start);
            #endif
            attribute_matching<<<tokens_count_uint8, BLOCKSIZE>>>(
                d_xmlContent, xml_length, d_token_indices, (uint8_t*) d_token_values, *transformed_depth, tokens_count,
                d_flattened_query, d_attr_lengths, d_attr_depth_lookup, d_attr_val_lengths, num_attr_depths, d_output);
            cudaDeviceSynchronize();

            #if defined(DEBUG_MODE) && DEBUG_MODE == 5
                printGpuMemoryUsage("After Attribute Cond Matching");
            #endif
            #if defined(DEBUG_MODE) && DEBUG_MODE == 2
                // Measure the time taken for the kernel execution
                cudaEventRecord(stop);
                cudaEventSynchronize(stop);

                time_ms = 0.0f;
                cudaEventElapsedTime(&time_ms, start, stop);
                
                time_ns = static_cast<uint64_t>(time_ms * 1e6);
                std::cout << "⏱️ attribute_matching execution time: " << time_ns << " ns" << std::endl;
                std::cout << "⏱️ attribute_matching execution time: " << time_ns / 1e6 << " ms" << std::endl;

                // Cleanup
                cudaEventDestroy(start);
                cudaEventDestroy(stop);
            #endif

            #if defined(DEBUG_MODE) && DEBUG_MODE == 1
                std::cout << "[attr] after attribute_matching kernel (full d_output)" << std::endl;
                print_query_output(d_output, fq.num_steps + 1, tokens_count, tokens_count);
            #endif

            #if defined(DEBUG_MODE) && DEBUG_MODE == 2
                // Measure the time taken for the kernel execution
                cudaEventCreate(&start);
                cudaEventCreate(&stop);
                cudaEventRecord(start);
            #endif  

            // int* reset_flags_ptr;
            // cudaMalloc((void**)&reset_flags_ptr, sizeof(int) * tokens_count * 2);
            // int* segment_keys_ptr = reset_flags_ptr + tokens_count;                 // 2nd half of the buffer
            // cudaMalloc((void**)&segment_keys_ptr, sizeof(int) * tokens_count);

            int* segment_keys_ptr;
            cudaMalloc((void**)&segment_keys_ptr, sizeof(int) * tokens_count);

            int8_t* copy_curr_row;
            cudaMalloc((void**)&copy_curr_row, sizeof(int8_t) * tokens_count); // buffer for copying current row

            #if defined(DEBUG_MODE) && DEBUG_MODE == 1
                std::cout << "[attr] allocated segment_keys_ptr and copy_curr_row; starting attribute masking loop"
                          << " (num_attr_depths=" << static_cast<int>(num_attr_depths) << ")" << std::endl;
            #endif

            // 3rd kernel call - attribute masking:
            for (size_t i = 0; i < num_attr_depths; ++i) {
                const int8_t depth_idx = h_attr_depths[i];  // depth index where attribute exists
                int8_t* output_row_ptr = d_output + (depth_idx * tokens_count);

                #if defined(DEBUG_MODE) && DEBUG_MODE == 1
                    std::cout << "[attr] --- iter " << i << " depth_idx=" << static_cast<int>(depth_idx) << " ---" << std::endl;
                    cudaDeviceSynchronize();
                    print_device_array(output_row_ptr, tokens_count, "[attr] output_row (depth row, before masking steps)");
                #endif

                // Step 1: compute reset flags (1 where -1, else 0)
                auto reset_iter = thrust::make_transform_iterator(
                    thrust::device_pointer_cast(output_row_ptr),
                    is_reset()  /* your functor: -1→1 else→0 */
                );

                #if defined(DEBUG_MODE) && DEBUG_MODE == 1
                    cudaDeviceSynchronize();
                    print_thrust_iterator(reset_iter, tokens_count, "[attr] reset_iter (is_reset: 1 at -1)");
                #endif

                // // Step 2: exclusive scan → segment keys
                thrust::exclusive_scan(
                    thrust::cuda::par,
                    reset_iter,
                    reset_iter + tokens_count,
                    thrust::device_pointer_cast(segment_keys_ptr),
                    0    /* init */
                );

                #if defined(DEBUG_MODE) && DEBUG_MODE == 1
                    cudaDeviceSynchronize();
                    print_device_array(segment_keys_ptr, tokens_count, "[attr] segment_keys (exclusive_scan of reset)");
                #endif

                // Step 3: mask out -1s to be zeros before scan
                auto masked_input = thrust::make_transform_iterator(
                    thrust::device_pointer_cast(output_row_ptr),
                    remove_neg_one()           // x→(x<0?0:x)
                );

                #if defined(DEBUG_MODE) && DEBUG_MODE == 1
                    cudaDeviceSynchronize();
                    print_thrust_iterator(masked_input, tokens_count, "[attr] masked_input (remove_neg_one)");
                #endif

                // Step : mask out -1s to be zeros before scan
                thrust::copy(
                    thrust::device_pointer_cast(output_row_ptr),
                    thrust::device_pointer_cast(output_row_ptr + tokens_count),
                    thrust::device_pointer_cast(copy_curr_row)
                );

                #if defined(DEBUG_MODE) && DEBUG_MODE == 1
                    cudaDeviceSynchronize();
                    print_device_array(copy_curr_row, tokens_count, "[attr] copy_curr_row (snapshot before inclusive_scan_by_key)");
                #endif

                // Step 4: run inclusive scan by segment key
                thrust::inclusive_scan_by_key(
                    thrust::device_pointer_cast(segment_keys_ptr), thrust::device_pointer_cast(segment_keys_ptr + tokens_count), masked_input,
                    thrust::device_pointer_cast(output_row_ptr),
                    thrust::equal_to<int>(), thrust::plus<int8_t>()
                );

                #if defined(DEBUG_MODE) && DEBUG_MODE == 1
                    cudaDeviceSynchronize();
                    print_device_array(output_row_ptr, tokens_count, "[attr] output_row after inclusive_scan_by_key");
                #endif
                
                // Step 5: compute and accumulate final result
                auto prev_iter = thrust::make_permutation_iterator(
                    thrust::device_pointer_cast(output_row_ptr),
                    thrust::make_transform_iterator(thrust::counting_iterator<int>(0), prev_index_lookup())
                );
                auto curr_iter = thrust::device_pointer_cast(output_row_ptr);
                auto next_iter = thrust::make_permutation_iterator(
                    thrust::device_pointer_cast(output_row_ptr),
                    thrust::make_transform_iterator(thrust::counting_iterator<int>(0), next_index_lookup_clamped(tokens_count))
                );

                auto copy_iter = thrust::device_pointer_cast(copy_curr_row);
                auto zipped_pcn = thrust::make_zip_iterator(
                    thrust::make_tuple(prev_iter, curr_iter, next_iter, copy_iter)
                );

                int8_t* final_row_ptr = d_output + tokens_count * fq.num_steps;
                auto zipped_full = thrust::make_zip_iterator(
                    thrust::make_tuple(zipped_pcn, thrust::device_pointer_cast(final_row_ptr))
                );

                thrust::transform(
                    zipped_full, zipped_full + tokens_count,
                    thrust::device_pointer_cast(final_row_ptr),
                    mask_attr()
                );

                #if defined(DEBUG_MODE) && DEBUG_MODE == 1
                    cudaDeviceSynchronize();
                    print_device_array(final_row_ptr, tokens_count, "[attr] final_row after mask_attr (OR row)");
                #endif

                #if defined(DEBUG_MODE) && DEBUG_MODE == 5
                    printGpuMemoryUsage("After Attribute Cond Matching 2");
                #endif
            }
            cudaFree(segment_keys_ptr);
            cudaFree(copy_curr_row);


            #if defined(DEBUG_MODE) && DEBUG_MODE == 5
                printGpuMemoryUsage("After Attribute Cond Matching 3");
            #endif
            #if defined(DEBUG_MODE) && DEBUG_MODE == 2
                // Measure the time taken for the kernel execution
                cudaEventRecord(stop);
                cudaEventSynchronize(stop);

                time_ms = 0.0f;
                cudaEventElapsedTime(&time_ms, start, stop);
                
                time_ns = static_cast<uint64_t>(time_ms * 1e6);
                std::cout << "⏱️ attribute_masking execution time: " << time_ns << " ns" << std::endl;
                std::cout << "⏱️ attribute_masking execution time: " << time_ns / 1e6 << " ms" << std::endl;

                // Cleanup
                cudaEventDestroy(start);
                cudaEventDestroy(stop);
            #endif 
                    
            #if defined(DEBUG_MODE) && DEBUG_MODE == 1
                cudaDeviceSynchronize();
                std::cout << "[attr] attribute masking loop finished; full d_output" << std::endl;
                print_query_output(d_output, fq.num_steps + 1, tokens_count, tokens_count);
            #endif

        }


        #if defined(DEBUG_MODE) && DEBUG_MODE == 1
            std::cout << "Kernel execution completed." << std::endl;
            print_query_output(d_output, debug_output_rows, tokens_count, tokens_count);
        #endif
        

        d_structural_output = d_output + tokens_count * fq.num_steps;
    }
    #if defined(DEBUG_MODE) && DEBUG_MODE == 2
            // Measure the time taken for the kernel execution
            cudaEventCreate(&start);
            cudaEventCreate(&stop);
            cudaEventRecord(start);
    #endif
        // exit(0);

    {
        // _____________________________Inclusive_Scan_____________________________
        // thrust::device_ptr<int8_t> thrust_structural_output(d_structural_output);
        // // Inclusive scan (sum)
        // thrust::inclusive_scan(
        //     thrust::cuda::par,
        //     thrust_structural_output, 
        //     thrust_structural_output + tokens_count, 
        //     thrust_structural_output
        // );
        // exit(0);
    }
    inclusive_scan_inplace_cub(d_structural_output, tokens_count);  // same performance as thrust

    #if defined(DEBUG_MODE) && DEBUG_MODE == 2
        // Measure the time taken for the kernel execution
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        time_ms = 0.0f;
        cudaEventElapsedTime(&time_ms, start, stop);
        
        time_ns = static_cast<uint64_t>(time_ms * 1e6);
        std::cout << "⏱️ inclusive_scan execution time: " << time_ns << " ns" << std::endl;
        std::cout << "⏱️ inclusive_scan execution time: " << time_ns / 1e6 << " ms" << std::endl;

        // Cleanup
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    #endif
    #if defined(DEBUG_MODE) && DEBUG_MODE == 1
        std::cout << "Kernel execution completed." << std::endl;
        print_query_output(d_output, debug_output_rows, tokens_count, tokens_count);
    #endif
    #if defined(DEBUG_MODE) && DEBUG_MODE == 2
        // Measure the time taken for the kernel execution
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);
    #endif

    // _____________________________Output_Generator_____________________________
    uint8_t* d_output_flag = nullptr;
    generate_output_flags(
        d_structural_output,  // pointer to scanned OR row
        tokens_count,         // total tokens
        fq.num_steps,         // target value to match
        &d_output_flag        // output flag pointer
    );


    #if defined(DEBUG_MODE) && DEBUG_MODE == 5
        printGpuMemoryUsage("Generate Output Flags");
    #endif

    #if defined(DEBUG_MODE) && DEBUG_MODE == 2
        // Measure the time taken for the kernel execution
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        time_ms = 0.0f;
        cudaEventElapsedTime(&time_ms, start, stop);
        
        time_ns = static_cast<uint64_t>(time_ms * 1e6);
        std::cout << "⏱️ generate_output_flags execution time: " << time_ns << " ns" << std::endl;
        std::cout << "⏱️ generate_output_flags execution time: " << time_ns / 1e6 << " ms" << std::endl;

        // Cleanup
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    #endif
    #if defined(DEBUG_MODE) && DEBUG_MODE == 1
        print_byte_map("accepted range", d_output_flag, tokens_count);
    #endif
    #if defined(DEBUG_MODE) && DEBUG_MODE == 2
        // Measure the time taken for the kernel execution
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);
    #endif

    // _____________________________Count_Output_Size_____________________________
    // Now we need to count the number of open/close tags
    *matched_tokens = count_ones_cub(d_output_flag, tokens_count);
    // *matched_tokens = thrust::count(
    //     thrust::cuda::par,
    //     thrust::device_pointer_cast(d_output_flag),
    //     thrust::device_pointer_cast(d_output_flag + tokens_count),
    //     1
    // );

    #if defined(DEBUG_MODE) && DEBUG_MODE == 2
        // Measure the time taken for the kernel execution
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        time_ms = 0.0f;
        cudaEventElapsedTime(&time_ms, start, stop);
        
        time_ns = static_cast<uint64_t>(time_ms * 1e6);
        std::cout << "⏱️ thrust::count execution time: " << time_ns << " ns" << std::endl;
        std::cout << "⏱️ thrust::count execution time: " << time_ns / 1e6 << " ms" << std::endl;

        // Cleanup
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    #endif
    #if defined(DEBUG_MODE) && DEBUG_MODE == 1
        cout << "Output flag count: " << *matched_tokens << endl;
    #endif
    #if defined(DEBUG_MODE) && DEBUG_MODE == 2
        // Measure the time taken for the kernel execution
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);
    #endif

    // _____________________________Output_COPYIF_____________________________
    cudaMalloc((void**)d_selected_token_indices, *matched_tokens * sizeof(uint32_t));           // Allocate memory for the selected token indices
    // thrust::copy_if(thrust::cuda::par,                                                          // Copy the selected token indices to the device
    //     d_token_indices, d_token_indices + tokens_count,                                        // input: token indices
    //     thrust::device_pointer_cast(d_output_flag),                                             // stencil: flags
    //     *d_selected_token_indices,                                                              // output
    //     thrust::identity<uint8_t>()                                                             // predicate: flag == 1
    // );
    scatter_cub(d_token_indices, d_output_flag, *d_selected_token_indices, tokens_count);       // a little bit faster than thrust::copy_if


    #if defined(DEBUG_MODE) && DEBUG_MODE == 5
        printGpuMemoryUsage("After Query scatter_cub");
    #endif
    #if defined(DEBUG_MODE) && DEBUG_MODE == 2
        // Measure the time taken for the kernel execution
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        time_ms = 0.0f;
        cudaEventElapsedTime(&time_ms, start, stop);
        
        time_ns = static_cast<uint64_t>(time_ms * 1e6);
        std::cout << "⏱️ copy_if execution time (generate matched array): " << time_ns << " ns" << std::endl;
        std::cout << "⏱️ copy_if execution time (generate matched array): " << time_ns / 1e6 << " ms" << std::endl;

        // Cleanup
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    #endif
    #if defined(DEBUG_MODE) && DEBUG_MODE == 1
        print_uint32_array("accepted range", *d_selected_token_indices, *matched_tokens);
    #endif
     
}