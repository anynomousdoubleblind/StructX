#include "utils.cu"
#include "loadfile.h"
#include "loadfile.cu"
#include "standard_parse.h"
#include "standard_parse.cu"
#include "stages/validation.h"
#include "stages/validation.cu"
#include "stages/tokenization.h"
#include "stages/tokenization.cu"
#include "stages/parser.h"
#include "stages/parser.cu"
#include "stages/query.h"
#include "stages/query.cu"
#include "stages/query_host.h"
#include "stages/query_host.cpp" 

int main(int argc, char **argv) {
    // Default file path and query
    std::string filePath = "./dataset/psd7003.xml";                                            // Default XML file path            
    // std::string xpath_query = "/ProteinDatabase/ProteinEntry/reference/refinfo/authors/author"; // Default XPath query


    // string xpath_query = "/ProteinDatabase/ProteinEntry/protein/name"; // Example XPath query --> psd7003
    // string xpath_query = "/ProteinDatabase/ProteinEntry/organism/formal"; // Example XPath query --> psd7003
    // string xpath_query = "/ProteinDatabase/ProteinEntry/reference/refinfo/authors/author"; // Example XPath query --> psd7003
    // string xpath_query = "/ProteinDatabase/ProteinEntry/reference/refinfo[@refid='A31764']/authors/author"; // Example XPath query --> psd7003
    // string xpath_query = "/ProteinDatabase/ProteinEntry/reference/refinfo[@refid='A31764']"; // Example XPath query --> psd7003
    // string xpath_query = "/ProteinDatabase/ProteinEntry/reference/refinfo"; // Example XPath query --> psd7003
    // string xpath_query = "/ProteinDatabase/ProteinEntry/reference/refinfo/authors/author[0]"; // Example XPath query --> psd7003
    // string xpath_query = "/ProteinDatabase/ProteinEntry/reference/refinfo[volume=85]"; // Example XPath query --> psd7003
    // string xpath_query = "/ProteinDatabase/ProteinEntry/reference/refinfo[volume=85]/authors/author[0]"; // Example XPath query --> psd7003
    // string xpath_query = "/ProteinDatabase/ProteinEntry/header[uid=CCHU]"; // Example XPath query --> psd7003
    string xpath_query = "/ProteinDatabase/ProteinEntry/reference/refinfo[@refid='A31764']/authors/author[0]"; // Example XPath query --> psd7003


    // std::string filePath = "./dataset/lucid_example.xml";                                            // Default XML file path            
    // std::string xpath_query = "/root/loc/state"; // Default XPath query


    // Check command-line arguments
    if (argc >= 2) {
        filePath = argv[1];  // XML file path
        cout << "\033[1;36m[INFO]\033[0m Using custom XML file from command line: " << filePath << "\n";
        if (argc >= 3) {
            xpath_query = argv[2];  // XPath query string
            cout << "\033[1;36m[INFO]\033[0m Using custom XPath query from command line: " << xpath_query << "\n";
        }
        std::cout << "\033[1;36m[INFO]\033[0m Using custom XML file and/or XPath query from command line.\n";
    } else {
        std::cout << "\033[1;36m[INFO]\033[0m Using default XML file and XPath query.\n";
    }

    // Load File
    size_t fileSize = 0;
    uint8_t* xmlContent = loadXMLFile(filePath, fileSize);

    // Parse
    if (xmlContent && fileSize > 0) {
        std::cout << "\033[1;33m XML file loaded successfully! \033[0m\n";
        std::cout << "\033[1;33m Starting parsing with XPath query: " << xpath_query << " \033[0m\n";
        std::cout << "\033[1;33m File size: " << fileSize << " bytes \033[0m\n";
        std::cout << "\033[1;33m Parsing XML file... \033[0m\n";
        standard_parse(xmlContent, fileSize, xpath_query);
        cudaFreeHost(xmlContent);
    } else {
        std::cerr << "\033[1;31m Failed to load XML file. \033[0m\n";
    }

    cudaDeviceReset();
    return 0;
}

