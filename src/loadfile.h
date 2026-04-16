// loadfile.h
#ifndef LOADFILE_H
#define LOADFILE_H

#include <string>

// Function to load XML file content into a string
uint8_t* loadXMLFile(const std::string& filePath, size_t &fileSize);


#endif // LOADFILE_H