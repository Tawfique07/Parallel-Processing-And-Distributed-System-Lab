#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <cuda_runtime.h>

struct Contact {
    char name[256];
    char phoneNumber[256];
};

__device__ bool deviceStrStr(const char* haystack, const char* needle, int needleLength) {
    for (int i = 0; i < needleLength; ++i) {
        if (haystack[i] != needle[i]) {
            return false;
        }
    }
    return true;
}

__global__ void searchContactsKernel(Contact* contacts, int size, const char* searchName, int searchNameLength) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < size) {
        if (deviceStrStr(contacts[tid].name, searchName, searchNameLength)) {
            printf("Name: %s, Phone: %s\n", contacts[tid].name, contacts[tid].phoneNumber);
        }
    }
}

void printPhonebook(const std::vector<Contact>& phonebook) {
    std::cout << "Phonebook:\n";
    for (const auto& contact : phonebook) {
        std::cout << "Name: " << contact.name << ", Phone: " << contact.phoneNumber << "\n";
    }
    std::cout << "\n";
}

void searchContactsCUDA(const Contact* h_phonebook, int size, const char* searchName) {
    size_t contactsSize = size * sizeof(Contact);

    // Allocate device memory for contacts
    Contact* d_contacts;
    cudaMalloc((void**)&d_contacts, contactsSize);

    // Copy contacts data from host to device
    cudaMemcpy(d_contacts, h_phonebook, contactsSize, cudaMemcpyHostToDevice);

    // Allocate device memory for searchName
    int searchNameLength = strlen(searchName) + 1;
    char* d_searchName;
    cudaMalloc((void**)&d_searchName, searchNameLength);

    // Copy searchName to device
    cudaMemcpy(d_searchName, searchName, searchNameLength, cudaMemcpyHostToDevice);

    // Set up grid and block dimensions
    int blockSize = 256;
    int gridSize = (size + blockSize - 1) / blockSize;

    // Launch the search kernel
    searchContactsKernel<<<gridSize, blockSize>>>(d_contacts, size, d_searchName, searchNameLength);

    // Synchronize to ensure kernel execution is complete
    cudaDeviceSynchronize();

    // Free device memory
    cudaFree(d_contacts);
    cudaFree(d_searchName);
}

std::vector<Contact> readPhonebook(const std::string& filename) {
    std::vector<Contact> phonebook;
    std::ifstream file(filename);

    if (!file.is_open()) {
        std::cerr << "Error opening file: " << filename << std::endl;
        return phonebook;
    }

    Contact contact;
    while (file >> contact.name >> contact.phoneNumber) {
        phonebook.push_back(contact);
    }

    file.close();
    return phonebook;
}

int main() {
    const std::string phonebookFilename = "phonebook.txt";
    const char* searchName = "Charlie";  // Change this to the desired search name

    // Read phonebook from file
    std::vector<Contact> phonebook = readPhonebook(phonebookFilename);

    if (!phonebook.empty()) {
        // Print the entire phonebook
        printPhonebook(phonebook);

        // Search contacts using CUDA
        searchContactsCUDA(phonebook.data(), phonebook.size(), searchName);
    }

    return 0;
}