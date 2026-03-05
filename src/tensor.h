#include <cuda_runtime.h>
#include <vector>
#include <numeric>

class Tensor
{
public:
    float *d_ptr;           // Pointer to the memory ON the GPU
    std::vector<int> shape; // e.g., {1, 784}
    size_t total_elements;  // Total count (1 * 784 = 784)

    Tensor(std::vector<int> s) : shape(s)
    {
        total_elements = 1;
        for (int dim : shape)
            total_elements *= dim;

        cudaMalloc(&d_ptr, total_elements * sizeof(float));
    }

    ~Tensor()
    {
        cudaFree(d_ptr);
    }

    Tensor(const Tensor &) = delete;
    Tensor &operator=(const Tensor &) = delete;
};