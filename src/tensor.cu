#include <cuda_runtime.h>
#include <vector>
#include <numeric>

class Tensor
{
public:
    float *d_ptr;
    float rank;
    std::vector<int> shape;
    size_t total_elements;

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