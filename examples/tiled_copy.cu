#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <cute/tensor.hpp>

template <class TensorS, class TensorD, class ThreadLayout, class Shape>
__global__ void copy_kernel_general(TensorS S, TensorD D, ThreadLayout, Shape real_shape)
{
    using namespace cute;

    // Slice the tiled tensors
    Tensor tile_S = S(make_coord(_, _), blockIdx.x, blockIdx.y); // (BlockShape_M, BlockShape_N)
    Tensor tile_D = D(make_coord(_, _), blockIdx.x, blockIdx.y); // (BlockShape_M, BlockShape_N)

    // Construct a partitioning of the tile among threads with the given thread arrangement.
    Tensor thr_tile_S = local_partition(tile_S, ThreadLayout{}, threadIdx.x);
    Tensor thr_tile_D = local_partition(tile_D, ThreadLayout{}, threadIdx.x);
    Tensor fragment = make_tensor_like(thr_tile_S);

    auto block_shape = shape(tile_S);
    Tensor identity = make_identity_tensor(block_shape);
    Tensor thr_identity = local_partition(identity, ThreadLayout{}, threadIdx.x);
    Tensor predicator = make_tensor<bool>(shape(thr_identity));
    auto const M = get<0>(real_shape);
    auto const N = get<1>(real_shape);
    for (int i = 0; i < size(thr_identity); ++i)
    {
        auto const m = get<0>(thr_identity(i));
        auto const n = get<1>(thr_identity(i));
        predicator(i) = (blockIdx.x * get<0>(block_shape) + m) < M && (blockIdx.y * get<1>(block_shape) + n) < N;
    }

    copy_if(predicator, thr_tile_S, fragment);
    copy_if(predicator, fragment, thr_tile_D);
}

int main(int argc, char **argv)
{
    //
    // Given a 2D shape, perform an efficient copy
    //

    using namespace cute;
    using Element = int;

    // Define a tensor shape with dynamic extents (m, n)
    auto const M = 257;
    auto const N = 513;
    auto tensor_shape = make_shape(M, N);

    //
    // Allocate and initialize
    //
    thrust::host_vector<Element> h_S(size(tensor_shape));
    thrust::host_vector<Element> h_D(size(tensor_shape));

    for (int i = 0; i < h_S.size(); ++i)
    {
        h_S[i] = static_cast<Element>(i);
        h_D[i] = Element{};
    }

    thrust::device_vector<Element> d_S = h_S;
    thrust::device_vector<Element> d_D = h_D;

    //
    // Make tensors
    //
    Tensor tensor_S = make_tensor(make_gmem_ptr(thrust::raw_pointer_cast(d_S.data())), make_layout(tensor_shape));
    Tensor tensor_D = make_tensor(make_gmem_ptr(thrust::raw_pointer_cast(d_D.data())), make_layout(tensor_shape));

    //
    // Tile tensors
    //

    // Define a statically sized block (M, N).
    // Note, by convention, capital letters are used to represent static modes.
    auto block_shape = make_shape(Int<128>{}, Int<64>{});

    // Tile the tensor (m, n) ==> ((M, N), m', n') where (M, N) is the static tile
    // shape, and modes (m', n') correspond to the number of tiles.
    //
    // These will be used to determine the CUDA kernel grid dimensions.
    Tensor tiled_tensor_S = tiled_divide(tensor_S, block_shape); // ((M, N), m', n')
    Tensor tiled_tensor_D = tiled_divide(tensor_D, block_shape); // ((M, N), m', n')

    // Construct a TiledCopy with a specific access pattern.
    // This version uses a Layout-of-Threads to describe the number and arrangement of threads (e.g. row-major, col-major, etc)

    // Thread arrangement
    Layout thr_layout = make_layout(make_shape(Int<32>{}, Int<8>{})); // (32, 8) -> thr_idx

    //
    // Determine grid and block dimensions
    //
    dim3 gridDim(size<1>(tiled_tensor_D), size<2>(tiled_tensor_D)); // Grid shape corresponds to modes m' and n'
    dim3 blockDim(size(thr_layout));

    // Launch the kernel
    copy_kernel_general<<<gridDim, blockDim>>>(
        tiled_tensor_S,
        tiled_tensor_D,
        thr_layout,
        shape(tensor_S));

    cudaError result = cudaDeviceSynchronize();
    if (result != cudaSuccess)
    {
        std::cerr << "CUDA Runtime error: " << cudaGetErrorString(result) << std::endl;
        return -1;
    }

    // Verify
    h_D = d_D;
    int32_t errors = 0;
    int32_t const kErrorLimit = 10;
    for (size_t i = 0; i < h_D.size(); ++i)
    {
        if (h_S[i] != h_D[i])
        {
            std::cerr << "Error. S[" << i << "]: " << h_S[i] << ",   D[" << i << "]: " << h_D[i] << std::endl;

            if (++errors >= kErrorLimit)
            {
                std::cerr << "Aborting on " << kErrorLimit << "nth error." << std::endl;
                return -1;
            }
        }
    }

    std::cout << "Success." << std::endl;

    return 0;
}