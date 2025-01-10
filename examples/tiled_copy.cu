#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <cute/tensor.hpp>

template <class TensorS, class TensorD, class ThreadLayout, class Shape>
__global__ void copy_kernel_general(TensorS S, TensorD D, ThreadLayout, Shape real_shape)
{
    using namespace cute;

    // Slice the tiled tensors
    Tensor tile_S = S(make_coord(_, _), blockIdx.y, blockIdx.x); // (BlockShape_M, BlockShape_N)
    Tensor tile_D = D(make_coord(_, _), blockIdx.y, blockIdx.x); // (BlockShape_M, BlockShape_N)

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
        predicator(i) = (blockIdx.y * get<0>(block_shape) + m) < M && (blockIdx.x * get<1>(block_shape) + n) < N;
    }

    copy_if(predicator, thr_tile_S, fragment);
    copy_if(predicator, fragment, thr_tile_D);
}

/// Vectorized copy kernel.
///
/// Uses `make_tiled_copy()` to perform a copy using vector instructions. This operation
/// has the precondition that pointers are aligned to the vector size.
///
template <class TensorS, class TensorD, class Tiled_Copy>
__global__ void copy_kernel_vectorized(TensorS S, TensorD D, Tiled_Copy tiled_copy)
{
    using namespace cute;

    // Slice the tensors to obtain a view into each tile.
    Tensor tile_S = S(make_coord(_, _), blockIdx.y, blockIdx.x); // (BlockShape_M, BlockShape_N)
    Tensor tile_D = D(make_coord(_, _), blockIdx.y, blockIdx.x); // (BlockShape_M, BlockShape_N)

    // Construct a Tensor corresponding to each thread's slice.
    ThrCopy thr_copy = tiled_copy.get_thread_slice(threadIdx.x);

    Tensor thr_tile_S = thr_copy.partition_S(tile_S); // (CopyOp, CopyM, CopyN)
    Tensor thr_tile_D = thr_copy.partition_D(tile_D); // (CopyOp, CopyM, CopyN)

    // Construct a register-backed Tensor with the same shape as each thread's partition
    // Use make_fragment because the first mode is the instruction-local mode
    Tensor fragment = make_fragment_like(thr_tile_D); // (CopyOp, CopyM, CopyN)

    // Copy from GMEM to RMEM and from RMEM to GMEM
    copy(tiled_copy, thr_tile_S, fragment);
    copy(tiled_copy, fragment, thr_tile_D);
}

template <typename T>
cudaError_t launch_matrix_copy(T *const src, T *dst, const int M, const int N)
{
    using namespace cute;
    auto tensor_shape = make_shape(M, N);
    //
    // Make tensors
    //
    Tensor tensor_src = make_tensor(make_gmem_ptr(src), make_layout(tensor_shape));
    Tensor tensor_dst = make_tensor(make_gmem_ptr(dst), make_layout(tensor_shape));

    // Define a statically sized block (M, N).
    // Note, by convention, capital letters are used to represent static modes.
    auto block_shape = make_shape(Int<128>{}, Int<64>{});

    // Tile the tensor (m, n) ==> ((M, N), m', n') where (M, N) is the static tile
    // shape, and modes (m', n') correspond to the number of tiles.
    //
    // These will be used to determine the CUDA kernel grid dimensions.
    Tensor tiled_tensor_src = tiled_divide(tensor_src, block_shape); // ((M, N), m', n')
    Tensor tiled_tensor_dst = tiled_divide(tensor_dst, block_shape); // ((M, N), m', n')

    // Thread arrangement
    Layout thr_layout = make_layout(make_shape(Int<32>{}, Int<8>{})); // (32, 8) -> thr_idx

    //
    // Determine grid and block dimensions
    //
    // Grid shape corresponds to modes m' and n'
    dim3 gridDim(size<2>(tiled_tensor_dst), size<1>(tiled_tensor_dst));
    dim3 blockDim(size(thr_layout));

    // Equivalent check to the above
    if (not evenly_divides(tensor_shape, block_shape))
    {
        // Launch the kernel of general version
        copy_kernel_general<<<gridDim, blockDim>>>(
            tiled_tensor_src,
            tiled_tensor_dst,
            thr_layout,
            shape(tensor_src));
    }
    else
    {
        // Value arrangement per thread
        Layout val_layout = make_layout(make_shape(Int<4>{}, Int<1>{})); // (4,1) -> val_idx
        // Define `AccessType` which controls the size of the actual memory access instruction.
        using CopyOp = UniversalCopy<uint_byte_t<sizeof(T) * size(val_layout)>>; // A very specific access width copy instruction
        // using CopyOp = UniversalCopy<cutlass::AlignedArray<Element, size(val_layout)>>;  // A more generic type that supports many copy strategies
        // using CopyOp = AutoVectorizingCopy;                                              // An adaptable-width instruction that assumes maximal alignment of inputs

        // A Copy_Atom corresponds to one CopyOperation applied to Tensors of type Element.
        using Atom = Copy_Atom<CopyOp, T>;
        // Construct tiled copy, a tiling of copy atoms.
        //
        // Note, this assumes the vector and thread layouts are aligned with contigous data
        // in GMEM. Alternative thread layouts are possible but may result in uncoalesced
        // reads. Alternative value layouts are also possible, though incompatible layouts
        // will result in compile time errors.
        TiledCopy tiled_copy = make_tiled_copy(Atom{},      // Access strategy
                                               thr_layout,  // thread layout (e.g. 32x4 Col-Major)
                                               val_layout); // value layout (e.g. 4x1)
        copy_kernel_vectorized<<<gridDim, blockDim>>>(
            tiled_tensor_src,
            tiled_tensor_dst,
            tiled_copy);
    }

    cudaDeviceSynchronize();
    return cudaGetLastError();
}

int main(int argc, char **argv)
{
    //
    // Given a 2D shape, perform an efficient copy
    //

    using namespace cute;
    using Element = int;

    // Define a tensor shape with dynamic extents (m, n)
    auto const M = 256;
    auto const N = 512;
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

    // Launch the kernel
    auto result = launch_matrix_copy(
        thrust::raw_pointer_cast(d_S.data()),
        thrust::raw_pointer_cast(d_D.data()),
        M,
        N);

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