#include <variant>
#include <cute/tensor.hpp>
#include <cutlass/util/command_line.h>

template<class SubShape, class SubCoord>
CUTE_HOST_DEVICE
int transform_coord(const SubShape& s, const SubCoord& c)
{
    using namespace cute;
    if constexpr (is_tuple<SubCoord>::value) {
        // c rank 2, c is subcoord of row or col
        // c is (col1, col2) or (row1, row2)
        // s is (m, n)
        // row = c[1] * s[0] + c[0];
        // col = c[1] * s[0] + c[0]
        CUTE_STATIC_ASSERT_V(rank(c) == Int<2>{});
        return get<1>(c) * get<0>(s) + get<0>(c);
    } else {
        return c;
    }

    CUTE_GCC_UNREACHABLE;
}

// Generic Tensor and Coord to LaTeX TikZ
template <class Layout, class Coord>
CUTE_HOST_DEVICE void
print_coord_latex(Layout const &layout, // (m,n) -> (tid,vid)
                  Coord const &coord)   // tid -> thr_idx
{
    using namespace cute;

    CUTE_STATIC_ASSERT_V(rank(layout) == Int<2>{});
    // auto identity = make_identity_layout
    auto identity = make_identity_tensor(shape(layout));
    auto id_coord = identity(coord);

    // Commented prints
    printf("%% Layout: ");
    print(layout);
    printf("\n");
    printf("%% Coord : ");
    print(coord);
    printf("\n");
    // Header
    printf("\\documentclass[convert]{standalone}\n"
           "\\usepackage{tikz}\n\n"
           "\\begin{document}\n"
           "\\begin{tikzpicture}[x={(0cm,-1cm)},y={(1cm,0cm)},every node/.style={minimum size=1cm, outer sep=0pt}]\n\n");

    // Layout
    for (int i = 0; i < size<0>(layout); ++i)
    {
        for (int j = 0; j < size<1>(layout); ++j)
        {
            printf("\\node[fill=%s] at (%d,%d) {\\shortstack{%d}};\n",
                   "white",
                   i, j,
                   layout(i, j));
        }
    }

    auto new_shape = make_shape(size<0>(layout), size<1>(layout));
    printf("%% new Shape: ");
    print(new_shape);
    printf("\n");
    for (int i = 0; i < size(id_coord); ++i)
    {
        auto coord = id_coord(i);
        auto idx = layout(id_coord(i));
        int new_row = 0, new_col = 0;
        printf("%% coord: ");
        print(id_coord(i));
        print("\n");
        new_row = transform_coord(shape<0>(layout), get<0>(coord));
        new_col = transform_coord(shape<1>(layout), get<1>(coord));

        printf("%% new coord: (%d, %d)\n", new_row, new_col);
        printf("\\node[fill=%s] at (%d,%d) {\\shortstack{%d}};\n",
               "{rgb,255:red,175;green,255;blue,175}",
               new_row,
               new_col,
               idx);
    }

    // Grid
    printf("\\draw[color=black,thick,shift={(-0.5,-0.5)}] (0,0) grid (%d,%d);\n\n",
           int(size<0>(layout)), int(size<1>(layout)));
    // Labels
    for (int i = 0, j = -1; i < size<0>(layout); ++i)
    {
        printf("\\node at (%d,%d) {\\Large{\\texttt{%d}}};\n", i, j, i);
    }
    for (int j = 0, i = -1; j < size<1>(layout); ++j)
    {
        printf("\\node at (%d,%d) {\\Large{\\texttt{%d}}};\n", i, j, j);
    }

    // Footer
    printf("\\end{tikzpicture}\n"
           "\\end{document}\n");
}


int str2num(std::string s) {
    if (s == "_") {
        return cute::_;
    } else {
        return std::stoi(s);
    }
}


using std::pair;
using std::string;

cute::tuple<int, int> pair2tuple(const std::pair<std::string, std::string>& str_pair) {
    return cute::tuple{str2num(str_pair.first), str2num(str_pair.second)};
}

template<class T>
cute::tuple<T> vec2tuple(const std::vector<pair<string, string>>& vec) {
    if (vec.size() == 1) {
        return pair2tuple(vec[0]);
    } else {
        return cute::tuple{pair2tuple(vec[0]), pair2tuple(vec[1])};
    }
}


int main(int argc, char const **argv)
{
    // Parse command line
    // std::vector<int> tensor_shape;
    // std::vector<int> tensor_stride;
    // std::vector<std::pair<std::string, std::string>> tensor_shape_str;
    // std::vector<std::pair<std::string, std::string>> tensor_stride_str;
    // std::vector<std::pair<std::string, std::string>> tensor_coord_str;
    // std::string tensor_shape;
    // cutlass::CommandLine cmd_line(argc, argv);

    // // cmd_line.get_cmd_line_arguments("shape", tensor_shape);
    // cmd_line.get_cmd_line_argument_pairs("shape", tensor_shape_str, ':', ',');
    // cmd_line.get_cmd_line_argument_pairs("stride", tensor_stride_str, ':', ',');
    // cmd_line.get_cmd_line_argument_pairs("coord", tensor_coord_str, ':', ',');

    using namespace cute;

    auto layout = make_layout(
        make_shape(make_shape(2, 4), make_shape(3, 5)),
        make_stride(make_stride(3, 6), make_stride(1, 24))
    );
    // auto coord = make_coord(
    //     make_coord(1, _),
    //     make_coord(1, _)
    // );
    auto coord = make_coord(
        make_coord(0, _),
        make_coord(1, 2)
    );
    // auto layout = make_layout(make_shape(2, 8));
    // auto coord = make_coord(_, 4);
    print_coord_latex(layout, coord);
    return 0;
}