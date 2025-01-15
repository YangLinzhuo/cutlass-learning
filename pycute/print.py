"""
Methods for print layout.
"""

import copy

from .layout import Layout, size, cosize, make_layout, rank, is_tuple


def num_digits(x: int) -> int:
    '''
    Count number of digits of an integer
    '''
    cnt = 0
    while x:
        cnt += 1
        x = x // 10
    return cnt


def printf(*args, **kwargs):
    '''
    C++ printf function mimic
    '''
    print(*args, **kwargs, end="")


def print_layout(layout: Layout):
    '''
    Print layout string
    '''
    assert rank(layout) == 2, "Layout must be 2-dimensional"
    idx_width = num_digits(cosize(layout)) + 2
    delim = "+-----------------------"
    printf(layout)
    printf("\n")

    # Column indices
    printf("    ")
    for n in range(size(layout.shape[1])):
        printf(f"  {n:{idx_width - 2}d} ")
    printf("\n")

    for m in range(size(layout.shape[0])):
        # Header
        printf("    ")
        for n in range(size(layout.shape[1])):
            printf(f"{delim:.{idx_width + 1}s}")
        printf("+\n")
        # Values
        printf(f"{m: 2d}  ")  # Row indices
        for n in range(size(layout.shape[1])):
            printf(f"| {int(layout(m, n)):{idx_width - 2}d} ")
        printf("|\n")

    # Footer
    printf("    ")
    for n in range(size(layout.shape[1])):
        printf(f"{delim:.{idx_width + 1}s}")
    printf("+\n")


class TikzColorBlackWhitex8:
    '''
    Latex Tikz package color mapping of BW (black and white)
    '''
    color_map = [
        "black!00",
        "black!40",
        "black!20",
        "black!60",
        "black!10",
        "black!50",
        "black!30",
        "black!70",
    ]

    def __call__(self, idx):
        return self.color_map[idx % len(self.color_map)]


class TikzColorThreadVal:
    '''
    Latex Tikz package color mapping of TV (thread and value)
    '''
    color_map = [
        "{rgb,255:red,175;green,175;blue,255}",
        "{rgb,255:red,175;green,255;blue,175}",
        "{rgb,255:red,255;green,255;blue,175}",
        "{rgb,255:red,255;green,175;blue,175}",
        "{rgb,255:red,210;green,210;blue,255}",
        "{rgb,255:red,210;green,255;blue,210}",
        "{rgb,255:red,255;green,255;blue,210}",
        "{rgb,255:red,255;green,210;blue,210}",
    ]

    def __call__(self, tid, vid):
        return self.color_map[tid % len(self.color_map)]


def print_latex(
    layout, color=None  # (m,n) -> idx
):  # lambda(idx) -> tikz color string
    '''
    Print layout latex for visualization
    '''
    if color is None:
        color = TikzColorBlackWhitex8()
    assert rank(layout) == 2, "Layout must be 2-dimensional"
    layout = make_layout(layout, Layout(1, 0))

    # Commented print(layout)
    printf("%% Layout: ")
    printf(layout)
    printf("\n")
    # Header
    printf(
        "\\documentclass[convert]{standalone}\n"
        "\\usepackage{tikz}\n\n"
        "\\begin{document}\n"
        "\\begin{tikzpicture}[x={(0cm,-1cm)},y={(1cm,0cm)},"
        "every node/.style={minimum size=1cm, outer sep=0pt}]\n\n"
    )

    # Layout
    for i in range(size(layout.shape[0])):
        for j in range(size(layout.shape[1])):
            idx = layout(i, j)
            printf("\\node[fill=%s] at (%d,%d) {%d};\n" % (color(idx), i, j, idx))
    # Grid
    printf(
        "\\draw[color=black,thick,shift={(-0.5,-0.5)}] (0,0) grid (%d,%d);\n\n"
        % (size(layout.shape[0]), size(layout.shape[1]))
    )
    # Labels
    j = -1
    for i in range(size(layout.shape[0])):
        printf("\\node at (%d,%d) {\\Large{\\texttt{%d}}};\n" % (i, j, i))

    i = -1
    for j in range(size(layout.shape[1])):
        printf("\\node at (%d,%d) {\\Large{\\texttt{%d}}};\n" % (i, j, j))

    # Footer
    printf("\\end{tikzpicture}\n\\end{document}\n")


def print_latex_tv(layout: Layout, thr: Layout, color=None):
    '''
    Print layout latex for color visualization
    '''
    assert rank(layout) == 2, "Layout must be 2-dimensional"
    if color is None:
        color = TikzColorThreadVal()

    # Commented prints
    printf("%% Layout: ")
    printf(layout)
    printf("\n")
    printf("%% ThrID : ")
    printf(thr)
    printf("\n")
    # Header
    printf(
        "\\documentclass[convert]{standalone}\n"
        "\\usepackage{tikz}\n\n"
        "\\begin{document}\n"
        "\\begin{tikzpicture}[x={(0cm,-1cm)},y={(1cm,0cm)},"
        "every node/.style={minimum size=1cm, outer sep=0pt}]\n\n"
    )

    # Layout
    for i in range(size(layout.shape[0])):
        for j in range(size(layout.shape[1])):
            thrid = layout(i, j) % size(thr)
            val_idx = layout(i, j) / size(thr)
            thr_idx = thr(thrid)

            printf(
                "\\node[fill=%s] at (%d,%d) {\\shortstack{T%d \\\\ V%d}};\n"
                % (color(thr_idx, val_idx), i, j, thr_idx, val_idx)
            )

    # Grid
    printf(
        "\\draw[color=black,thick,shift={(-0.5,-0.5)}] (0,0) grid (%d,%d);\n\n"
        % (size(layout.shape[0]), size(layout.shape[1]))
    )
    # Labels
    j = -1
    for i in range(size(layout.shape[0])):
        printf("\\node at (%d,%d) {\\Large{\\texttt{%d}}};\n" % (i, j, i))
    i = -1
    for j in range(size(layout.shape[1])):
        printf("\\node at (%d,%d) {\\Large{\\texttt{%d}}};\n" % (i, j, j))

    # Footer
    printf("\\end{tikzpicture}\n\\end{document}\n")


def print_hier_layout(layout: Layout):
    '''
    Print layout latex for color visualization
    '''
    assert rank(layout) == 2, "Layout must be 2-dimensional"

    # Commented prints
    printf("%% Layout: ")
    printf(layout)
    printf("\n")
    # Header
    printf(
        "\\documentclass[convert]{standalone}\n"
        "\\usepackage{tikz}\n"
        "\\usepackage{amsmath}\n"
        "\\begin{document}\n"
        "\\begin{tikzpicture}[x={(0cm,-1cm)},y={(1cm,0cm)},"
        "every node/.style={minimum size=1cm, outer sep=0pt}]\n\n"
    )

    line_nodes = []

    # Layout
    for i in range(size(layout.shape[0])):
        for j in range(size(layout.shape[1])):
            idx = layout(i, j)

            printf(
                "\\node[fill=white] at (%d,%d) {\\shortstack{%d}};\n"
                % (i, j, idx)
            )

            line_nodes.append((idx, (i, j)))

    # Grid
    printf(
        "\\draw[color=gray,very thin,shift={(-0.5,-0.5)}] (0,0) grid (%d,%d);\n\n"
        % (size(layout.shape[0]), size(layout.shape[1]))
    )

    printf(
        "\\draw[color=black,line width=1mm,shift={(-0.5,-0.5)}] "
        "(%d, %d) rectangle (%d, %d);\n"
        % (0, 0,
           size(layout.shape[0]), size(layout.shape[1]))
    )

    line_str = "\\draw[color=gray,line width=1mm,opacity=0.3] "
    line_str += f"{line_nodes[0][1]}"
    line_nodes.sort(key=lambda x: x[0])
    for node in line_nodes[1:]:
        line_str += f" -- {node[1]}"
    line_str += ";\n"
    printf(line_str)

    colors = ["olive", "violet", "teal", "cyan"]
    # stride
    if is_tuple(layout.shape[0]) or is_tuple(layout.shape[1]):
        # make a slide window
        window_size = (
            1 if not is_tuple(layout.shape[0]) else layout.shape[0][0],
            1 if not is_tuple(layout.shape[1]) else layout.shape[1][0],
        )
        for i in range(size(layout.shape[0]) // window_size[0]):
            for j in range(size(layout.shape[1]) // window_size[1]):
                # draw inner grid
                start_point = (i * window_size[0], j * window_size[1])
                printf(
                    "\\draw[color=gray,line width=0.5mm,shift={(-0.5,-0.5)}] "
                    "(%d, %d) rectangle (%d, %d);\n"
                    % (start_point[0], start_point[1],
                    start_point[0] + window_size[0], start_point[1] + window_size[1])
                )

        if is_tuple(layout.stride[0]):  # len(stride[0]) == 2
            printf("\\draw[color=%s, line width=0.5mm,shift={(0,-0.5)},-latex] "
                   "(0, 0) to[out=-135,in=135] (%d, 0);\n" % (colors[0], window_size[0] - 1))
            printf("\\node[text=%s] at ([shift={(0.1,-0.75)}]%d, 0) {\\texttt{+%d}};\n"
                   % (colors[0], window_size[0] - 1, layout.stride[0][0]))
            printf("\\draw[color=%s, line width=0.5mm,shift={(0,-0.5)},-latex] "
                   "(0, 0) to[out=-135,in=135] (%d, 0);\n" % (colors[1], window_size[0]))
            printf("\\node[text=%s] at ([shift={(0.1,-0.75)}]%d, 0) {\\texttt{+%d}};\n"
                   % (colors[1], window_size[0], layout.stride[0][1]))
        else:
            printf("\\draw[color=%s, line width=0.5mm,shift={(0,-0.5)},-latex] "
                   "(0, 0) to[out=-135,in=135] (%d, 0);\n" % (colors[0], window_size[0]))
            printf("\\node[text=%s] at ([shift={(0.1,-0.75)}]%d, 0) {\\texttt{+%d}};\n"
                   % (colors[0], window_size[0], layout.stride[0]))

        n_row = size(layout.shape[0]) - 1
        if is_tuple(layout.stride[1]):  # len(stride[1]) == 2
            printf("\\draw[color=%s, line width=0.5mm,shift={(0.5,0)},-latex] "
                   "(%d, 0) to[out=-45,in=-135] (%d, %d);\n"
                   % (colors[2], n_row, n_row, window_size[1] - 1))
            printf("\\node[text=%s] at ([shift={(0.75,0.1)}]%d, %d) {\\texttt{+%d}};\n"
                   % (colors[2], n_row, window_size[1] - 1, layout.stride[1][0]))
            printf("\\draw[color=%s, line width=0.5mm,shift={(0.5,0)},-latex] "
                   "(%d, 0) to[out=-45,in=-135] (%d, %d);\n"
                   % (colors[3], n_row, n_row, window_size[1]))
            printf("\\node[text=%s] at ([shift={(0.75,0.1)}]%d, %d) {\\texttt{+%d}};\n"
                   % (colors[3], n_row, window_size[1], layout.stride[1][1]))
        else:
            printf("\\draw[color=%s, line width=0.5mm,shift={(0.5,0)},-latex] "
                   "(%d, 0) to[out=-45,in=-135] (0, %d);\n"
                   % (colors[2], n_row, window_size[1]))
            printf("\\node[text=%s] at ([shift={(0.75,0.1)}]%d, %d) {\\texttt{+%d}};\n"
                   % (colors[2], n_row, window_size[1], layout.stride[1]))

        # Show colorized layout
        shape_str=  f"{layout.shape[0]}, {layout.shape[1]}"
        stride_str = ""
        if is_tuple(layout.stride[0]):
            stride_str += (f"(\\textcolor{{{colors[0]}}}{{{layout.stride[0][0]}}},"
                        f"\\textcolor{{{colors[1]}}}{{{layout.stride[0][1]}}})")
        else:
            stride_str += f"\\textcolor{{{colors[0]}}}{{{layout.stride[0]}}}"
        stride_str += ","
        if is_tuple(layout.stride[1]):
            stride_str += (f"(\\textcolor{{{colors[2]}}}{{{layout.stride[1][0]}}},"
                        f"\\textcolor{{{colors[3]}}}{{{layout.stride[1][1]}}})")
        else:
            stride_str += f"\\textcolor{{{colors[2]}}}{{{layout.stride[1]}}}"

        printf("\\node [draw=white] at (-2, 0.5) {\n"
        "\\begin{minipage}{0.50\\textwidth}\n"
        "$$\n"
        "    \\begin{bmatrix}\n"
        "    %s\\\\\n"
        "    %s\n"
        "    \\end{bmatrix}\n"
        "$$\n"
        "\\end{minipage}\n"
        "};\n" % (shape_str, stride_str))

    # Labels
    j = -1.2
    for i in range(size(layout.shape[0])):
        printf("\\node at (%d,%.2f) {\\Large{\\texttt{%d}}};\n" % (i, j, i))
    i = -1.0
    for j in range(size(layout.shape[1])):
        printf("\\node at (%.2f,%d) {\\Large{\\texttt{%d}}};\n" % (i, j, j))

    # Footer
    printf("\\end{tikzpicture}\n\\end{document}\n")
