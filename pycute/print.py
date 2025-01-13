"""
Methods for print layout.
"""

from .layout import Layout, size, cosize, make_layout


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
    assert len(layout.shape) == 2, "Layout must be 2-dimensional"
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
    assert len(layout.shape) == 2, "Layout must be 2-dimensional"
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
    assert len(layout.shape) == 2, "Layout must be 2-dimensional"
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
