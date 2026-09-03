# Overleaf nonzero-IC experiment bundle

Upload the contents of this directory to one Overleaf project directory.

- `nonzero_ic_standalone_report.tex` is a self-contained Elsevier-style report.
- `nonzero_ic_experimental_section.tex` is an insert for the existing paper's
  Experimental Validation section.
- `nonzero_ic_repeat_2.pdf` is the figure referenced by both TeX files.
- `nonzero_ic_repeat_2.svg` is the editable vector source.
- The two `rep01`/`rep02` CSV and JSON pairs are the underlying evidence.

To merge the section into the existing paper, place the section and PDF in the
same Overleaf directory and add:

```tex
\input{nonzero_ic_experimental_section}
```

The figure uses `\includegraphics` with the PDF, so the main document only
needs `\usepackage{graphicx}`. The SVG is supplied for editing and does not
require Overleaf shell escape.

