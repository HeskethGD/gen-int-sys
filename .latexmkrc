# Allow the manuscript to be built from the repository root.
do './latex/.latexmkrc' or die "Cannot load latex/.latexmkrc: $@ $!";
