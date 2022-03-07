rocprof --hip-trace -o tmp.csv ./launch $1
cat tmp.stats.csv
