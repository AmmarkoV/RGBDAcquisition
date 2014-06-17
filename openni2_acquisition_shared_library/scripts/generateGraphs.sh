#!/bin/bash

gnuplot -e 'set terminal png; set output "width.png"; set xlabel "Samples"; set ylabel "BBox X dimensions"; plot "bboxx.txt" using 1 with lines title "bbox X"'

gnuplot -e 'set terminal png; set output "height.png"; set xlabel "Samples"; set ylabel "BBox Y dimensions"; plot "bboxy.txt" using 1 with lines title "bbox Y"'

gnuplot -e 'set terminal png; set output "depth.png"; set xlabel "Samples"; set ylabel "BBox Z dimensions"; plot "bboxz.txt" using 1 with lines title "bbox Z"'

exit 0
