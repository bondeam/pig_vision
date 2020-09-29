#!/bin/bash

outfile="video_lengths.txt"

for d in */
do
    cd $d
for file in *.mp4
do
    printf "%s," "$file" >> $outfile
    mediainfo --Inform="General;%Duration/String3%" "$file" >> $outfile
done

cd ..
done
