#!/bin/bash

for f in 2019*/
do
    cd $f
    cd 2019*
    for folder in */
    do
	mv $folder* ../videos/00_03/
    done
    cd ..
    cd ..
 done
