#!/bin/bash

if [ "$1" == 1 ]
then
    python3 Q1.py $2 $3 $4 $5
elif [ "$1" == 2 ]
then
    python3 Q2.py $2 $3 $4
else
    echo "Enter valid arguments"
fi