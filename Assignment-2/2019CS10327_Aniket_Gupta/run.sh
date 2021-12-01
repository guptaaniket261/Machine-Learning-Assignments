#!/bin/bash


if [ "$1" == 1 ]
then
    python3 Q1.py $2 $3 $4

elif [[ $1 = 2 && $4 == 0 && $5 == "a" ]]
then
    python3 2a1.py $2 $3

elif [[ $1 = 2 && $4 = 0 && $5 = "b" ]]
then
    python3 2a2.py $2 $3

elif [[ $1 = 2 && $4 = 0 && $5 = "c" ]]
then
    python3 2a3.py $2 $3

elif [[ $1 = 2 && $4 = 1 && $5 = "a" ]]
then
    python3 2b1.py $2 $3 0

elif [[ $1 = 2 && $4 = 1 && $5 = "b" ]]
then
    python3 2b2.py $2 $3 0

elif [[ $1 = 2 && $4 = 1 && $5 = "c" ]]
then
    python3 2b1.py $2 $3 1
    python3 2b2.py $2 $3 1

elif [[ $1 = 2 && $4 = 1 && $5 = "d" ]]
then
    python3 2b4.py $2 $3
else
    echo "Enter valid arguments"
fi
