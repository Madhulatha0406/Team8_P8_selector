# STREAM (official source)
`stream.c` is John D. McCalpin's STREAM benchmark (https://www.cs.virginia.edu/stream/), unmodified, taken from the public mirror
https://github.com/jeffhammond/STREAM (file `stream.c`, revision 5.10). Its licence text is in the file header and must stay with it.
Run through `python3 suite.py stream` (compiles with `gcc -O3 -fopenmp`, array size set at compile time, runs at several thread counts).
