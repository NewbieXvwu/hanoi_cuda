nvcc -O3 -arch=native --use_fast_math -Xcompiler "/O2 /fp:fast" hanoi_cuda.cu -o hanoi_cuda.exe
pause