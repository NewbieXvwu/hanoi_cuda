nvcc -O3 -arch=native --use_fast_math -Xptxas "-O3" --threads 0 -Xcompiler "/O2,/fp:fast,/Ot,/Qpar" hanoi_cuda.cu -o hanoi_cuda.exe
pause
