#### 100 days of CUDA Challenge

This is my attempt to write a kernel every day for the next 100 days and document my learning journey.

Mentor : https://github.com/hkproj | https://github.com/hkproj/100-days-of-gpu

Day1

1. [CheckDeviceInfo](day1/checkDeviceInfo.cu) - a kernel to understand every detail about your GPU present in the system.

Day2

2. - [VectorAdd](day2/sumArray.cu) - a kernel to do vector addition
   - learned how to configure and use nvidia nsight to profile kernels and understand bottlenecks

Day3

3. - [TritonVectorAdd](day3/vectorAdd.py) - a triton kernel to do vector addition.
   - watching the gpu mode tutorial on triton - https://www.youtube.com/watch?v=DdTsX6DQk24&ab_channel=GPUMODE

Day4

4. - [TritonGrayScale](day4/grayscale.py) - a triton kernel to grayscale an image.
   - watching the gpu mode tutorial on triton - https://www.youtube.com/watch?v=DdTsX6DQk24&ab_channel=GPUMODE

Day5

5. - [TritonGrayScaleBenchmarking](day5/grayscale_with_benchmark.py) - benchmarking the previous grayscale kernel.

![alt text](assets/grayscale_benchmark.png)
