# 5. Performance Guidelines
# 5. 性能优化指南

标签（空格分隔）： CUDA

---

[TOC]

# 正文中加粗内容为翻译不太准确的部分，需要仔细斟酌

## 5.1. Overall Performance Optimization Strategies
## 5.1 性能优化策略概要


----------

Performance optimization revolves around three basic strategies:

- Maximize parallel execution to achieve maximum utilization;
- Optimize memory usage to achieve maximum memory throughput;
- Optimize instruction usage to achieve maximum instruction throughput.
 
CUDA性能优化围绕三个基本策略展开：

- 最大化并行度以实现最大利用率;
- 内存优化，以实现最大内存吞吐量;
- 指令优化，以实现最大指令吞吐量。

----------

Which strategies will yield the best performance gain for a particular portion of an application depends on the performance limiters for that portion; optimizing instruction usage of a kernel that is mostly limited by memory accesses will not yield any significant performance gain, for example. Optimization efforts should therefore be constantly directed by measuring and monitoring the performance limiters, for example using the CUDA profiler. Also, comparing the  or  - whichever makes more sense - of a particular kernel to the corresponding peak theoretical throughput of the device indicates how much room for improvement there is for the kernel.

对程序进行优化时不能盲目的使用优化策略，某个优化策略能够获得的性能提升取决于程序的性能瓶颈。比如，一个kernel的性能瓶颈是访存，那么指令类优化策略不会产生明显的性能提升。因此，在进行CUDA优化之前，应该先通过CUDA分析器（NVPP，nvprof）分析出程序的性能瓶颈，再选择对应的优化策略进行优化。直接简单的做法就是，将kernel的某个性能指标（比如，浮点运算吞吐量，floating-point operation throughput，或内存吞吐量，memory throughput）与GPU设备的相应理论峰值进行比较，可以得出该kernel在这方面还有多大的提升空间。

----------

## 5.2. Maximize Utilization
## 5.2  如何最大化利用率
To maximize utilization the application should be structured in a way that it exposes as much parallelism as possible and efficiently maps this parallelism to the various components of the system to keep them busy most of the time.

为了最大化GPU利用率，应该从多个层次尽可能的发掘程序的并行性，从而充分的利用GPU资源，保证GPU的各个模块大部分时间都处于忙碌状态。
下面将从上到下，依次从应用层次，设备层次，Multiprocessor层次三个层次，探究如何最大化GPU利用率

----------

### 5.2.1. Application Level
### 5.2.1  应用层次
At a high level, the application should maximize parallel execution between the host, the devices, and the bus connecting the host to the devices, by using asynchronous functions calls and streams as described in Asynchronous Concurrent Execution. It should assign to each processor the type of work it does best: serial workloads to the host; parallel workloads to the devices.

在应用层次，应该使用异步执行特性（异步函数和流，见 Asynchronous Concurrent Execution章节）最大化主机端（host）任务、设备端（device）任务和主机设备通信任务的并行性。基本准则：串行任务分配到host端运行，并行任务分配到device上运行。

For the parallel workloads, at points in the algorithm where parallelism is broken because some threads need to synchronize in order to share data with each other, there are two cases: Either these threads belong to the same block, in which case they should use __syncthreads() and share data through shared memory within the same kernel invocation, or they belong to different blocks, in which case they must share data through global memory using two separate kernel invocations, one for writing to and one for reading from global memory. The second case is much less optimal since it adds the overhead of extra kernel invocations and global memory traffic. Its occurrence should therefore be minimized by mapping the algorithm to the CUDA programming model in such a way that the computations that require inter-thread communication are performed within a single thread block as much as possible.

在并行任务中，某些情况下一些线程需要进行互相通信，此时需要线程同步操作，常见有以下两种情况，（1）这些线程在一个线程块（block）中，这时使用__syncthreads()函数进行同步即可，并使用共享内存存储数据进行通信；（2）如果不在同一个block中，这种情况下必须使用两个独立的kernel，一个kernel负责向全局内存（Global Memory）写数据，一个kernel负责读数据。很明显，第二种情况会产生额外的kernel调用开销和Global Memory读写开销，不是理想情况。因此在设计并行算法时，应该尽可能将需要通信的线程放到一个block中。

----------

### 5.2.2. Device Level
### 5.2.2. 设备层次
    
At a lower level, the application should maximize parallel execution between the multiprocessors of a device.
    
在设备层次中，应该设计应用程序，最大限度地提高multiprocessors的并行执行。

Multiple kernels can execute concurrently on a device, so maximum utilization can also be achieved by using streams to enable enough kernels to execute concurrently as described in Asynchronous Concurrent Execution.
多个kernel可以同时在一个GPU上运行，如《异步并行执行》章节中所述，应该使用stream技术来保证同时执行的kernel足够多，从而实现最大化利用率。

### 5.2.3. Multiprocessor Level
### 5.2.3. Multiprocessor 层次

At an even lower level, the application should maximize parallel execution between the various functional units within a multiprocessor.
在Multiprocessor层次中，应该最大化Multiprocessor内各功能单元之间的并行执行。

As described in Hardware Multithreading, a GPU multiprocessor relies on thread-level parallelism to maximize utilization of its functional units. Utilization is therefore directly linked to the number of resident warps. At every instruction issue time, a warp scheduler selects a warp that is ready to execute its next instruction, if any, and issues the instruction to the active threads of the warp. The number of clock cycles it takes for a warp to be ready to execute its next instruction is called the latency, and full utilization is achieved when all warp schedulers always have some instruction to issue for some warp at every clock cycle during that latency period, or in other words, when latency is completely "hidden". 

如硬件多线程章节中所述，最大化 GPU multiprocessor各个功能单元的利用率，需要尽可能的最大化线程并行度。也就是说，利用率与执行状态warp的数量有直接关系。在每个指令发射周期，一个warp调度器选择一个处于准备状态的warp（如果有的话）执行其下一条指令，然后发射指令给该warp中的活跃线程。延迟（latency）即一个warp准备好执行下一条指令所花费的时钟周期数。实现充分利用资源，需要在每个时钟周期内，都有一些指令经过warp调度器发射给warp执行，不存在空闲时钟周期，换句话说，就是latency被完全隐藏了。

The number of instructions required to hide a latency of L clock cycles depends on the respective throughputs of these instructions (see Arithmetic Instructions for the throughputs of various arithmetic instructions). Assuming maximum throughput for all instructions, it is: 8L for devices of compute capability 3.x since a multiprocessor issues a pair of instructions per warp over one clock cycle for four warps at a time, as mentioned in Compute Capability 3.x.
完全隐藏长度为L时钟周期的latency所需要的指令数目取决于各指令的吞吐量（即指令执行所需要的时钟周期）（请参阅Arithmetic Instructions以了解各种算术指令的吞吐量）。在计算能力3.x的设备上，假设所有指令的吞吐量都是最大（8时钟周期），那么就需要在一个时钟周期内向4个warp发射*两条*指令，才能隐藏latency（如计算能力3.x章节中所述）。

For devices of compute capability 3.x, the eight instructions issued every cycle are four pairs for four different warps, each pair being for the same warp.

对于计算能力为3.x的设备，每个周期向4个不同的warp发射8条指令，即一个warp 2条。
The most common reason a warp is not ready to execute its next instruction is that the instruction's input operands are not available yet.
    
一个warp没有准备好执行下一条指令的最常见原因是指令的输入操作数还未到位。

If all input operands are registers, latency is caused by register dependencies, i.e., some of the input operands are written by some previous instruction(s) whose execution has not completed yet. In the case of a back-to-back register dependency (i.e., some input operand is written by the previous instruction), the latency is equal to the execution time of the previous instruction and the warp schedulers must schedule instructions for different warps during that time. Execution time varies depending on the instruction, but it is typically about 11 clock cycles for devices of compute capability 3.x, which translates to 44 warps for devices of compute capability 3.x (assuming that warps execute instructions with maximum throughput, otherwise fewer warps are needed). This is also assuming enough instruction-level parallelism so that schedulers are always able to issue pairs of instructions for each warp.

举例说明，如果指令I的输入操作数都位于寄存器，那么latency由寄存器是否依赖其他指令决定的，通俗来说，就是一些输入操作数是前一条指令的结果数，这种情况下，指令I的latency等于前一条指令的执行时间，在这期间，该warp处于空闲（idle）状态，warp调度器必须为其他warp调度指令。执行时间因指令而异，但对于计算能力3.x的设备而言，通常约为11个时钟周期，即需要44个warp同时执行才能完全隐藏延迟（这里假设warp执行每一条指令的时间都是11时钟周期）。当然，这种计算方式的假设前提是kernel有足够的指令集并行性，即warp调度器始终都在向warp发射指令。

If some input operand resides in off-chip memory, the latency is much higher: 200 to 400 clock cycles for devices of compute capability 3.x. The number of warps required to keep the warp schedulers busy during such high latency periods depends on the kernel code and its degree of instruction-level parallelism. In general, more warps are required if the ratio of the number of instructions with no off-chip memory operands (i.e., arithmetic instructions most of the time) to the number of instructions with off-chip memory operands is low (this ratio is commonly called the arithmetic intensity of the program). For example, assume this ratio is 30, also assume the latencies are 300 cycles on devices of compute capability 3.x. Then about 40 warps are required for devices of compute capability 3.x (with the same assumptions as in the previous paragraph).
如果指令的输入操作数位于片外（off-chip）存储器中，那么latency会更长：对于计算能力3.x的设备，通常需要200至400个时钟周期。 在如此长的等待时间内保持warp调度器繁忙所需的warp数量取决于kernel代码及其指令级并行度。 这里引入一个比率概念，这个比率通常被称为运算强度（Arithmetic Intensity），是指一个程序内，输入操作数不位于片外内存上的指令数目（通常是算数指令） / 输入操作数位于片外内存上的指令数目。Arithmetic Intensity越小，代表需要同时执行更多的warp才能隐藏latency。 例如，在计算能力3.x的设备上，假设运算强度为30，latency为300个周期，那么大概需要40个warp同时执行才能隐藏latency。

Another reason a warp is not ready to execute its next instruction is that it is waiting at some memory fence (Memory Fence Functions) or synchronization point (Memory Fence Functions). A synchronization point can force the multiprocessor to idle as more and more warps wait for other warps in the same block to complete execution of instructions prior to the synchronization point. Having multiple resident blocks per multiprocessor can help reduce idling in this case, as warps from different blocks do not need to wait for each other at synchronization points.
warp没有准备好执行下一条指令的另一个原因是，它是等待某个memory fence（Memory Fence Functions）或同步操作（Memory Synchronization Functions）。同步操作会强制multiprocessor空闲，因为同一个block内的warp必须在同步点等待直到所有warp都执行到同步点。在这种情况下，在一个multiprocessor内执行多个block有利于减少空闲，因为同步操作仅在block内有效。

The number of blocks and warps residing on each multiprocessor for a given kernel call depends on the execution configuration of the call (Execution Configuration), the memory resources of the multiprocessor, and the resource requirements of the kernel as described in Hardware Multithreading. Register and shared memory usage are reported by the compiler when compiling with the -ptxas-options=-v option.
对于给定的kernel，每一个multiprocessor内block和warp数量是由三个因素决定的，分别是调用时执行配置（见执行配置章节，就是调用核函数时的四个参数），multiprocessor的内存资源以及kernel的资源需求（即kernel内所使用的寄存器和共享内存等资源，见硬件多线程中描述）。 当使用-ptxas-options = -v选项进行编译时，编译器会打印出寄存器和共享内存使用情况。

The total amount of shared memory required for a block is equal to the sum of the amount of statically allocated shared memory and the amount of dynamically allocated shared memory.
一个block所需的共享内存总量等于静态分配的共享内存量与动态分配的共享内存量的总和（注：共享内存有静态和动态两种分配方式，详见https://devblogs.nvidia.com/using-shared-memory-cuda-cc/）。

The number of registers used by a kernel can have a significant impact on the number of resident warps. For example, for devices of compute capability 6.x, if a kernel uses 64 registers and each block has 512 threads and requires very little shared memory, then two blocks (i.e., 32 warps) can reside on the multiprocessor since they require 2x512x64 registers, which exactly matches the number of registers available on the multiprocessor. But as soon as the kernel uses one more register, only one block (i.e., 16 warps) can be resident since two blocks would require 2x512x65 registers, which are more registers than are available on the multiprocessor. Therefore, the compiler attempts to minimize register usage while keeping register spilling (see Device Memory Accesses) and the number of instructions to a minimum. Register usage can be controlled using the maxrregcount compiler option or launch bounds as described in Launch Bounds.
kernel中每个线程使用的寄存器数量对活跃warp的数量有重大影响。 例如，对于计算能力6.x的设备，如果每个thread使用64个寄存器并且每个block有512个线程（假设只需要很少的共享内存，即共享内存不会造成限制），则一个 multiprocessor 同时只能执行两个block（即32个warp），因为它们需要2x512x64寄存器，与multiprocessor上可用的寄存器数量完全一致。 这种情况下，如果一个thread哪怕多使用一个寄存器，那么只能执行一个block（即16个warp），因为两个block需要2×512×65个寄存器，多于multiprocessor上可用的寄存器数量。 因此，编译器进行编译优化，尽量减少寄存器使用量，从而防止寄存器溢出问题发生（请参阅“Device Memory Accesses”）。 开发者可以使用maxrregcount编译器选项或launch bounds来控制寄存器的使用量，如Launch Bounds章节中所述。

Each double variable and each long long variable uses two registers.
每个double和long long类型占用两个寄存器。

The effect of execution configuration on performance for a given kernel call generally depends on the kernel code. Experimentation is therefore recommended. Applications can also parameterize execution configurations based on register file size and shared memory size, which depends on the compute capability of the device, as well as on the number of multiprocessors and memory bandwidth of the device, all of which can be queried using the runtime (see reference manual).

执行配置对于给定kernel的性能影响通常取决于kernel代码。因此建议进行实验从而在当前设备上获得最佳性能（注：比如实验不同block大小下kernel的性能）。核函数的最佳运行配置是跟设备有关的，开发者可以根据寄存器数量和共享内存大小来修改执行配置，设备的寄存器数量，共享内存大小，设备的计算能力，multiprocessors的数量以及内存带宽，所有的这些参数都可以手册中查到。

The number of threads per block should be chosen as a multiple of the warp size to avoid wasting computing resources with under-populated warps as much as possible.
在设置执行配置时，block大小应该尽量为warp的倍数，否则会生成一些空闲thread来对齐，造成计算资源浪费。注：GPU内的最小执行单元是warp，如果设置block大小为20，那么实际运行中，会生成12个空闲thread补齐一个warp来执行。

#### 5.2.3.1. Occupancy Calculator
#### 5.2.3.1  占用率计算器

Several API functions exist to assist programmers in choosing thread block size based on register and shared memory requirements.
有几个API函数可以帮助程序员根据kernel内寄存器和共享内存的使用量计算出最优线程块大小。

The occupancy calculator API, cudaOccupancyMaxActiveBlocksPerMultiprocessor, can provide an occupancy prediction based on the block size and shared memory usage of a kernel. This function reports occupancy in terms of the number of concurrent thread blocks per multiprocessor.
    
Occupancy Calculator API， cudaOccupancyMaxActiveBlocksPerMultiprocessor，可以根据kernel的block大小、共享内存使用情况和每个multiprocessor内的资源量计算出每个multiprocessor内可并发block数目，然后再计算出设备占用率（occupancy）。

Note that this value can be converted to other metrics. Multiplying by the number of warps per block yields the number of concurrent warps per multiprocessor; further dividing concurrent warps by max warps per multiprocessor gives the occupancy as a percentage.
    
值得注意的是，占用率可以转换为其他指标。占用率乘以block内warp数量会得到每个multiprocessor的并发warp数量;并发warp数量除以每个多处理器的最大warp数可以得到占有率的百分比形式。
    
The occupancy-based launch configurator APIs, cudaOccupancyMaxPotentialBlockSize and cudaOccupancyMaxPotentialBlockSizeVariableSMem, heuristically calculate an execution configuration that achieves the maximum multiprocessor-level occupancy.
    
API cudaOccupancyMaxPotentialBlockSize和cudaOccupancyMaxPotentialBlockSizeVariableSMem 能够启发式地计算出最优执行配置，实现最大 multiprocessor 占用率。

The following code sample calculates the occupancy of MyKernel. It then reports the occupancy level with the ratio between concurrent warps versus maximum warps per multiprocessor.

以下代码示例计算MyKernel的占用情况，输出为活跃warp与multiprocessor最大warp之间的比率。

```c++
// Device code
__global__ void MyKernel(int *d, int *a, int *b)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    d[idx] = a[idx] * b[idx];
}

// Host code
int main()
{
    int numBlocks;        // Occupancy in terms of active blocks
    int blockSize = 32;

    // These variables are used to convert occupancy to warps
    int device;
    cudaDeviceProp prop;
    int activeWarps;
    int maxWarps;

    cudaGetDevice(&device);
    cudaGetDeviceProperties(&prop, device);
    
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &numBlocks,
        MyKernel,
        blockSize,
        0);

    activeWarps = numBlocks * blockSize / prop.warpSize;
    maxWarps = prop.maxThreadsPerMultiProcessor / prop.warpSize;

    std::cout << "Occupancy: " << (double)activeWarps / maxWarps * 100 << "%" << std::endl;
    
    return 0;
}
```
The following code sample configures an occupancy-based kernel launch of MyKernel according to the user input.
以下代码示例，根据输入数据大小，使用占用率API计算出运行配置，并运行kernel。
```c++
// Device code
__global__ void MyKernel(int *array, int arrayCount)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < arrayCount) {
        array[idx] *= array[idx];
    }
}

// Host code
int launchMyKernel(int *array, int arrayCount)
{
    int blockSize;      // The launch configurator returned block size
    int minGridSize;    // The minimum grid size needed to achieve the
                        // maximum occupancy for a full device
                        // launch
    int gridSize;       // The actual grid size needed, based on input
                        // size

    cudaOccupancyMaxPotentialBlockSize(
        &minGridSize,
        &blockSize,
        (void*)MyKernel,
        0,
        arrayCount);

    // Round up according to array size
    gridSize = (arrayCount + blockSize - 1) / blockSize;

    MyKernel<<<gridSize, blockSize>>>(array, arrayCount);
    cudaDeviceSynchronize();

    // If interested, the occupancy can be calculated with
    // cudaOccupancyMaxActiveBlocksPerMultiprocessor

    return 0;
}
```
The CUDA Toolkit also provides a self-documenting, standalone occupancy calculator and launch configurator implementation in <CUDA_Toolkit_Path>/include/cuda_occupancy.h for any use cases that cannot depend on the CUDA software stack. A spreadsheet version of the occupancy calculator is also provided. The spreadsheet version is particularly useful as a learning tool that visualizes the impact of changes to the parameters that affect occupancy (block size, registers per thread, and shared memory per thread).
对于不能直接使用CUDA软件环境的程序，CUDA Toolkit提供了独立的自带文档的占用率计算器和启用配置器，目录为 <CUDA_Toolkit_Path>/include/cuda_occupancy.h。 还提供了电子表格版本的占用率计算器。 电子表格版本是一个有效的学习工具，它可以可视化一些参数变化对占用率的影响，包括block大小，每个线程的寄存器使用量和共享内存使用量等。

## 5.3. Maximize Memory Throughput
## 5.3. 最大化内存吞吐量
The first step in maximizing overall memory throughput for the application is to minimize data transfers with low bandwidth.
That means minimizing data transfers between the host and the device, as detailed in Data Transfer between Host and Device, since these have much lower bandwidth than data transfers between global memory and the device    .
That also means minimizing data transfers between global memory and the device by maximizing use of on-chip memory: shared memory and caches (i.e., L1 cache and L2 cache available on devices of compute capability 2.x and higher, texture cache and constant cache available on all devices).
应用程序最大化整体内存吞吐量的首要步骤是，尽可能的减少低带宽的数据传输。换句话说就是要最大限度地减少主机和设备之间的数据传输，详见主机和设备之间的数据传输章节，因为主机和设备之间带宽非常低，远远低于全局内存的数据传输带宽。在设计CUDA应用程序时，要尽可能的使用片上（on-chip）内存，即共享内存和cache，最大限度地减少主机和设备之间的数据传输，所有的GPU都有纹理cache和常量cache，计算能力大于等于2.x的设备上有L1 和 L2 cache。

Shared memory is equivalent to a user-managed cache: The application explicitly allocates and accesses it. As illustrated in CUDA C Runtime, a typical programming pattern is to stage data coming from device memory into shared memory; in other words, to have each thread of a block:

共享内存可以理解为用户可以管理的L1 cache（注：共享内存和L1 cache使用同一块硬件）：应用程序需要显式分配并访问共享内存。 如CUDA C Runtime章节所示，典型的共享内存使用模式是将全局内存的数据放入共享内存; 换句话说，每个thread内的流程如下：

Load data from device memory to shared memory,
Synchronize with all the other threads of the block so that each thread can safely read shared memory locations that were populated by different threads,
Process the data in shared memory,
Synchronize again if necessary to make sure that shared memory has been updated with the results,
Write the results back to device memory.

 1. 将全局内存加载到共享内存；
 2. 在block内进行同步操作，以保证每一个thread可以安全的读取共享内存；
 3. 计算；
 4. 进行必要的同步操作。要确保共享内存已更新结果；
 5. 将结果写回全局内存；

For some applications (e.g., for which global memory access patterns are data-dependent), a traditional hardware-managed cache is more appropriate to exploit data locality. As mentioned in Compute Capability 3.x and Compute Capability 7.x, for devices of compute capability 3.x and 7.x, the same on-chip memory is used for both L1 and shared memory, and how much of it is dedicated to L1 versus shared memory is configurable for each kernel call.

而常规的cache是由硬件管理的。对于某些访存密集型应用，这类cache能够更好地利用数据局部性。正如Compute Capability 3.x和Compute Capability 7.x中所述，对于计算能力3.x和7.x的设备，L1 cache和共享内存使用相同的片上存储器，执行每一个kernel之前，可以使用相关API配置 L1 cache和共享内存的大小。
    
The throughput of memory accesses by a kernel can vary by an order of magnitude depending on access pattern for each type of memory. The next step in maximizing memory throughput is therefore to organize memory accesses as optimally as possible based on the optimal memory access patterns described in Device Memory Accesses. This optimization is especially important for global memory accesses as global memory bandwidth is low, so non-optimal global memory accesses have a higher impact on performance.

不同内存的访存模式对kernel的访存吞吐量有非常大的影响。因此最大化内存吞吐量的第二步就是根据每种内存的最佳访存模式来设计kernel的内存访存模式。全局内存带宽最低，因此优化对全局内存访问模式往往能取得明显的性能提升。

### 5.3.1. Data Transfer between Host and Device
### 5.3.1 如何进行host和device之间的数据传输

Applications should strive to minimize data transfer between the host and the device. One way to accomplish this is to move more code from the host to the device, even if that means running kernels with low parallelism computations. Intermediate data structures may be created in device memory, operated on by the device, and destroyed without ever being mapped by the host or copied to host memory.
应用程序应尽量减少主机和设备之间的数据传输。 实现此目的的一种方法是将更多的运算任务从主机端移动到设备端，甚至可以为此牺牲kernel的并行性。这种情况下，计算步骤产生的中间数据一般存放在设备端内存中，可以有效地避免拷贝到主机端内存中。

Also, because of the overhead associated with each transfer, batching many small transfers into a single large transfer always performs better than making each transfer separately.
每次内存传输会有一些额外的开销，因此将多个小内存传输合并成一次大内存传输会获得一定的性能提升。

On systems with a front-side bus, higher performance for data transfers between host and device is achieved by using page-locked host memory as described in Page-Locked Host Memory.
在具有前端总线的系统上，与主机内存相比，使用页锁定内存与设备端内存的传输性能更优，详细见Page-Locked Host Memory章节。

In addition, when using mapped page-locked memory (Mapped Memory), there is no need to allocate any device memory and explicitly copy data between device and host memory. Data transfers are implicitly performed each time the kernel accesses the mapped memory. For maximum performance, these memory accesses must be coalesced as with accesses to global memory (see Device Memory Accesses). Assuming that they are and that the mapped memory is read or written only once, using mapped page-locked memory instead of explicit copies between device and host memory can be a win for performance.
此外，在使用映射页面锁定内存（映射内存）时，不需要手动申请设备内存和显式地调用API函数在主机和设备之间传输数据。当在kernel中访问Mapped Memory时，会隐式地进行数据传输。 为获得最佳性能，像访问全局内存一样，这些内存访问会被合并（请参阅Device Memory Accesses章节）。 假设内存访问被合并，且Mapped Memory仅被读取或写入一次，这种情况下，使用映射的页面锁定内存的性能优于显式地主机和设备之间数据传输。

On integrated systems where device memory and host memory are physically the same, any copy between host and device memory is superfluous and mapped page-locked memory should be used instead. Applications may query a device is integrated by checking that the integrated device property (see Device Enumeration) is equal to 1.
在设备内存和主机内存在同一个物理存储器的集成系统上，主机和设备内存之间的任何数据拷贝都是多余的，应该使用映射页面锁定内存。可以通过检查集成设备属性（请参阅Device Enumeration章节），如果等于1，则代表是集成设备。

### 5.3.2. Device Memory Accesses
### 5.3.2 如何访问设备端内存

An instruction that accesses addressable memory (i.e., global, local, shared, constant, or texture memory) might need to be re-issued multiple times depending on the distribution of the memory addresses across the threads within the warp. How the distribution affects the instruction throughput this way is specific to each type of memory and described in the following sections. For example, for global memory, as a general rule, the more scattered the addresses are, the more reduced the throughput is.
一个warp内的所有活跃线程同一时刻执行相同的指令。一条访问可寻址存储器（即，全局内存，本地内存，共享内存，常量内存或纹理内存）的指令可能会被发射多次，这取决于warp内线程所访问数据在存储器中的地址是如何分布的。
在下面各小节中，将介绍对于每种内存，不同的分布对指令吞吐量的影响。举个例子，对于全局内存，一般来讲，一个warp内线程所访存数据的地址越分散，吞吐量越低。

Global Memory
全局内存

Global memory resides in device memory and device memory is accessed via 32-, 64-, or 128-byte memory transactions. These memory transactions must be naturally aligned: Only the 32-, 64-, or 128-byte segments of device memory that are aligned to their size (i.e., whose first address is a multiple of their size) can be read or written by memory transactions.
全局内存位于设备内存中，设备内存的每次内存事务的大小必须是32,64或128字节。内存事务必须对齐，即只有大小为32,64或128字节且首地址是其大小倍数的设备内存段才可以进行内存访问。

When a warp executes an instruction that accesses global memory, it coalesces the memory accesses of the threads within the warp into one or more of these memory transactions depending on the size of the word accessed by each thread and the distribution of the memory addresses across the threads. In general, the more transactions are necessary, the more unused words are transferred in addition to the words accessed by the threads, reducing the instruction throughput accordingly. For example, if a 32-byte memory transaction is generated for each thread's 4-byte access, throughput is divided by 8.
当一个warp执行访问全局内存的指令时，32个线程会发射出32条访问内存指令，合并访问操作会将这32条访存指令合并成一条或者多条，具体取决于每个线程访问数据的大小以及所访问数据的地址分布。一般来说，合并后的内存事务越多，传输的冗余数据越多（传输的数据大小大于warp所需要的数据大小），指令吞吐量就越低。举个例子，如果每个线程只需要访问4字节的数据，但却产生了32字节的内存事务，那么吞吐量为1/8.

How many transactions are necessary and how much throughput is ultimately affected varies with the compute capability of the device. Compute Capability 3.x, Compute Capability 5.x, Compute Capability 6.x and Compute Capability 7.x give more details on how global memory accesses are handled for various compute capabilities.
具体的合并结果以及对吞吐量的影响取决于设备的计算能力。 在计算能力3.x，计算能力5.x，计算能力6.x和计算能力7.x章节里分别详细介绍了，在不同计算能力下是如何处理各种全局内存访问情况的。

To maximize global memory throughput, it is therefore important to maximize coalescing by:
通过以下措施可以有效的提高内存合并访问，从而最大限度地提高全局内存吞吐量。

 - Following the most optimal access patterns based on Compute
   Capability 3.x, Compute Capability 5.x, Compute Capability 6.x and
   Compute Capability 7.x, 
 - Using data types that meet the size and
   alignment requirement detailed in Device Memory Accesses, 
 - Padding
   data in some cases, for example, when accessing a two-dimensional
   array as described in Device Memory Accesses.

 - 遵循基于计算能力3.x，计算能力5.x，计算能力6.x和计算能力7.x章节内描述的最优访存模式；
 - 使用满足大小和对齐要求的数据类型，详细见Device Memory Accesses章节；
 - 在某些情况下填充数据，例如Device Memory Accesses中所述的访问二维数组例子。

Size and Alignment Requirement
对齐要求
Global memory instructions support reading or writing words of size equal to 1, 2, 4, 8, or 16 bytes. Any access (via a variable or a pointer) to data residing in global memory compiles to a single global memory instruction if and only if the size of the data type is 1, 2, 4, 8, or 16 bytes and the data is naturally aligned (i.e., its address is a multiple of that size).
全局存储器指令支持读取或写入1,2,4,8或16字节的数据。当且仅当所访问数据数据类型的大小是1,2,4,8或16字节且数据的地址是其大小的倍数时，该内存访问才会被编译成一条全局内存访问指令。

If this size and alignment requirement is not fulfilled, the access compiles to multiple instructions with interleaved access patterns that prevent these instructions from fully coalescing. It is therefore recommended to use types that meet this requirement for data that resides in global memory.
如果此大小和对齐要求未满足，则访问将编译为具有交叉存取模式的多条指令，以防止这些指令完全合并。 因此，建议使用满足要求的类型来存储全局内存中的数据。

The alignment requirement is automatically fulfilled for the built-in types of char, short, int, long, longlong, float, double like float2 or float4.
内置类型的char，short，int，long，longlong，float，double类型（如float2或float4）会自动满足对齐要求。

For structures, the size and alignment requirements can be enforced by the compiler using the alignment specifiers __align__(8) or __align__(16), such as
对于结构体，使用对齐说明符__align __（8）或__align __（16）来让编译器强制满足大小和对齐要求，例如

```c++
struct __align__(8) {
    float x;
    float y;
};
or

struct __align__(16) {
    float x;
    float y;
    float z;
};
```
Any address of a variable residing in global memory or returned by one of the memory allocation routines from the driver or runtime API is always aligned to at least 256 bytes.
由驱动API或者运行API分配的内存的地址，都是对齐至少256个字节。

Reading non-naturally aligned 8-byte or 16-byte words produces incorrect results (off by a few words), so special care must be taken to maintain alignment of the starting address of any value or array of values of these types. A typical case where this might be easily overlooked is when using some custom global memory allocation scheme, whereby the allocations of multiple arrays (with multiple calls to cudaMalloc() or cuMemAlloc()) is replaced by the allocation of a single large block of memory partitioned into multiple arrays, in which case the starting address of each array is offset from the block's starting address.
读取非自然对齐的8字节或16字节字会产生不正确的结果，因此在为变量或者数组分配空间时，一定要注意保证起始地址的对齐。在自定义的全局内存分配方案中，这个问题很容易被忽略。自定义全局内存分配分案就是先申请一大块全局内存，然后通过地址偏移的方式为变量或者数组分配空间，优点是减少多次申请全局内存造成的额外开销。（自定义的全局内存分配方案可参照博客《[CUDA进阶第六篇-GPU资源（显存、句柄等）管理][1]》）

Two-Dimensional Arrays
二维数组
    
A common global memory access pattern is when each thread of index (tx,ty) uses the following address to access one element of a 2D array of width width, located at address BaseAddress of type type* (where type meets the requirement described in Maximize Utilization):

BaseAddress + width * ty + tx
全局内存内二维数组的访问模式是根据每个线程的坐标ID（tx，ty）访问二维数据中的对应的数据，假设二维数组的首地址为BaseAddress，宽度为width，则代码为：BaseAddress +width* ty + tx

For these accesses to be fully coalesced, both the width of the thread block and the width of the array must be a multiple of the warp size.
为了使内存访问完全合并，block的宽度和数组的宽度必须是warp大小的倍数。

In particular, this means that an array whose width is not a multiple of this size will be accessed much more efficiently if it is actually allocated with a width rounded up to the closest multiple of this size and its rows padded accordingly. The cudaMallocPitch() and cuMemAllocPitch() functions and associated memory copy functions described in the reference manual enable programmers to write non-hardware-dependent code to allocate arrays that conform to these constraints.
特殊情况，如果数组的宽度不是warp大小的倍数，那么将数组的宽度向上补齐为warp的倍数，能够有效的提高访存效率。这正是cudaMallocPitch（）和cuMemAllocPitch（） API和对应的复制API的功能，这些API使程序员能够不编写硬件相关的代码的情况下，实现申请分配满足这些约束的数据空间。

Local Memory
本地内存

Local memory accesses only occur for some automatic variables as mentioned in Variable Memory Space Specifiers. Automatic variables that the compiler is likely to place in local memory are:

 - Arrays for which it cannot determine that they are indexed with constant quantities, 
 - Large structures or arrays that would consume too much register space, 
 - Any variable if the kernel uses more registers than available (this is also known as register spilling).
只有一些自动变量会被存放到本地内存，见Variable Memory Space Specifiers章节中所述。编译器可能将以下三种情况的数据放到本地内存中：
 - kernel中动态申请的数组，即在编译时无法确定申请大小的数组， 
 - 会消耗太多的寄存器空间大型结构体或数组，
 - 寄存器溢出，即寄存器已经被用完。

Inspection of the PTX assembly code (obtained by compiling with the -ptx or-keep option) will tell if a variable has been placed in local memory during the first compilation phases as it will be declared using the .local mnemonic and accessed using the ld.local and st.local mnemonics. Even if it has not, subsequent compilation phases might still decide otherwise though if they find it consumes too much register space for the targeted architecture: Inspection of the cubin object using cuobjdump will tell if this is the case. Also, the compiler reports total local memory usage per kernel (lmem) when compiling with the --ptxas-options=-v option. Note that some mathematical functions have implementation paths that might access local memory.
查看PTX汇编代码（使用-ptx或-keep编译选项编译时可以获得kernel的PTX代码），可以看到在编译第一阶段，满足以上条件的变量被放到了本地内存中。被放到本地内存的变量的声明会标记.local符号，访问会标记ld.local 和 st.local符号。对于没有被放入本地内存的变量，但消耗了太多了寄存器，在后续的编译阶段中也可能会被放到本地内存中，使用cuobjdump命令查看cubin文件可以看到是否有这种情况发生。此外，当使用--ptxas-options = -v选项进行编译时，编译器会报告每个内核的本地内存使用情况（lmem）。请注意，一些数学函数的具体实现过程可能会访问本地内存。

The local memory space resides in device memory, so local memory accesses have same high latency and low bandwidth as global memory accesses and are subject to the same requirements for memory coalescing as described in Device Memory Accesses. Local memory is however organized such that consecutive 32-bit words are accessed by consecutive thread IDs. Accesses are therefore fully coalesced as long as all threads in a warp access the same relative address (e.g., same index in an array variable, same member in a structure variable).
本地内存位于设备内存中，因此本地内存与全局内存一样，高延迟和低带宽，并且也有内存访问合并机制（见Device Memory Accesses所述）。

**然而，本地存储器被组织为使得连续的32位字由连续的线程ID访问。** 因此，只要warp中的所有线程访问相同的相对地址（例如，都访问数组中某个元素或结构体中的某个成员变量），所有访问请求会被完全合并。

On some devices of compute capability 3.x local memory accesses are always cached in L1 and L2 in the same way as global memory accesses (see Compute Capability 3.x).

On devices of compute capability 5.x and 6.x, local memory accesses are always cached in L2 in the same way as global memory accesses (see Compute Capability 5.x and Compute Capability 6.x).

在计算能力3.x的某些设备上，与全局内存相同，访问本地内存会缓存在L1和L2中（请Compute Capability 3.x）。

在计算能力5.x和6.x的设备上，与全局内存相同，访问本地内存会缓存在L2中（请Compute Capability 5.x and Compute Capability 6.x）。

Shared Memory
Because it is on-chip, shared memory has much higher bandwidth and much lower latency than local or global memory.
共享内存是片上内存，与本地内存和全局内存相比，带宽更高，延迟更低。

To achieve high bandwidth, shared memory is divided into equally-sized memory modules, called banks, which can be accessed simultaneously. Any memory read or write request made of n addresses that fall in n distinct memory banks can therefore be serviced simultaneously, yielding an overall bandwidth that is n times as high as the bandwidth of a single module.
为了实现高带宽，共享内存被分成大小相同的内存模块，称为bank，多个bank可以同时被访问。
因此，当访问共享内存的n个访存请求的地址，正好落入n个bank内，这n个访存请求就可以同时访问共享内存，此时的总带宽是访问单个bank带宽的n倍。

However, if two addresses of a memory request fall in the same memory bank, there is a bank conflict and the access has to be serialized. The hardware splits a memory request with bank conflicts into as many separate conflict-free requests as necessary, decreasing throughput by a factor equal to the number of separate memory requests. If the number of separate memory requests is n, the initial memory request is said to cause n-way bank conflicts.

但是，如果一个内存请求的两个地址落在同一个bank中，则会发生bank conflict，此时这两个访问只能顺序执行。通常用n-way bank conflict来衡量bank conflict程度，即需要经过n次访问共享内存，才能得到warp所需要的数据。如果同一个 warp 中的所有线程访问一个 bank 中的 32 个不同地址，则需要分 32 次，称为 32-way bank conflict。

To get maximum performance, it is therefore important to understand how memory addresses map to memory banks in order to schedule the memory requests so as to minimize bank conflicts. This is described in Compute Capability 3.x, Compute Capability 5.x, Compute Capability 6.x, and Compute Capability 7.x for devices of compute capability 3.x, 5.x, 6.x and 7.x, respectively.

为了获得最佳性能，必须了解内存地址是如何映射到共享内存bank上的，从而有效地设计访存模式，以最小化bank conflict。

Constant Memory
常量内存
The constant memory space resides in device memory and is cached in the constant cache.
常量内存位于设备内存中，并有对应的常量缓存。

A request is then split into as many separate requests as there are different memory addresses in the initial request, decreasing throughput by a factor equal to the number of separate requests.

The resulting requests are then serviced at the throughput of the constant cache in case of a cache hit, or at the throughput of device memory otherwise.


**然后，将请求分割为多个独立请求，因为初始请求中存在不同的内存地址，从而将吞吐量降低了一个等于单独请求数的因数。
然后在缓存命中的情况下，以恒定缓存的吞吐量来处理所产生的请求，否则以设备存储器的吞吐量来处理所产生的请求。**

Texture and Surface Memory
    The texture and surface memory spaces reside in device memory and are cached in texture cache, so a texture fetch or surface read costs one memory read from device memory only on a cache miss, otherwise it just costs one read from texture cache. The texture cache is optimized for 2D spatial locality, so threads of the same warp that read texture or surface addresses that are close together in 2D will achieve best performance. Also, it is designed for streaming fetches with a constant latency; a cache hit reduces DRAM bandwidth demand but not fetch latency.

texture Memory和surface Memory位于设备内存中，并有对应的texture cache。texture cache针对2D空间局部性进行了优化，所以当warp内的线程读取texture Memory或surface Memory时，如果访存地址在二维坐标空间上越接近，则性能越好。
**此外，它设计用于具有恒定延迟的流式抓取;高速缓存命中减少了DRAM带宽需求，但不能提取等待时间。**

Reading device memory through texture or surface fetching present some benefits that can make it an advantageous alternative to reading device memory from global or constant memory:

 - If the memory reads do not follow the access patterns that global or constant memory reads must follow to get good performance, higher bandwidth can be achieved providing that there is locality in the texture fetches or surface reads;
 - Addressing calculations are performed outside the kernel by dedicated units;
 - Packed data may be broadcast to separate variables in a single operation;
 - 8-bit and 16-bit integer input data may be optionally converted to 32 bit floating-point values in the range [0.0, 1.0] or [-1.0, 1.0] (see Texture Memory).

texture Memory或surface Memory的一些特性，使其在某些场景下，性能优于全局内存或常量内存：

 - 如果内存读取不满足全局内存或常量内存的最佳访存规则，且具有一定的局部性，那么使用texture Memory或surface Memory会获得更佳的性能。
 -  texture Memory和surface Memory的寻址计算kernel之外由专用单元执行;
 -  ** 压缩过的数据可以在一次操作中广播到其他变量中。**
 -  8-bit和16-bit的整型数据，可以选择性的转化为[0.0，1.0]或[-1.0,1.0]范围内的32位浮点值（请参阅Texture Memory）


  [1]: https://blog.csdn.net/litdaguang/article/details/79330973
