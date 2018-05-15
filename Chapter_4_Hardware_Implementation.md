
# Chapter 4. Hardware Implementation  
 
 第四章 硬件实现  


---

>The NVIDIA GPU architecture is built around a scalable array of multithreaded Streaming Multiprocessors (SMs).    

NVIDIA GPU 架构是围绕可扩展的多线程阵列构建的 Streaming Multiprocessors（SMs）。

>When a CUDA program on the host CPU invokes a kernel grid, the blocks of the grid are enumerated and distributed to multiprocessors with available execution capacity.   

当主机端的 CUDA 程序使用给定的执行配置调用 kernel 时，CUDA 设备对启动的线程块进行统计，并将其中一些线程块分配到有执行资源的 SM 上。

>The threads of a thread block execute concurrently on one multiprocessor, and multiple thread blocks can execute concurrently on one multiprocessor. 

这些线程块内的线程在 SM 上并发地执行，且多个线程块也可以在一个 SM 上并发地执行。

>As thread blocks terminate, new blocks are launched on the vacated multiprocessors.

线程块终止时，便在空闲的 SM 上分配新的线程块。  

---

>A multiprocessor is designed to execute hundreds of threads concurrently. To manage such a large amount of threads, it employs a unique architecture called SIMT (Single-Instruction, Multiple-Thread) that is described in SIMT Architecture. 

SM 被设计为可以并发地执行上百个线程。为了管理好如此多的线程，SM 采用了一种称为 SIMT（Single-Instruction, Multiple-Thread）的独特架构，[4.1](#4.1)节对此进行了详细描述。

>The instructions are pipelined to leverage instruction-level parallelism within a single thread, as well as thread-level parallelism extensively through simultaneous hardware multithreading as detailed in Hardware Multithreading. 

[4.2](#4.2)节则对硬件多线程进行描述。SM 执行线程时为了利用单个线程内的指令级并行性会将指令流水线化，而 SM 则通过同步硬件多线程来实现广泛的线程级并行。

>Unlike CPU cores they are issued in order however and there is no branch prediction and no speculative execution.

与 CPU 核心不同的是，线程的分配是按顺序进行的，并且没有分支预测与猜测执行。  

--- 

>SIMT Architecture and Hardware Multithreading describe the architecture features of the streaming multiprocessor that are common to all devices. Compute Capability 3.x, Compute Capability 5.x, Compute Capability 6.x, and Compute Capability 7.x provide the specifics for devices of compute capabilities 3.x, 5.x, 6.x, and 7.x respectively.
>The NVIDIA GPU architecture uses a little-endian representation.

[4.1](#4.1)节和[4.2](#4.2)节描述的 SM 体系结构特征对所有设备都是通用的。[link]() [计算能力3.x]()、[计算能力5.x]()、[计算能力6.x]() 和 [计算能力7.x]() 分别提供了对应计算能力设备的具体细节。

--- 

## 4.1 SIMT Architecture

4.1 SIMT架构

>The multiprocessor creates, manages, schedules, and executes threads in groups of 32 parallel threads called warps. 

SM 创建、管理、调度和执行线程是以每32个线程为一组构成的 warps 来进行的。

>Individual threads composing a warp start together at the same program address, but they have their own instruction address counter and register state and are therefore free to branch and execute independently. 

一个 warp 内的所有线程在同一个程序地址处组成一个 warp 起始点，但这些线程有它们自己的指令地址计数器和寄存器状态，因此可以自由分支并独立执行。

>The term warp originates from weaving, the first parallel thread technology.   

warp 这个术语来源于编织，这是第一种线程并行技术。

>A half-warp is either the first or second half of a warp. A quarter-warp is either the first, second, third, or fourth quarter of a warp.

half-warp 指一个线程束的前一半或后一半，quarter-warp 指一个线程束的第一、第二、第三或第四个四分之一。  

---

>When a multiprocessor is given one or more thread blocks to execute, it partitions them into warps and each warp gets scheduled by a warp scheduler for execution.

当一个 SM 被赋予一个或多个线程块时，它将每个线程块中的线程以 warp 为单位进行分组，并由调度器来执行分组后的 warp。

---

>The way a block is partitioned into warps is always the same; each warp contains threads of consecutive, increasing thread IDs with the first warp containing thread 0. 

线程块被分隔成 warp 的方式总是相同的；每个 warp 包含线程ID连续递增的线程，且第一个 warp 包含线程0。

>Thread Hierarchy describes how thread IDs relate to thread indices in the block.

[link]() [2.2]()节描述了线程块中的线程 ID 的计算方法。


--- 

>A warp executes one common instruction at a time, so full efficiency is realized when all 32 threads of a warp agree on their execution path. 

一个 warp 同时执行一条共同的指令，所以如果 warp 内所有的32个线程在同一条路径上执行时会达到最高效率率。

>If threads of a warp diverge via a data-dependent conditional branch, the warp executes each branch path taken, disabling threads that are not on that path. 

如果由于数据依赖条件分支导致 warp 分岔，warp 会顺序执行每个分支路径，而禁用不在此路径上的线程，直到所有路径完成，线程重新汇合到同一执行路径。

>Branch divergence occurs only within a warp; different warps execute independently regardless of whether they are executing common or disjoint code paths.

线程分支执行只会在同一线程束内发生，不同的 warp 总是独立执行的，不论它们执行的路径是否相同。

---

>The SIMT architecture is akin to SIMD (Single Instruction, Multiple Data) vector organizations in that a single instruction controls multiple processing elements. 

在使用单指令控制多元素上，SIMT 架构类似于 SIMD （Single Instruction,Multiple Data）向量架构。

>A key difference is that SIMD vector organizations expose the SIMD width to the software, whereas SIMT instructions specify the execution and branching behavior of a single thread. 

一项主要差别在于 SIMD 向量架构会向应用暴露 SIMD 宽度，而 SIMT 指定单线程的执行和分支行为。

>In contrast with SIMD vector machines, SIMT enables programmers to write thread-level parallel code for independent, scalar threads, as well as data-parallel code for coordinated threads. 

与 SIMD 向量架构相反，SIMT 允许程序员为独立标量线程编写线程级并行代码，也允许他们为协作线程编写数据级并行代码。

>For the purposes of correctness, the programmer can essentially ignore the SIMT behavior; however, substantial performance improvements can be realized by taking care that the code seldom requires threads in a warp to diverge. 

为了正确性，程序员可忽略 SIMT 行为；只要尽量减少 warp 内线程的分支执行就可以使得代码性能得到显著提升。

>In practice, this is analogous to the role of cache lines in traditional code: Cache line size can be safely ignored when designing for correctness but must be considered in the code structure when designing for peak performance. 

实际上，这类似于传统代码中缓存线的角色：以正确性为目标进行设计时，可忽略缓存线尺寸，但如果以峰值性能为目标进行设计，就必须考虑代码结构。

>Vector architectures, on the other hand, require the software to coalesce loads into vectors and manage divergence manually.

另外，SIMD 向量架构要求开发者将数据凑成合适的向量长度，并手动管理分支。 

---

>Prior to Volta, warps used a single program counter shared amongst all 32 threads in the warp together with an active mask specifying the active threads of the warp. 

Volta 架构之前的所有架构，warp 使用32个线程共享一个活动掩码的单个程序计数器来标定 warp 中的活动线程。

>As a result, threads from the same warp in divergent regions or different states of execution cannot signal each other or exchange data, and algorithms requiring fine-grained sharing of data guarded by locks or mutexes can easily lead to deadlock, depending on which warp the contending threads come from.

这不仅会导致同一 warp 内属于发散区域或不同执行状态的线程之间不能相互通信及交换数据，还会导致需要由锁或互斥锁保护的细粒度共享数据的算法很容易死锁，这取决于竞争线程来自于哪个 warp。

---

>Starting with the Volta architecture, Independent Thread Scheduling allows full concurrency between threads, regardless of warp. 

从Volta架构开始，新的 Independent Thread Scheduling 机制使得不同 warp 之间的线程可以并发地执行。

>With Independent Thread Scheduling, the GPU maintains execution state per thread, including a program counter and call stack, and can yield execution at a per-thread granularity, either to make better use of execution resources or to allow one thread to wait for data to be produced by another.

该机制下，在线程调度时，GPU 会维护每个线程的执行状态，包括程序计数器和调用堆栈，这样便能进行单线程的调度，从而更好地利用执行资源或允许一个线程等待另一个线程产生数据。

>A schedule optimizer determines how to group active threads from the same warp together into SIMT units. This retains the high throughput of SIMT execution as in prior NVIDIA GPUs, but with much more flexibility: threads can now diverge and reconverge at sub-warp granularity.

此外，有相应的计划优化程序来确定如何将来自同一个 warp 的活动线程分组为 SIMT 单元。与以前的架构一样，Independent Thread Scheduling 不仅保留了 SIMT 执行的高吞吐量，同时还具有更多的灵活性：warp 可以以线程为单位执行线程发射和重新对齐。 

---

>Independent Thread Scheduling can lead to a rather different set of threads participating in the executed code than intended if the developer made assumptions about warp-synchronicity of previous hardware architectures. 

如果开发人员对 Volta 之前的硬件架构做出 warp-synchronicity 的假设，Independent Thread Scheduling 可能导致他们看到参与执行代码的是一组与事先意料不同的线程。

>In particular, any warp-synchronous code (such as synchronization-free, intra-warp reductions) should be revisited to ensure compatibility with Volta and beyond. See Compute Capability 7.x for further details.

事实上，对现有的任何 warp 同步代码（synchronization-free, intra-warp reductions）都应重新考虑，以确保与 Volta 及以后的架构相互兼容，详细内容参考[link]() [计算能力7.x]()。

---

## Notes（注）

>The threads of a warp that are participating in the current instruction are called the active threads, whereas threads not on the current instruction are inactive (disabled).

参与当前指令的 warp 的线程称为活动线程，而不在当前指令的线程处于非活动状态（禁用）。

>Threads can be inactive for a variety of reasons including having exited earlier than other threads of their warp, having taken a different branch path than the branch path currently executed by the warp, or being the last threads of a block whose number of threads is not a multiple of the warp size.

线程处于非活动状态的原因有很多，如退出的时间早于其它 warp 线程，采用与 warp 当前执行的分支路径不同的分支路径，或者位于线程数量不是 warp 大小的倍数的线程块内。 

---

>If a non-atomic instruction executed by a warp writes to the same location in global or shared memory for more than one of the threads of the warp, the number of serialized writes that occur to that location varies depending on the compute capability of the device (see Compute Capability 3.x, Compute Capability 5.x, Compute Capability 6.x, and Compute Capability 7.x), and which thread performs the final write is undefined.

如果一个 warp 对多个 warp 线程的全局或共享内存中的同一位置进行非原子指令操作，对该位置的序列化写入次数取决于设备的计算能力（详细内容请参阅[link]() [计算能力3.x]()、[计算能力5.x]()、[计算能力6.x]() 和 [计算能力7.x]()，并且哪个线程执行最后的写入是未定义的。

---

>If an atomic instruction executed by a warp reads, modifies, and writes to the same location in global memory for more than one of the threads of the warp, each read/modify/write to that location occurs and they are all serialized, but the order in which they occur is undefined.

如果一个 warp 对多个 warp 线程的全局或共享内存中的同一位置进行原子指令操作，则每个读取/修改/写入操作都会发生，并且它们都是序列化的，但是它们发生的顺序是未定义的。

---

## 4.2 Hardware Multithreading
 
4.2 硬件多线程

>The execution context (program counters, registers, etc.) for each warp processed by a multiprocessor is maintained on-chip during the entire lifetime of the warp. 

SM 调度 warp 时为每个 warp 配置的执行上下文（程序计数器，寄存器等）在 warp 生命周期内都保存在芯片上。

>Therefore, switching from one execution context to another has no cost, and at every instruction issue time, a warp scheduler selects a warp that has threads ready to execute its next instruction (the active threads of the warp) and issues the instruction to those threads.

因此，在各个执行上下文间进行切换是没有代价的，而且每个指令执行时，warp 调度器选择一个具有准备好执行其下一个指令的线程（warp的活动线程）的 warp，然后将指令分配给那些线程。  

---

>In particular, each multiprocessor has a set of 32-bit registers that are partitioned among the warps, and a parallel data cache or shared memory that is partitioned among the thread blocks.

每个 SM 都有一组32位寄存器,并将它们划分给 warps， 此外，每个 SM 还会将其上拥有的 parallel data cache 和 shared memory 划分给线程块。  

---

>The number of blocks and warps that can reside and be processed together on the multiprocessor for a given kernel depends on the amount of registers and shared memory used by the kernel and the amount of registers and shared memory available on the multiprocessor. 

对于一个给定启动配置的 kernel，每个 SM 上可以同时驻留的线程块数量和 warp 数量取决于 SM 拥有的寄存器、共享内存数量和该 kernel 需要的寄存器、共享内存数量。

>There are also a maximum number of resident blocks and a maximum number of resident warps per multiprocessor. 

且不同的计算设备都有其允许的最大线程块、warp、寄存器及共享内存值。

>These limits as well the amount of registers and shared memory available on the multiprocessor are a function of the compute capability of the device and are given in Appendix Compute Capabilities. 

这些限制值与设备的计算能力符合一定的函数关系，附录[link]()[Comute Capabilities]()对此给出了详细描述。

>If there are not enough registers or shared memory available per multiprocessor to process at least one block, the kernel will fail to launch.

当每个 SM 没有足够的寄存器和共享内存来处理一个线程块时，kernel 就会启动失败。一个线程块的 warps 数量可以按下式计算  

>The total number of warps in a block is as follows:

一个线程块的 warps 数量可以按下式计算  
$$ ceil(\frac{ T }{ W_{size} }, 1)$$
* $ T $ 是每个线程块的线程总数  
* $W_{size}$是 warp 大小，取值32  
* $ ceil(x,y)$表示 $ \frac{ x }{ y }$ 向上取整的值。   
 
 
>The total number of registers and total amount of shared memory allocated for a block are documented in the CUDA Occupancy Calculator provided in the CUDA Toolkit.

随 CUDA Toolkit 发布的文档 [CUDA Occupancy Calculator](https://developer.download.nvidia.com/compute/cuda/CUDA_Occupancy_calculator.xls) 给出了不同计算设备为每个线程块提供的寄存器及共享内存数量。

