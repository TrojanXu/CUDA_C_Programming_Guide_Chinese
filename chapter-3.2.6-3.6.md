 ---
 ### 3.2.6.Multi-Device System
 ### 3.2.6 多设备系统
 
 #### 3.2.6.1 Device Enumeration
 #### 3.2.6.1 设备枚举
 
 > A host system can have multiple devices. The following code sample shows how to enumerate these devices, query their properties, and determine the number of CUDA-enabled devices.
 
 一个 host 系统可以有多个 device。以下的代码示例展示了如何枚举这些 device、查询它们的属性，以及确定 CUDA-enabled 的 device 个数。
 
 ``` cuda
int deviceCount;
cudaGetDeviceCount(&deviceCount);
int device;
for (device = 0; device < deviceCount; ++device) {
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, device);
    printf("Device %d has compute capability %d.%d.\n",
           device, deviceProp.major, deviceProp.minor);
}
 ```
 
 #### 3.2.6.2. Device Selection
 #### 3.2.6.2 设备选择
 
 > A host thread can set the device it operates on at any time by calling cudaSetDevice(). Device memory allocations and kernel launches are made on the currently set device; streams and events are created in association with the currently set device. If no call to cudaSetDevice() is made, the current device is device 0.  

一个 host 线程可以在任何时刻通过调用 *cudaSetDevice()* 来设置起作用的 device。 device 内存的分配和 kernel 的发起都是在当前设置的 device 上。stream 和 events 的创建也与当前设置的 device 有关。如果没有调用 *cudaSetDevice()*，那么默认的当前 device 就是 0。

> The following code sample illustrates how setting the current device affects memory allocation and kernel execution.

以下代码示例展示了设置当前的 device 对内存的分配和 kernel 的执行会产生怎样的影响。

``` cuda
size_t size = 1024 * sizeof(float);
cudaSetDevice(0);            // 设置 device 0 为当前 device
float* p0;
cudaMalloc(&p0, size);       // 在 device 0 上分配内存
MyKernel<<<1000, 128>>>(p0); // 在 device 0 上启动 kernel
cudaSetDevice(1);            // 设置 device 1 为当前 device
float* p1;
cudaMalloc(&p1, size);       // 在 device 1 上分配内存
MyKernel<<<1000, 128>>>(p1); // 在 device 1 上启动 kernel
```

#### 3.2.6.3. Stream and Event Behavior
#### 3.2.6.3 流和事件的行为

> A kernel launch will fail if it is issued to a stream that is not associated to the current device as illustrated in the following code sample.

如果 kernel 被发给了一个和当前 device 不相干的 stream ，那么这个 kernel 就会启动失败，如以下代码所示。

``` cuda
cudaSetDevice(0);               // 设置 device 0 为当前 device
cudaStream_t s0;
cudaStreamCreate(&s0);          // 在 device 0 上创建 stream s0
MyKernel<<<100, 64, 0, s0>>>(); // 在 device 0 上启动 kernel，并发送给 s0
cudaSetDevice(1);               // 设置 device 1 为当前 device
cudaStream_t s1;
cudaStreamCreate(&s1);          // 在 device 1 上创建 stream s1
MyKernel<<<100, 64, 0, s1>>>(); // 在 device 1 上启动 kernel 并发送给 s1

// 这个 kernel 启动将会失败:
MyKernel<<<100, 64, 0, s0>>>(); // 在 device 1 上启动 kernel 并发送给 s0
```

> A memory copy will succeed even if it is issued to a stream that is not associated to the current device.

即使将内存拷贝发送到一个和当前 device 并不相干的 stream 中，它仍然会执行成功。

> cudaEventRecord() will fail if the input event and input stream are associated to different devices.

如果 input event 和 input stream 关联的是不同的 device，那么 *cudaEventRecord()* 将会执行失败。

> cudaEventElapsedTime() will fail if the two input events are associated to different devices.

如果两个 input event 关联的是不同的 device，那么 *cudaEventElapsedTime()* 将会执行失败。

> cudaEventSynchronize() and cudaEventQuery() will succeed even if the input event is associated to a device that is different from the current device.

即使 input event 关联的 device 和当前 device 不同，*cudaEventSynchronize()* 和 *cudaEventQuery()* 仍然会执行成功。

> cudaStreamWaitEvent() will succeed even if the input stream and input event are associated to different devices. cudaStreamWaitEvent() can therefore be used to synchronize multiple devices with each other.

即使 input stream 和 input event 关联的是不同的 device，*cudaStreamWaitEvent()* 仍然会执行成功。因此 *cudaStreamWaitEvent()*  可以用于多个 device 之间的同步

> Each device has its own default stream (see Default Stream), so commands issued to the default stream of a device may execute out of order or concurrently with respect to commands issued to the default stream of any other device.

每个 device 都有其默认 stream [link](http://note.youdao.com/) （参阅 “Default Stream” 章节），因此被发送到某一 device 的默认 stream 的命令相对于那些被发送到其他 device 的默认 stream 的命令可能是乱序执行的，或者是并发执行的。

#### 3.2.6.4. Peer-to-Peer Memory Access
#### 3.2.6.4 端到端的内存访问

> When the application is run as a 64-bit process, devices of compute capability 2.0 and higher from the Tesla series may address each other's memory (i.e., a kernel executing on one device can dereference a pointer to the memory of the other device). This peer-to-peer memory access feature is supported between two devices if cudaDeviceCanAccessPeer() returns true for these two devices.

当应用程序以 64 位进程的形式运行时，计算能力为 2.0 及高于 Tesla 系列的设备，就可以相互进行内存寻址操作（即，在一个 device 上执行的 kernel 可以解引用(*) 其他 device 内存上的指针）。如果 *cudaDeviceCanAccessPeer()* 返回为真，那么就是说明两个 device 之间支持 peer-to-peer 的内存访问特性。

> Peer-to-peer memory access must be enabled between two devices by calling cudaDeviceEnablePeerAccess() as illustrated in the following code sample. Each device can support a system-wide maximum of eight peer connections.

两个 device 之间的 peer-to-peer 特性必须通过调用 *cudaDeviceEnablePeerAccess()* 来启用，如以下代码所示。在系统范围内，每个 device 最多可以支持 8 个 peer 连接。

> A unified address space is used for both devices (see Unified Virtual Address Space), so the same pointer can be used to address memory from both devices as shown in the code sample below.

两个 device 上使用统一地址空间 [link](http://note.youdao.com/) （参阅 “Unified Virtual Address Space” 章节），所以可通过同一个指针从两个 device 上进行内存寻址，如以下代码所示。

``` cuda
cudaSetDevice(0);                   // 设置 device 0 为当前 device
float* p0;
size_t size = 1024 * sizeof(float);
cudaMalloc(&p0, size);              // 在 device 0 上分配内存
MyKernel<<<1000, 128>>>(p0);        // 在 device 0 上启动 kernel
cudaSetDevice(1);                   // 设置 device 1 为当前 device
cudaDeviceEnablePeerAccess(0, 0);   // 启动与 device 0 的 peer-to-peer 访问
                                    
// 在 device 1 上启动 kernel
// 该 kernel 可以访问 device 0 上的内存地址 p0 
MyKernel<<<1000, 128>>>(p0);
```

#### 3.2.6.5. Peer-to-Peer Memory Copy
#### 3.2.6.5 端到端的内存拷贝

> Memory copies can be performed between the memories of two different devices.

在两个不同 device 的内存之间可以进行内存拷贝。

> When a unified address space is used for both devices (see Unified Virtual Address Space), this is done using the regular memory copy functions mentioned in Device Memory.

当两个 device 使用了统一地址空间（参阅 “Unified Virtual Address Space”），通过在“设备内存”中提到的常规内存拷贝方法就可以完成（两个设备之间的内存拷贝）。

> Otherwise, this is done using cudaMemcpyPeer(), cudaMemcpyPeerAsync(), cudaMemcpy3DPeer(), or cudaMemcpy3DPeerAsync() as illustrated in the following code sample.

否则，就需要通过 *cudaMemcpyPeer()*, *cudaMemcpyPeerAsync()*, *cudaMemcpy3DPeer()*, 或 *cudaMemcpy3DPeerAsync()* 方法来完成，如以下代码所示。

``` cuda
cudaSetDevice(0);                   // 将 device 0 设置为当前设备
float* p0;
size_t size = 1024 * sizeof(float);
cudaMalloc(&p0, size);              // 在 device 0 上分配内存
cudaSetDevice(1);                   // 将 device 1 设置为当前设备
float* p1;
cudaMalloc(&p1, size);              // 在 device 1 上分配内存
cudaSetDevice(0);                   // 设置 device 0 为当前设备
MyKernel<<<1000, 128>>>(p0);        // 在 device 0 上启动 kernel
cudaSetDevice(1);                   // 将 device 1 设置为当前设备
cudaMemcpyPeer(p1, 1, p0, 0, size); // 从 p0 拷贝到 p1
MyKernel<<<1000, 128>>>(p1);        // 在 device 1 上启动 kernel
```
> A copy (in the implicit NULL stream) between the memories of two different devices:

在两个不同 device 的内存间的拷贝（在隐含的 NULL Stream 中）:

> does not start until all commands previously issued to either device have completed and
runs to completion before any commands (see Asynchronous Concurrent Execution) issued after the copy to either device can start.
Consistent with the normal behavior of streams, an asynchronous copy between the memories of two devices may overlap with copies or kernels in another stream.

当之前发送给每个 device 的命令都执行完，才会开始执行内存拷贝，直到拷贝完成，其他晚于拷贝命令的被发送到每个 device 上的命令才能开始执行。 [link](http://note.youdao.com/) （参阅 “异步并行执行” 章节）

> Note that if peer-to-peer access is enabled between two devices via cudaDeviceEnablePeerAccess() as described in Peer-to-Peer Memory Access, peer-to-peer memory copy between these two devices no longer needs to be staged through the host and is therefore faster.

要注意的是，如果两个 device 间的 peer-to-peer 访问被开启了 [link](http://note.youdao.com/) （如“端到端内存访问”章节所述，通过调用 cudaDeviceEnablePeerAccess()），那么两个 device 间的 peer-to-peer 的内存拷贝就不需要通过 host 进行分阶段，因而会更快一些。

---

### 3.2.7. Unified Virtual Address Space
### 3.2.7. 统一虚拟地址空间
> When the application is run as a 64-bit process, a single address space is used for the host and all the devices of compute capability 2.0 and higher. All host memory allocations made via CUDA API calls and all device memory allocations on supported devices are within this virtual address range. As a consequence:

当应用程序以 64 位进程的形式运行，host 和所有计算能力在 2.0 及以上的 device 使用的是一个单地址空间。所有通过 CUDA API 调用的 host 内存分配和所有支持的 device 上的 device 内存分配都是在这个虚拟地址的范围内。所以：

> The location of any memory on the host allocated through CUDA, or on any of the devices which use the unified address space, can be determined from the value of the pointer using cudaPointerGetAttributes().
When copying to or from the memory of any device which uses the unified address space, the cudaMemcpyKind parameter of cudaMemcpy*() can be set to cudaMemcpyDefault to determine locations from the pointers. This also works for host pointers not allocated through CUDA, as long as the current device uses unified addressing.
Allocations via cudaHostAlloc() are automatically portable (see Portable Memory) across all the devices for which the unified address space is used, and pointers returned by cudaHostAlloc() can be used directly from within kernels running on these devices (i.e., there is no need to obtain a device pointer via cudaHostGetDevicePointer() as described in Mapped Memory.

通过 CUDA 分配的 host 任意内存的位置，或使用统一地址空间的 device 任意内存的位置，都可以使用 *cudaPointerGetAttributes()* 获得指针的值来确定。
当使用统一地址空间的任意 device 内存进行拷入或拷出时，*cudaMemcpy\*()* 的“cudaMemcpy”样式的参数可以设置为 cudaMemcpyDefault，而由指针的值来确定位置。 这同样适用于不是通过 CUDA 分配的 host 指针，只要当前 device 使用统一寻址即可。
通过 *cudaHostAlloc()* 分配的内存在所有使用统一地址空间的 device 上是自动 portable 的 [link]() （参阅 “Portable Memory” 章节），并且 *cudaHostAlloc()* 返回的指针可以直接在这些 device 的 kernel 内使用 [link]() （即，不必像 “Mapped Memory”章节所描述的那样，通过 *cudaHostGetDevicePointer()* 获得 device 的指针）

> Applications may query if the unified address space is used for a particular device by checking that the unifiedAddressing device property (see Device Enumeration) is equal to 1.

应用程序可以通过检查 *unifiedAddressing* 的 device 属性是否等于 1，来查询特定的 device 是否使用了统一地址空间 [link]() （参阅 “Device Enumeration” 章节）

---

### 3.2.8. Interprocess Communication
### 3.2.8. 跨进程通信

> Any device memory pointer or event handle created by a host thread can be directly referenced by any other thread within the same process. It is not valid outside this process however, and therefore cannot be directly referenced by threads belonging to a different process.

由某个 host 线程所创建的任意 device 内存指针或 事件句柄可以被相同进程里的其他线程直接引用。然而在该进程之外就是无效的，因此属于不同进程的线程无法直接引用。

> To share device memory pointers and events across processes, an application must use the Inter Process Communication API, which is described in detail in the reference manual. The IPC API is only supported for 64-bit processes on Linux and for devices of compute capability 2.0 and higher.

为了跨进程共享 device 内存指针和事件，应用程序必须使用跨进程通信 API（在 reference manual 中有详细描述）。IPC API 仅支持Linux系统的 64 位进程，且 device 的计算能力在 2.0 及以上。

> Using this API, an application can get the IPC handle for a given device memory pointer using cudaIpcGetMemHandle(), pass it to another process using standard IPC mechanisms (e.g., interprocess shared memory or files), and use cudaIpcOpenMemHandle() to retrieve a device pointer from the IPC handle that is a valid pointer within this other process. Event handles can be shared using similar entry points.

通过该 API，一个应用程序可以通过 *cudaIpcGetMemHandle()* 获得一个用于给定的 device 内存指针的 IPC 句柄，使用标准的 IPC 机制，将它传给其他进程（例如，跨进程的共享内存或文件），然后通过 *cudaIpcOpenMemHandle()* 从 IPC 句柄中取得一个 device 指针（它是在其他进程内有效的指针）。时间句柄可以通过类似的切入点进行共享。

> An example of using the IPC API is where a single master process generates a batch of input data, making the data available to multiple slave processes without requiring regeneration or copying.

一个使用 IPC API 的例子是：单个主进程生成了一批输入数据，无需通过重新生成或拷贝，就使得这些数据对多个子进程可用。

---

### 3.2.9. Error Checking
### 3.2.9. 错误检查

> All runtime functions return an error code, but for an asynchronous function (see Asynchronous Concurrent Execution), this error code cannot possibly report any of the asynchronous errors that could occur on the device since the function returns before the device has completed the task; the error code only reports errors that occur on the host prior to executing the task, typically related to parameter validation; if an asynchronous error occurs, it will be reported by some subsequent unrelated runtime function call.

所有的运行时函数都会返回一个错误码，对于异步的函数 [link](http://note.youdao.com/) （参阅 “Asynchronous Concurrent Execution” 章节），它的错误码不可能报告所有异步的错误，因为它发生在函数返回时，而此时 device 尚未执行完任务；这个错误代码只报告在执行任务之前发生在 host 上的错误,通常与参数验证相关；如果发生了一个异步错误，它将由一些后续不相关的运行时函数调用进行报告。

> The only way to check for asynchronous errors just after some asynchronous function call is therefore to synchronize just after the call by calling cudaDeviceSynchronize() (or by using any other synchronization mechanisms described in Asynchronous Concurrent Execution) and checking the error code returned by cudaDeviceSynchronize().

因此，在一些异步函数调用之后，检查异步错误的唯一方法是通过在调用异步函数之后调用 *cudaDeviceSynchronize()* 进行同步（或者使用其他的同步机制，如 [link](http://note.youdao.com/)  “Asynchronous Concurrent Execution” 章节所述）并检查 *cudaDeviceSynchronize()* 返回的错误码

> The runtime maintains an error variable for each host thread that is initialized to cudaSuccess and is overwritten by the error code every time an error occurs (be it a parameter validation error or an asynchronous error). cudaPeekAtLastError() returns this variable. cudaGetLastError() returns this variable and resets it to cudaSuccess.

运行时为每一个 host 线程 维护了一个错误变量，初始化 cudaSuccess，每次发生错误后（它可能是一个参数验证错误或是一个异步错误）错误代码都会覆盖该变量。 *cudaPeekAtLastError()* 返回该变量。*cudaGetLastError()* 返回这个变量并重置它为 cudaSuccess。

> Kernel launches do not return any error code, so cudaPeekAtLastError() or cudaGetLastError() must be called just after the kernel launch to retrieve any pre-launch errors. To ensure that any error returned by cudaPeekAtLastError() or cudaGetLastError() does not originate from calls prior to the kernel launch, one has to make sure that the runtime error variable is set to cudaSuccess just before the kernel launch, for example, by calling cudaGetLastError() just before the kernel launch. Kernel launches are asynchronous, so to check for asynchronous errors, the application must synchronize in-between the kernel launch and the call to cudaPeekAtLastError() or cudaGetLastError().

kernel 的启动不会返回任务的错误码，所以 *cudaPeekAtLastError()* 或 *cudaGetLastError()* 必须在 kernel 启动之后调用，以取得所有启动前的错误。为了确保 *cudaPeekAtLastError()* 或 *cudaGetLastError()* 返回的任意错误并不是源于 kernel 启动之前的错误，还必须确保运行时错误变量在 kernel 启动之前被设置为 cudaSuccess，例如，在 kernel 启动之前调用 *cudaGetLastError()* 。因为 kernel 的启动是异步的，所以为了检查异步的错误，应用程序必须在 kernel 启动和 *cudaPeekAtLastError()* 或 *cudaGetLastError()* 调用之间进行同步。

> Note that cudaErrorNotReady that may be returned by cudaStreamQuery() and cudaEventQuery() is not considered an error and is therefore not reported by cudaPeekAtLastError() or cudaGetLastError().

请注意， *cudaStreamQuery()* 和 *cudaEventQuery()* 可能会返回 *cudaErrorNotReady* ，但它不会被认为是一个错误，因此不会被 *cudaPeekAtLastError()* 或 *cudaGetLastError()* 报告。

---

### 3.2.10. Call Stack
### 3.2.10. 调用堆栈

> On devices of compute capability 2.x and higher, the size of the call stack can be queried using cudaDeviceGetLimit() and set using cudaDeviceSetLimit().

在计算能力2.x和更高的 device 上，可以使用 *cudaDeviceGetLimit()* 查询调用堆栈的大小，并使用 *cudadeviceesetlimit()* 设置其大小。

> When the call stack overflows, the kernel call fails with a stack overflow error if the application is run via a CUDA debugger (cuda-gdb, Nsight) or an unspecified launch error, otherwise.

当调用堆栈溢出时，如果通过CUDA调试器(CUDA -gdb, Nsight)运行程序，kernrl 调用将会失败并带着堆栈溢出的错误；反之则是未指定的启动错误。

---

### 3.2.11. Texture and Surface Memory
### 3.2.11. 纹理和表面内存

> CUDA supports a subset of the texturing hardware that the GPU uses for graphics to access texture and surface memory. Reading data from texture or surface memory instead of global memory can have several performance benefits as described in Device Memory Accesses.

CUDA 支持一个 texturing 硬件的子集，GPU 将它用于图形访问 texture 和 surface 内存。从 texture 或 surface 内存读取数据，而不是从 global memory，可以有一些性能的提升 [link](http://note.youdao.com/) （如 “Device Memory Accesses” 章节所述）

> There are two different APIs to access texture and surface memory:

有两种不同的 API 来访问 texture 和 surface 内存：

> The texture reference API that is supported on all devices,
The texture object API that is only supported on devices of compute capability 3.x.
The texture reference API has limitations that the texture object API does not have. They are mentioned in Texture Reference API.

texture reference API 是全 device 支持的，texture object API 仅有计算能力 3.x 的 device 才支持。
不过 texture reference API 有 texture object API 所没有的一些限制。这些将会在 [link]() "Texture Reference API" 章节提到。

#### 3.2.11.1. Texture Memory
#### 3.2.11.1 texture 内存

> Texture memory is read from kernels using the device functions described in Texture Functions. The process of reading a texture calling one of these functions is called a texture fetch. Each texture fetch specifies a parameter called a texture object for the texture object API or a texture reference for the texture reference API. 

从 kernel 中读取 texture 内存使用的是 [link]() “Texture Functions 章节” 描述的 device 方法。读取一张 texture 并调用其中一个函数的过程称为 texture fetch 。每一个 texture fetch 都为 texture object API 指定一个名为 texture object 的参数，或者为 texture reference API 指定一个名为 texture reference 的 参数。

> The texture object or the texture reference specifies:

texture object 或 texture reference 的详细说明：

>- The texture, which is the piece of texture memory that is fetched. Texture objects are created at runtime and the texture is specified when creating the texture object as described in Texture Object API. Texture references are created at compile time and the texture is specified at runtime by bounding the texture reference to the texture through runtime functions as described in Texture Reference API; several distinct texture references might be bound to the same texture or to textures that overlap in memory. A texture can be any region of linear memory or a CUDA array (described in CUDA Arrays).

- texture ，是从 texture 内存中 fetch 出的一块。texture object 在运行时创建 texture object，当如 [link]()  “Texture Object API 章节” 所述创建了 texture object，texture 就被指定了。texture reference 是在编译期创建的，通过使用 [link]()  “Texture Reference API 章节” 所述的运行时方法绑定 texture reference 到 texture，以便在运行时指定 texture。一些不同的 texture refernce 可能被绑定到相同的 texture 或在内存中有重叠的 texture 。texture 可以是线性内存或者 CUDA 数组中的任意区域（如 [link]() “CUDA Arrays 章节” 所述）

>- Its dimensionality that specifies whether the texture is addressed as a one dimensional array using one texture coordinate, a two-dimensional array using two texture coordinates, or a three-dimensional array using three texture coordinates. Elements of the array are called texels, short for texture elements. The texture width, height, and depth refer to the size of the array in each dimension. Table 14 lists the maximum texture width, height, and depth depending on the compute capability of the device.

- 纹理的维度指定了它的寻址方式，比如一维数组使用一维纹理坐标，二维数组使用二维纹理坐标，三维数组使用三维纹理坐标。每个数组的元素被称作 texels，texture elements 的简称。纹理的宽度、高度、深度都和数组的每一维的大小有关。[link]() 表 14 展示了依赖于 device 计算能力的最大纹理宽度、高度、深度。

>- The type of a texel, which is restricted to the basic integer and single-precision floating-point types and any of the 1-, 2-, and 4-component vector types defined in char, short, int, long, longlong, float, double that are derived from the basic integer and single-precision floating-point types.

- texel 的类型只能是基本的整型或者单精度浮点型，任何的以 [link]() char, short, int, long, longlong, float, double定义的 1-，2-，和4-成员 vector 类型都是派生自基础的整型和单精度浮点型。

>- The read mode, which is equal to cudaReadModeNormalizedFloat or cudaReadModeElementType. If it is cudaReadModeNormalizedFloat and the type of the texel is a 16-bit or 8-bit integer type, the value returned by the texture fetch is actually returned as floating-point type and the full range of the integer type is mapped to [0.0, 1.0] for unsigned integer type and [-1.0, 1.0] for signed integer type; for example, an unsigned 8-bit texture element with the value 0xff reads as 1. If it is cudaReadModeElementType, no conversion is performed.

- 读取模式分为 *cudaReadModeNormalizedFloat* 或 *cudaReadModeElementType* 两种。如果读取模式为 *cudaReadModeNormalizedFloat* 且 texel 的类型是是 16 位或 8 位的整型，那么由 texture fetch 返回的值实际上是浮点型， 无符号整型将被映射到 [0.0, 1.0] 之间，而有符号整型则被映射到 [-1.0, 1.0] 之间。例如，一个无符号 8 位的 texture element 的值为 0xff，那么它读取后的值为 1.0。如果读取模式设置为 *cudaReadModeElementType* ，则不会进行这种转换。

>- Whether texture coordinates are normalized or not. By default, textures are referenced (by the functions of Texture Functions) using floating-point coordinates in the range [0, N-1] where N is the size of the texture in the dimension corresponding to the coordinate. For example, a texture that is 64x32 in size will be referenced with coordinates in the range [0, 63] and [0, 31] for the x and y dimensions, respectively. Normalized texture coordinates cause the coordinates to be specified in the range [0.0, 1.0-1/N] instead of [0, N-1], so the same 64x32 texture would be addressed by normalized coordinates in the range [0, 1-1/N] in both the x and y dimensions. Normalized texture coordinates are a natural fit to some applications' requirements, if it is preferable for the texture coordinates to be independent of the texture size.

- texture 坐标是否被规范化。默认情况下，texture 的引用（通过 [link]() “Texture Functions” 章节的方法）使用的是 [0, N-1] 范围的浮点坐标，其中 N 表示对应于坐标的某一维度上的 texture 的大小。例如，64x32 大小的 texture 将分别以 x 维度在 [0, 63]、y 维度在 [0, 31] 的范围内的坐标进行引用。规范化的 texture 坐标将原本的 [0, N-1] 替换为 [0.0, 1.0 - 1/N] 的范围，因此，同样 64x32 大小的 texture 将在 x、y 两个维度上都以 [0, 1 - 1/N] 的规范化坐标进行寻址。规范化的 texture 坐标更适用于某些需要与 texture 大小无关的应用程序。

>- The addressing mode. It is valid to call the device functions of Section B.8 with coordinates that are out of range. The addressing mode defines what happens in that case. The default addressing mode is to clamp the coordinates to the valid range: [0, N) for non-normalized coordinates and [0.0, 1.0) for normalized coordinates. If the border mode is specified instead, texture fetches with out-of-range texture coordinates return zero. For normalized coordinates, the wrap mode and the mirror mode are also available. When using the wrap mode, each coordinate x is converted to frac(x)=x - floor(x) where floor(x) is the largest integer not greater than x. When using the mirror mode, each coordinate x is converted to frac(x) if floor(x) is even and 1-frac(x) if floor(x) is odd. The addressing mode is specified as an array of size three whose first, second, and third elements specify the addressing mode for the first, second, and third texture coordinates, respectively; the addressing mode are cudaAddressModeBorder, cudaAddressModeClamp, cudaAddressModeWrap, and cudaAddressModeMirror; cudaAddressModeWrap and cudaAddressModeMirror are only supported for normalized texture coordinates

- 寻址模式。以超过坐标范围的方式调用 Section B.8 device 方法是合法的。寻址模式定义了这种情况下会发生什么。默认的寻址模式是截断坐标到一个合法的范围：对于非规范化的坐标是 [0, N)，对于规范化的坐标 [0.0, 1.0)。如果指定的是 border 模式，那么超出 texture 坐标范围的 texture fetch 返回值将是 0。对于规范化的坐标，wrap 模式和 mirror 模式同样适用。当使用的是 wrap 模式的话，每个坐标的 x 被转换为 frac(x)=x - floor(x)，其中 floor(x) 表示不大于 x 的最大整型。当适用的是 mirror 模式的话，每个坐标的 x 在 floor(x) 为偶数时为转换为 frac(x)，在 floor(x) 为奇数时为转换为 1-frac(x)。寻址模式被指定为一个大小为3的数组，其第一、第二和第三个元素分别为第一、第二和第三个纹理坐标指定寻址模式。寻址模式是 *cudaAddressModeBorder*、*cudaAddressModeClamp*、*cudaAddressModeWrap* 和 *cudaAddressModeMirror*；其中 *cudaAddressModeWrap* 和 *cudaAddressModeMirror* 只支持规范化纹理坐标

>- The filtering mode which specifies how the value returned when fetching the texture is computed based on the input texture coordinates. Linear texture filtering may be done only for textures that are configured to return floating-point data. It performs low-precision interpolation between neighboring texels. When enabled, the texels surrounding a texture fetch location are read and the return value of the texture fetch is interpolated based on where the texture coordinates fell between the texels. Simple linear interpolation is performed for one-dimensional textures, bilinear interpolation for two-dimensional textures, and trilinear interpolation for three-dimensional textures. Texture Fetching gives more details on texture fetching. The filtering mode is equal to cudaFilterModePoint or cudaFilterModeLinear. If it is cudaFilterModePoint, the returned value is the texel whose texture coordinates are the closest to the input texture coordinates. If it is cudaFilterModeLinear, the returned value is the linear interpolation of the two (for a one-dimensional texture), four (for a two dimensional texture), or eight (for a three dimensional texture) texels whose texture coordinates are the closest to the input texture coordinates. cudaFilterModeLinear is only valid for returned values of floating-point type.

- 过滤模式指定了当 texture 执行 fetch 时的返回值是如何根据输入 texture 的坐标进行计算的。线性 texture 过滤仅仅适用于被配置为返回值浮点型的 texture。它执行的是和邻域 texels 的低精度插值。当启用时，texture fetch 连同周围的 texels 也一起读取， texture fetch 的返回值是基于这些 texels 之间的 texture 坐标进行插值的。简单线性插值用于一维纹理，双线性插值用于二维纹理，三线性插值用于三维纹理。[link]() “Texture Fetching 章节” 给出了更多的 texture fetching 细节。过滤模式包括两种：*cudaFilterModePoint* 或 *cudaFilterModeLinear* 。如果是 *cudaFilterModePoint* ，返回的 是其纹理坐标最接近输入纹理坐标的 exel。如果是cudaFilterModeLinear，返回的是 2 个（对于一维纹理）、4 个（对于二维纹理）、或 8 个（对于一个三维纹理）其 texutre 坐标最接近输入 texture 坐标的 texels 的线性插值。*cudaFilterModeLinear* 只适用于浮点类型的返回值。

> Texture Object API introduces the texture object API.

[link]() Texture Object API 介绍了texture object API。

> Texture Reference API introduces the texture reference API.

[link]() Texture Reference API 介绍了texture reference API

> 16-Bit Floating-Point Textures explains how to deal with 16-bit floating-point textures.

[link]() 16-Bit Floating-Point Textures 解释了如何处理 16 位浮点型 texture。

> Textures can also be layered as described in Layered Textures.

texture 也可以被分层，如[link]() Layered Texture 所述

> Cubemap Textures and Cubemap Layered Textures describe a special type of texture, the cubemap texture.

[link]() Cubemap Textures 和[link]() Cubemap Layered Textures 描述了一种特殊类型的 texture，立方体贴图 texture。

> Texture Gather describes a special texture fetch, texture gather.

[link]() Texture Gather 描述了一种特殊的 texture fetch，texture 聚集。

##### 3.2.11.1.1. Texture Object API
##### 3.2.11.1.1. 纹理对象 API

> A texture object is created using cudaCreateTextureObject() from a resource description of type struct cudaResourceDesc, which specifies the texture, and from a texture description defined as such: 

texture object 是由一个结构体类型的资源描述 *cudaResourceDesc*  和一个 texture 描述结构体通过调用 *cudaCreateTextureObject()* 来创建的。*cudaResourceDesc* 指定了 texture。texture 描述结构体，如下所示

``` cuda
struct cudaTextureDesc
{
    enum cudaTextureAddressMode addressMode[3];
    enum cudaTextureFilterMode  filterMode;
    enum cudaTextureReadMode    readMode;
    int                         sRGB;
    int                         normalizedCoords;
    unsigned int                maxAnisotropy;
    enum cudaTextureFilterMode  mipmapFilterMode;
    float                       mipmapLevelBias;
    float                       minMipmapLevelClamp;
    float                       maxMipmapLevelClamp;
};
```
>- addressMode specifies the addressing mode; 

- *addressMode* 指定了寻址模式；

>- filterMode specifies the filter mode; 

- *filterMode* 指定了过滤模式；

>- readMode specifies the read mode; 

- *readMode* 指定了读取模式；

>- normalizedCoords specifies whether texture coordinates are normalized or not; 

- *normalizedCoords* 指定了 texture 坐标是否规范化；

>- See reference manual for sRGB, maxAnisotropy, mipmapFilterMode, mipmapLevelBias, minMipmapLevelClamp, and maxMipmapLevelClamp. 

- *sRGB*, *maxAnisotropy*, *mipmapFilterMode*, *mipmapLevelBias*, *minMipmapLevelClamp*, and *maxMipmapLevelClamp* 的概念清查看参考手册。

> The following code sample applies some simple transformation kernel to a texture. 

以下代码应用了一些 texture 上的简单变换 kernel

``` cuda
// 简单的变换 kernel
__global__ void transformKernel(float* output,
                                cudaTextureObject_t texObj,
                                int width, int height,
                                float theta) 
{
    //  计算规范化的 texture 坐标
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

    float u = x / (float)width;
    float v = y / (float)height;

    // 坐标变换
    u -= 0.5f;
    v -= 0.5f;
    float tu = u * cosf(theta) - v * sinf(theta) + 0.5f;
    float tv = v * cosf(theta) + u * sinf(theta) + 0.5f;

    // 从 texture 读取并写入 global memory
    output[y * width + x] = tex2D<float>(texObj, tu, tv);
}
```
``` cuda
// Host 端代码
int main()
{
    // 在 device 内存上分配 CUDA 数组
    cudaChannelFormatDesc channelDesc =
               cudaCreateChannelDesc(32, 0, 0, 0,
                                     cudaChannelFormatKindFloat);
    cudaArray* cuArray;
    cudaMallocArray(&cuArray, &channelDesc, width, height);

    // 拷贝 host 内存上 h_data 地址的一些数据到 device 内存
    cudaMemcpyToArray(cuArray, 0, 0, h_data, size,
                      cudaMemcpyHostToDevice);

    // 指定 texture
    struct cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = cuArray;

    // 指定 texture object 参数
    struct cudaTextureDesc texDesc;
    memset(&texDesc, 0, sizeof(texDesc));
    texDesc.addressMode[0]   = cudaAddressModeWrap;
    texDesc.addressMode[1]   = cudaAddressModeWrap;
    texDesc.filterMode       = cudaFilterModeLinear;
    texDesc.readMode         = cudaReadModeElementType;
    texDesc.normalizedCoords = 1;

    // 创建 texture object
    cudaTextureObject_t texObj = 0;
    cudaCreateTextureObject(&texObj, &resDesc, &texDesc, NULL);

    // 在 device 内存上分配变换结果的内存
    float* output;
    cudaMalloc(&output, width * height * sizeof(float));

    // 发起 kernel
    dim3 dimBlock(16, 16);
    dim3 dimGrid((width  + dimBlock.x - 1) / dimBlock.x,
                 (height + dimBlock.y - 1) / dimBlock.y);
    transformKernel<<<dimGrid, dimBlock>>>(output,
                                           texObj, width, height,
                                           angle);

    // 销毁 texture object
    cudaDestroyTextureObject(texObj);

    // 释放 device 内存
    cudaFreeArray(cuArray);
    cudaFree(output);

    return 0;
}
```

##### 3.2.11.1.2. Texture Reference API
##### 3.2.11.1.2 纹理引用 API

> Some of the attributes of a texture reference are immutable and must be known at compile time; they are specified when declaring the texture reference. A texture reference is declared at file scope as a variable of type texture: 

texture reference 的一些属性是不变的，并且必须是编译期可知的；它们在 texture reference 声明的时候就被指定了。一个 texture reference 是被声明在文件范围的一种 texture 变量。

``` cuda
texture<DataType, Type, ReadMode> texRef;
```
> where:
> •DataType specifies the type of the texel; 

•DataType 指定了 texel 的类型；

> •Type specifies the type of the texture reference and is equal to cudaTextureType1D, cudaTextureType2D, or cudaTextureType3D, for a one-dimensional, two-dimensional, or three-dimensional texture, respectively, or cudaTextureType1DLayered or cudaTextureType2DLayered for a one-dimensional or two-dimensional layered texture respectively; Type is an optional argument which defaults to cudaTextureType1D; 

•Type 指定了 texture reference 的类型，其中值为 *cudaTextureType1D*, *cudaTextureType2D*, 或 *cudaTextureType3D*，分别对应于 1 维, 2 维, 或 3 维 texture；或者值为 
*cudaTextureType1DLayered* 或 *cudaTextureType2DLayered* 分别对应于 1 维或 2 维 layered texture。Type 是可选参数，默认值为 *cudaTextureType1D*。

> •ReadMode specifies the read mode; it is an optional argument which defaults to cudaReadModeElementType. 

•ReadMode 指定了读取模式；它也是一个可选参数，默认值是 *cudaReadModeElementType* 。

> A texture reference can only be declared as a static global variable and cannot be passed as an argument to a function. 

texture reference 只能被声明为静态全局变量，不能作为参数传递到方法当中。

> The other attributes of a texture reference are mutable and can be changed at runtime through the host runtime. As explained in the reference manual, the runtime API has a low-level C-style interface and a high-level C++-style interface. The texture type is defined in the high-level API as a structure publicly derived from the textureReference type defined in the low-level API as such: 

texture reference 的其他属性是可变的，并且是在运行时通过 host 运行时改变。正如参考手册中所阐述的，运行时 API 包含底层的 C 样式的接口和高级的 C++ 样式接口。texture 类型在高级 API 中被定义为一种从底层 API 中定义的 * textureReference* 类型公开派生的结构，例如：

``` cuda
struct textureReference {
    int                          normalized;
    enum cudaTextureFilterMode   filterMode;
    enum cudaTextureAddressMode  addressMode[3];
    struct cudaChannelFormatDesc channelDesc;
    int                          sRGB;
    unsigned int                 maxAnisotropy;
    enum cudaTextureFilterMode   mipmapFilterMode;
    float                        mipmapLevelBias;
    float                        minMipmapLevelClamp;
    float                        maxMipmapLevelClamp;
}
```
> •normalized specifies whether texture coordinates are normalized or not; 

•normalized 指定了纹理坐标是否进行规范化；

> •filterMode specifies the filtering mode; 

•filterMode 指定了过滤的模式；

> •addressMode specifies the addressing mode; 

•addressMode 指定了寻址模式；

> •channelDesc describes the format of the texel; it must match the DataType argument of the texture reference declaration; channelDesc is of the following type: 

•channelDesc 描述了 texel 的格式；它必须匹配 texture reference 声明的 DataType 参数；channelDesc 的类型如下：

``` cuda
struct cudaChannelFormatDesc {
  int x, y, z, w;
  enum cudaChannelFormatKind f;
};
```

> where x, y, z, and w are equal to the number of bits of each component of the returned value and f is:  
◦cudaChannelFormatKindSigned if these components are of signed integer type,  
◦cudaChannelFormatKindUnsigned if they are of unsigned integer type,  
◦cudaChannelFormatKindFloat if they are of floating point type.  

其中 x, y, z 和 w 是和返回值的每个成员的位数一致的，f 是一种枚举类型：
◦如果这些成员是有符号的整型，那么 f 的值为 *cudaChannelFormatKindSigned* ，
◦如果它们是无符号整型，那么 f 的值为 *cudaChannelFormatKindUnsigned* ，
◦如果它们是浮点型，那么 f 的值为 *cudaChannelFormatKindFloat* 。

> •See reference manual for sRGB, maxAnisotropy, mipmapFilterMode, mipmapLevelBias, minMipmapLevelClamp, and maxMipmapLevelClamp. 

 关于 sRGB, maxAnisotropy, mipmapFilterMode, mipmapLevelBias, minMipmapLevelClamp, 和 maxMipmapLevelClamp 的概念，请参阅参考手册。
 
 > normalized, addressMode, and filterMode may be directly modified in host code. 
 
 normalized, addressMode, 和 filterMode 是可以直接在 host 的代码进行修改的。
 
 > Before a kernel can use a texture reference to read from texture memory, the texture reference must be bound to a texture using cudaBindTexture() or cudaBindTexture2D() for linear memory, or cudaBindTextureToArray() for CUDA arrays. cudaUnbindTexture() is used to unbind a texture reference. Once a texture reference has been unbound, it can be safely rebound to another array, even if kernels that use the previously bound texture have not completed. It is recommended to allocate two-dimensional textures in linear memory using cudaMallocPitch() and use the pitch returned by cudaMallocPitch() as input parameter to cudaBindTexture2D(). 
 
 在 kernel 可以使用一个 texture reference 去读取 texture 内存之前，texture reference 必须绑定到一个 texture，对于线性内存可以使用 *cudaBindTexture()* 或 *cudaBindTexture2D()*，对于 CUDA array 则使用 *cudaBindTextureToArray()* 。*cudaUnbindTexture()* 是用于解绑 texture reference 的。一旦 texture reference 解绑了，即使使用之前绑定的 texture reference 的 kernel 还没有执行完毕，它仍然可以安全地重新绑定到其他数组上。官方推荐在线性内存上使用 *cudaMallocPitch()* 分配二维的 texture，并使用 *cudaMallocPitch()* 返回的 pitch 参数作为 *cudaBindTexture2D()* 的输入参数。
 
 > The following code samples bind a 2D texture reference to linear memory pointed to by devPtr: 
 
 以下的代码样例绑定了一个 2 维 texture reference 到 devPtr 指向的线性内存：
 
 
> •Using the low-level API:

•使用底层的 API：

``` cuda
texture<float, cudaTextureType2D,
        cudaReadModeElementType> texRef;
textureReference* texRefPtr;
cudaGetTextureReference(&texRefPtr, &texRef);
cudaChannelFormatDesc channelDesc =
                             cudaCreateChannelDesc<float>();
size_t offset;
cudaBindTexture2D(&offset, texRefPtr, devPtr, &channelDesc,
                  width, height, pitch);
```

> •Using the high-level API:

•使用高级的 API：

``` cuda
texture<float, cudaTextureType2D,
        cudaReadModeElementType> texRef;
cudaChannelFormatDesc channelDesc =
                             cudaCreateChannelDesc<float>();
size_t offset;
cudaBindTexture2D(&offset, texRef, devPtr, channelDesc,
                  width, height, pitch);
```

> The following code samples bind a 2D texture reference to a CUDA array cuArray: 

以下代码样例绑定了一个 2 维 texture reference 到 CUDA 数组 cuArray：

> •Using the low-level API:

•使用底层的 API：

``` cuda
texture<float, cudaTextureType2D,
        cudaReadModeElementType> texRef;
textureReference* texRefPtr;
cudaGetTextureReference(&texRefPtr, &texRef);
cudaChannelFormatDesc channelDesc;
cudaGetChannelDesc(&channelDesc, cuArray);
cudaBindTextureToArray(texRef, cuArray, &channelDesc);
```

> •Using the high-level API:

•使用高级的 API:

``` cuda
texture<float, cudaTextureType2D,
        cudaReadModeElementType> texRef;
cudaBindTextureToArray(texRef, cuArray);
```

> The format specified when binding a texture to a texture reference must match the parameters specified when declaring the texture reference; otherwise, the results of texture fetches are undefined. 

当绑定一个 texture 到 texture reference 的时候，指定的格式必须匹配 texture reference 声明时指定的参数；否则 texture fetch 的结果是未定义的。

> There is a limit to the number of textures that can be bound to a kernel as specified in Table 14. 

[link]() Table 14 说明了一个 kernel 可以绑定的 texture 数的限制。

> The following code sample applies some simple transformation kernel to a texture. 

以下代码样例对 texture 运用了一些简单的变换 kernel。

``` cuda
// 二维浮点纹理
texture<float, cudaTextureType2D, cudaReadModeElementType> texRef;

// 简单变换的 kernel
__global__ void transformKernel(float* output,
                                int width, int height,
                                float theta) 
{
    // 计算规范化的纹理坐标
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

    float u = x / (float)width;
    float v = y / (float)height;

    // 坐标变换
    u -= 0.5f;
    v -= 0.5f;
    float tu = u * cosf(theta) - v * sinf(theta) + 0.5f;
    float tv = v * cosf(theta) + u * sinf(theta) + 0.5f;

    // 从纹理读取，并写到全局内存
    output[y * width + x] = tex2D(texRef, tu, tv);
}

// Host 代码
int main()
{
    // 在 device 内存上 分配 CUDA array
    cudaChannelFormatDesc channelDesc =
               cudaCreateChannelDesc(32, 0, 0, 0,
                                     cudaChannelFormatKindFloat);
    cudaArray* cuArray;
    cudaMallocArray(&cuArray, &channelDesc, width, height);

    // 从 host 内存上的 h_data 位置拷贝一些数据到 device 内存中
    cudaMemcpyToArray(cuArray, 0, 0, h_data, size,
                      cudaMemcpyHostToDevice);

    // 设置 texture reference 参数
    texRef.addressMode[0] = cudaAddressModeWrap;
    texRef.addressMode[1] = cudaAddressModeWrap;
    texRef.filterMode     = cudaFilterModeLinear;
    texRef.normalized     = true;

    // 绑定数组到 texture reference
    cudaBindTextureToArray(texRef, cuArray, channelDesc);

    // 在 device 内存上分配变换的结果
    float* output;
    cudaMalloc(&output, width * height * sizeof(float));

    // 发起 kernel
    dim3 dimBlock(16, 16);
    dim3 dimGrid((width  + dimBlock.x - 1) / dimBlock.x,
                 (height + dimBlock.y - 1) / dimBlock.y);
    transformKernel<<<dimGrid, dimBlock>>>(output, width, height,
                                           angle);

    // 释放 device 内存
    cudaFreeArray(cuArray);
    cudaFree(output);

    return 0;
}

```

##### 3.2.11.1.3. 16-Bit Floating-Point Textures
##### 3.2.11.1.3. 16 位的浮点纹理

> The 16-bit floating-point or half format supported by CUDA arrays is the same as the IEEE 754-2008 binary2 format. 

CUDA 数组支持 16 位的浮点型或称作半精度的格式，都是和 IEEE 754-2008 中的 binary2 相同的格式。

> CUDA C does not support a matching data type, but provides intrinsic functions to convert to and from the 32-bit floating-point format via the unsigned short type: __float2half_rn(float) and __half2float(unsigned short). These functions are only supported in device code. Equivalent functions for the host code can be found in the OpenEXR library, for example. 

虽然 CUDA C 并不支持一致的（半精度）数据类型，但提供了通过无符号短整型的内部方法来与 32 位浮点型间进行转换：*__float2half_rn(float)* 和 *__half2float(unsigned short)*。这些方法仅在 deivce 代码中支持。host 代码中等价的 host 方法，举个例子，可以在 OpenEXR 库中找到。

> 16-bit floating-point components are promoted to 32 bit float during texture fetching before any filtering is performed.

在执行任何过滤之前，texture fetch 会将 16 位的浮点型成员提升为 32 位浮点型。

> A channel description for the 16-bit floating-point format can be created by calling one of the cudaCreateChannelDescHalf*() functions. 

一个对于 16 位浮点型格式的通道描述可以通过调用 *cudaCreateChannelDescHalf\*()* 中的其中一个方法来进行创建。

##### 3.2.11.1.4. Layered Textures
##### 3.2.11.1.4 分层纹理

> A one-dimensional or two-dimensional layered texture (also known as texture array in Direct3D and array texture in OpenGL) is a texture made up of a sequence of layers, all of which are regular textures of same dimensionality, size, and data type. 

一维或者二维的分层 texture（在 D3D 中被称作 texture array，OGL 中被称作 array texture）是一种由一系列的层组成的 texture，每一层都是同样维度、大小、数据类型的常规 texture。

> A one-dimensional layered texture is addressed using an integer index and a floating-point texture coordinate; the index denotes a layer within the sequence and the coordinate addresses a texel within that layer. A two-dimensional layered texture is addressed using an integer index and two floating-point texture coordinates; the index denotes a layer within the sequence and the coordinates address a texel within that layer. 

一维的分层纹理由一个整型索引和一个浮点型纹理坐标进行寻址的；这个索引表示了层序列中的某一层，而坐标则在该层的纹理上寻址得到纹素。二维的分层的纹理由一个整型索引和两个浮点型纹理坐标来寻址；同样地，索引表示了层序列中的某一层，而坐标则在该层的纹理上寻址得到纹素。

> A layered texture can only be a CUDA array by calling cudaMalloc3DArray() with the cudaArrayLayered flag (and a height of zero for one-dimensional layered texture). 

分层纹理只能是通过调用 *cudaMalloc3DArray()* 并且标志位为 *cudaArrayLayered* 创建的CUDA 数组（一维的分层纹理的高度为 0 ）

> Layered textures are fetched using the device functions described in tex1DLayered(), tex1DLayered(), tex2DLayered(), and tex2DLayered(). Texture filtering (see Texture Fetching) is done only within a layer, not across layers. 

使用如 [link]() tex1DLayered(), [link]() tex1DLayered(), [link]() tex2DLayered(), 和 [link]() tex2DLayered() 所描述的 device 方法来 fetch 分层纹理。纹理过滤（参阅 [link]() “Texture Fetching” 章节）仅仅在层内进行，并不会跨层。

> Layered textures are only supported on devices of compute capability 2.0 and higher.

分层纹理仅在计算能力在 2.0 及以上的设备上被支持

##### 3.2.11.1.5. Cubemap Textures
##### 3.2.11.1.5 立方体贴图纹理

> A cubemap texture is a special type of two-dimensional layered texture that has six layers representing the faces of a cube: 

立方体贴图纹理是一种特殊类型的二维分层纹理，它有六个层次分别代表了立方体的六个面：

> •The width of a layer is equal to its height.

层的高度等价于它的高度。

> • The cubemap is addressed using three texture coordinates x, y, and z that are interpreted as a direction vector emanating from the center of the cube and pointing to one face of the cube and a texel within the layer corresponding to that face. More specifically, the face is selected by the coordinate with largest magnitude m and the corresponding layer is addressed using coordinates (s/m+1)/2 and (t/m+1)/2 where s and t are defined in Table 1. 

立方体贴图的寻址采用的是三个 texture 坐标x, y, z，可以理解为立方体中心射出的，并指向立方体其中一个面的方向向量，其值是该面层内的纹素。更具体来说，该面是由坐标和最大幅值 m 来选择的，而对应的层是由 (s/m+1)/2 和 (t/m+1)/2 坐标来寻址的， s 和 t 如下表所定义。

 **Table 1. Cubemap Fetch**

<table>
  <tr>
    <th> </th>
    <th> </th>
    <th>face</th>
	<th>m</th>
	<th>s</th>
	<th>t</th>
  </tr>
  <tr>
    <td>|x| > |y| and |x| > |z|</td>
    <td>x>=0</td>
    <td>0</td>
	<td>x</td>
	<td>-z</td>
	<td>-y</td>
  </tr>
  <tr>
    <td> </td>
    <td>x<0</td>
    <td>1</td>
	<td>-x</td>
	<td>z</td>
	<td>-y</td>
  </tr>
  <tr>
    <td>|y| > |x| and |y| > |z|</td>
    <td>y>=0</td>
    <td>2</td>
	<td>y</td>
	<td>x</td>
	<td>z</td>
  </tr>
  <tr>
    <td> </td>
    <td>y<0</td>
    <td>3</td>
	<td>-y</td>
	<td>x</td>
	<td>-z</td>
  </tr>
  <tr>
    <td>|z| > |x| and |z| > |y|</td>
    <td>z>=0</td>
    <td>4</td>
	<td>z</td>
	<td>x</td>
	<td>-y</td>
  </tr>
  <tr>
    <td> </td>
    <td>z<0</td>
    <td>5</td>
	<td>-z</td>
	<td>-x</td>
	<td>-y</td>
  </tr>
</table>

> A layered texture can only be a CUDA array by calling cudaMalloc3DArray() with the cudaArrayCubemap flag. 

分层纹理只能是通过调用 *cudaMalloc3DArray()* 并且标志位为 *cudaArrayCubemap* 创建的CUDA 数组

> Cubemap textures are fetched using the device function described in texCubemap() and texCubemap(). 

使用如 [link]() texCubemap(), [link]() texCubemap() 所描述的 device 方法来 fetch 分层纹理

> Cubemap textures are only supported on devices of compute capability 2.0 and higher.

立方体贴图纹理仅被计算能力在 2.0 及以上的设备支持

##### 3.2.11.1.6. Cubemap Layered Textures
##### 3.2.11.1.6. 立方体贴图分层纹理

> A cubemap layered texture is a layered texture whose layers are cubemaps of same dimension. 

立方体贴图分层纹理是一种每层是维度相同的立方体贴图的分层纹理。

> A cubemap layered texture is addressed using an integer index and three floating-point texture coordinates; the index denotes a cubemap within the sequence and the coordinates address a texel within that cubemap. 

立方体贴图分层纹理使用一个整型索引和三个浮点型纹理坐标来寻址；整型索引表示序列内的一个立方体贴图，坐标则是用于寻址立方体贴图上的纹素。

> A layered texture can only be a CUDA array by calling cudaMalloc3DArray() with the cudaArrayLayered and cudaArrayCubemap flags. 

分层纹理仅限于通过调用 *cudaMalloc3DArray()* 创建的 CUDA 数组，且标志位为 *cudaArrayLayered*。

> Cubemap layered textures are fetched using the device function described in texCubemapLayered() and texCubemapLayered(). Texture filtering (see Texture Fetching) is done only within a layer, not across layers. 

立方体贴图分层纹理使用 [link]() *texCubemapLayered()* [link]() *texCubemapLayered()* 描述的 device 方法来 fetch，纹理过滤仅在一层有效（参见 [link]() “Texture Fetching” 章节），无法跨层。

> Cubemap layered textures are only supported on devices of compute capability 2.0 and higher.

立方体贴图分层纹理仅被计算能力 2.0 及以上的设备支持。

##### 3.2.11.1.7. Texture Gather
##### 3.2.11.1.7. 纹理收集

> Texture gather is a special texture fetch that is available for two-dimensional textures only. It is performed by the tex2Dgather() function, which has the same parameters as tex2D(), plus an additional comp parameter equal to 0, 1, 2, or 3 (see tex2Dgather() and tex2Dgather()). It returns four 32-bit numbers that correspond to the value of the component comp of each of the four texels that would have been used for bilinear filtering during a regular texture fetch. For example, if these texels are of values (253, 20, 31, 255), (250, 25, 29, 254), (249, 16, 37, 253), (251, 22, 30, 250), and comp is 2, tex2Dgather() returns (31, 29, 37, 30). 

texture gather 是一种特殊的 texture fetch ，仅适用于二维纹理。通过 *tex2Dgather()* 方法执行，除了与 *tex2D()* 具有相同的参数外，还有一个参数 comp，其值为 0, 1, 2 或 3 （参见 *tex2Dgather()* 和 *tex2Dgather()* ）。它返回 4 个 32 位的数字，这些数字对应于在常规纹理 fetch 过程中用于双线性过滤的四个 texel 的第 comp 个成员  的值。例如，如果这些texel值为（253, 20, **31**, 255）、（250, 25, **29**, 254）、（249, 16, **37**, 253）、（251, 22, **30**, 250）和 comp 的值为 2，则 *tex2Dgather()* 返回（31, 29, 37, 30）。

> Note that texture coordinates are computed with only 8 bits of fractional precision. tex2Dgather() may therefore return unexpected results for cases where tex2D() would use 1.0 for one of its weights (α or β, see Linear Filtering). For example, with an x texture coordinate of 2.49805: xB=x-0.5=1.99805, however the fractional part of xB is stored in an 8-bit fixed-point format. Since 0.99805 is closer to 256.f/256.f than it is to 255.f/256.f, xB has the value 2. A tex2Dgather() in this case would therefore return indices 2 and 3 in x, instead of indices 1 and 2. 

注意，纹理坐标仅计算 8 位的小数部分精度。*tex2Dgather()* 可能因此会返回一些预料不到的结果，因为 *tex2D()* 将使用 1.0 作为其权重之一(α 或 β, 参见 [link]() “Linear Filtering” 章节)。例如，x纹理坐标是 2.49805 : xB=x-0.5=1.99805，但xB的小数部分是以 8 位定点格式存储的。因为 0.99805 比 255.f/256.f 更接近 256.f/256.f ，所以 xB 的值是2。在这种情况下，tex2Dgather（）将返回 x 中的索引 2 和 3 的数，而不是索引 1 和 2。

> Texture gather is only supported for CUDA arrays created with the cudaArrayTextureGather flag and of width and height less than the maximum specified in Table 14 for texture gather, which is smaller than for regular texture fetch. 

纹理 gather 仅被带 *cudaArrayTextureGather* 标志位的 CUDA 数组所支持，宽度和高度小于 [link]() Table 14 中为纹理 gather 所指定的最大值，这比常规纹理 fetch 要小。

> Texture gather is only supported on devices of compute capability 2.0 and higher. 

纹理 gather 仅被计算能力在 2.0 及以上的设备支持。

#### 3.2.11.2. Surface Memory
#### 3.2.11.2. 表面内存

> For devices of compute capability 2.0 and higher, a CUDA array (described in Cubemap Surfaces), created with the cudaArraySurfaceLoadStore flag, can be read and written via a surface object or surface reference using the functions described in Surface Functions.

对于计算能力在 2.0 及以上的设备，带 *cudaArraySurfaceLoadStore* 标志位创建的 CUDA 数组（在 [link]() “Cubemap Surfaces” 章节中提及）是可以通过 surface 对象或引用使用 [link]() “Surface Functions” 章节中提到的方法读写的

> Table 14 lists the maximum surface width, height, and depth depending on the compute capability of the device.

##### 3.2.11.2.1. Surface Object API
##### 3.2.11.2.1. 表面对象 API

> A surface object is created using cudaCreateSurfaceObject() from a resource description of type struct cudaResourceDesc.

一个 Surface 对象是使用 *cudacreateSurfaceObject()* 从 *cudaResourceDesc* 类型结构体的资源描述中创建的。

> The following code sample applies some simple transformation kernel to a texture.

以下代码示例运用了一些简单的变换 kernel 于纹理上

``` cuda
// 简单的拷贝 kernel
__global__ void copyKernel(cudaSurfaceObject_t inputSurfObj,
                           cudaSurfaceObject_t outputSurfObj,
                           int width, int height) 
{
    // 计算 surface 坐标
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < width && y < height) {
        uchar4 data;
        // 从输入 surface 读取
        surf2Dread(&data,  inputSurfObj, x * 4, y);
        // 写到输出 surface
        surf2Dwrite(data, outputSurfObj, x * 4, y);
    }
}

// Host 代码
int main()
{
    // 在 device 内存上分配 CUDA 数组
    cudaChannelFormatDesc channelDesc =
             cudaCreateChannelDesc(8, 8, 8, 8,
                                   cudaChannelFormatKindUnsigned);
    cudaArray* cuInputArray;
    cudaMallocArray(&cuInputArray, &channelDesc, width, height,
                    cudaArraySurfaceLoadStore);
    cudaArray* cuOutputArray;
    cudaMallocArray(&cuOutputArray, &channelDesc, width, height,
                    cudaArraySurfaceLoadStore);

    // 从 host 内存的 h_data 位置上拷贝一些数组到 device 内存上 
    cudaMemcpyToArray(cuInputArray, 0, 0, h_data, size,
                      cudaMemcpyHostToDevice);

    // 指定 surface
    struct cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeArray;

    // 创建 surface 对象
    resDesc.res.array.array = cuInputArray;
    cudaSurfaceObject_t inputSurfObj = 0;
    cudaCreateSurfaceObject(&inputSurfObj, &resDesc);
    resDesc.res.array.array = cuOutputArray;
    cudaSurfaceObject_t outputSurfObj = 0;
    cudaCreateSurfaceObject(&outputSurfObj, &resDesc);

    // 启动 kernel
    dim3 dimBlock(16, 16);
    dim3 dimGrid((width  + dimBlock.x - 1) / dimBlock.x,
                 (height + dimBlock.y - 1) / dimBlock.y);
    copyKernel<<<dimGrid, dimBlock>>>(inputSurfObj,
                                      outputSurfObj,
                                      width, height);


    // 销毁 surface 对象
    cudaDestroySurfaceObject(inputSurfObj);
    cudaDestroySurfaceObject(outputSurfObj);

    // 释放 device 内存
    cudaFreeArray(cuInputArray);
    cudaFreeArray(cuOutputArray);

    return 0;
}
```

##### 3.2.11.2.2. Surface Reference API
##### 3.2.11.2.2. 表面引用 API

> A surface reference is declared at file scope as a variable of type surface: 

surface reference 是一个声明在文件范围内 surface 类型变量：

*surface<void, Type> surfRef;*

> where Type specifies the type of the surface reference and is equal to cudaSurfaceType1D, cudaSurfaceType2D, cudaSurfaceType3D, cudaSurfaceTypeCubemap, cudaSurfaceType1DLayered, cudaSurfaceType2DLayered, or cudaSurfaceTypeCubemapLayered; Type is an optional argument which defaults to cudaSurfaceType1D. A surface reference can only be declared as a static global variable and cannot be passed as an argument to a function. 

其中 *Type* 指定了 surface reference 的类型，可选的值为 *cudaSurfaceType1D*, *cudaSurfaceType2D*, *cudaSurfaceType3D*, *cudaSurfaceTypeCubemap*, *cudaSurfaceType1DLayered*, *cudaSurfaceType2DLayered*, 或 *cudaSurfaceTypeCubemapLayered*；
*Type* 是可选的参数，默认值为 cudaSurfaceType1D 。surface reference 只能被声明为静态全局变量，并不能作为参数传递给方法。

> Before a kernel can use a surface reference to access a CUDA array, the surface reference must be bound to the CUDA array using cudaBindSurfaceToArray(). 

在 kernel 能够使用 surface reference 去访问 CUDA 数组之前，必须通过 *cudaBindSurfaceToArray()* 将 surface reference 绑定到 CUDA 数组。

> The following code samples bind a surface reference to a CUDA array cuArray: 

以下的代码样例绑定了一个 surface reference 到 CUDA 数组 cuArray:


> •Using the low-level API:

•使用底层的 API：

``` cuda
surface<void, cudaSurfaceType2D> surfRef;
surfaceReference* surfRefPtr;
cudaGetSurfaceReference(&surfRefPtr, "surfRef");
cudaChannelFormatDesc channelDesc;
cudaGetChannelDesc(&channelDesc, cuArray);
cudaBindSurfaceToArray(surfRef, cuArray, &channelDesc);
```

> •Using the high-level API:

•使用高级的 API：

``` cuda
surface<void, cudaSurfaceType2D> surfRef;
cudaBindSurfaceToArray(surfRef, cuArray);
```

> A CUDA array must be read and written using surface functions of matching dimensionality and type and via a surface reference of matching dimensionality; otherwise, the results of reading and writing the CUDA array are undefined. 

CUDA 数组的读写必须通过匹配维度的 surface reference 并使用匹配维度和类型的 surface 方法；否则读写 CUDA 数组的结果将是未定义的。

> Unlike texture memory, surface memory uses byte addressing. This means that the x-coordinate used to access a texture element via texture functions needs to be multiplied by the byte size of the element to access the same element via a surface function. For example, the element at texture coordinate x of a one-dimensional floating-point CUDA array bound to a texture reference texRef and a surface reference surfRef is read using tex1d(texRef, x) via texRef, but surf1Dread(surfRef, 4*x) via surfRef. Similarly, the element at texture coordinate x and y of a two-dimensional floating-point CUDA array bound to a texture reference texRef and a surface reference surfRef is accessed using tex2d(texRef, x, y) via texRef, but surf2Dread(surfRef, 4*x, y) via surfRef (the byte offset of the y-coordinate is internally calculated from the underlying line pitch of the CUDA array). 

不同于 texture 内存，surface 内存使用字节寻址。这意味着通过 texture 方法访问的 x 坐标的 texture 元素，通过 surface 方法的话还需要乘上相应元素的字节大小。例如，一个一维的浮点 CUDA 数组分别被绑定到一个 texture reference —— texRef 和一个 surface reference —— surfRef 上，要访问位于纹理坐标 x 上的元素，texRef 通过 *tex1d（texRef，x）* 来读取，但是 surfRef 则是通过 *surf1Dread（surfRef，4\*x）* 来读取。类似地，，一个分别绑定到一个 texture reference —— texRef 和一个 surface reference —— surfRef 的二维浮点 CUDA 数组，其位于纹理坐标 x 和 y 的元素，texRef 是通过 *tex2d（texRef，x，y）* 来访问，而 surfRef 则是通过  *surf2Dread（surfRef，4\*x，y）* 来访问（y 坐标的字节偏移量是从 CUDA 数组潜在的 line pitch 内部计算的）。

> The following code sample applies some simple transformation kernel to a texture. 

以下代码的示例运用了一些简单的变换 kernel 于纹理之上。

``` cuda
// 2D surfaces
surface<void, 2> inputSurfRef;
surface<void, 2> outputSurfRef;
            
// 简单的拷贝 kernel
__global__ void copyKernel(int width, int height) 
{
    // 计算 surface 坐标
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < width && y < height) {
        uchar4 data;
        // 从输入 surface 中读取
        surf2Dread(&data,  inputSurfRef, x * 4, y);
        // 写出到输出 surface
        surf2Dwrite(data, outputSurfRef, x * 4, y);
    }
}

// Host 代码
int main()
{
    // 在 device 内存你上分配 CUDA 数组
    cudaChannelFormatDesc channelDesc =
             cudaCreateChannelDesc(8, 8, 8, 8,
                                   cudaChannelFormatKindUnsigned);
    cudaArray* cuInputArray;
    cudaMallocArray(&cuInputArray, &channelDesc, width, height,
                    cudaArraySurfaceLoadStore);
    cudaArray* cuOutputArray;
    cudaMallocArray(&cuOutputArray, &channelDesc, width, height,
                    cudaArraySurfaceLoadStore);

    // 拷贝 host 内存上位于 h_data 的若干数据到 device 内存。
    cudaMemcpyToArray(cuInputArray, 0, 0, h_data, size,
                      cudaMemcpyHostToDevice);

    // 绑定数组到 surface references
    cudaBindSurfaceToArray(inputSurfRef, cuInputArray);
    cudaBindSurfaceToArray(outputSurfRef, cuOutputArray);

    // 启动 kernel
    dim3 dimBlock(16, 16);
    dim3 dimGrid((width  + dimBlock.x - 1) / dimBlock.x,
                 (height + dimBlock.y - 1) / dimBlock.y);
    copyKernel<<<dimGrid, dimBlock>>>(width, height);

    // 释放 device 内存
    cudaFreeArray(cuInputArray);
    cudaFreeArray(cuOutputArray);

    return 0;
}
```

##### 3.2.11.2.3. Cubemap Surfaces
##### 3.2.11.2.3. 立方体贴图表面

> Cubemap surfaces are accessed using surfCubemapread() and surfCubemapwrite() (surfCubemapread and surfCubemapwrite) as a two-dimensional layered surface, i.e., using an integer index denoting a face and two floating-point texture coordinates addressing a texel within the layer corresponding to this face. Faces are ordered as indicated in Table 1. 

Cubemap surfaces 作为一个二维的分层 surface 是通过 *surfCubemapread()* 和 *surfCubemapwrite()* 来访问的（参阅 [link]() “surfCubemapread”章节 和 [link]() “surfCubemapwrite” 章节）。即使用一个整型 index 代表面，两个浮点型纹理坐标用来在对应面的层内进行纹素寻址。面的排序如 [link]() Table 1 所示。

##### 3.2.11.2.4. Cubemap Layered Surfaces
##### 3.2.11.2.4. 立方体分层表面

> Cubemap layered surfaces are accessed using surfCubemapLayeredread() and surfCubemapLayeredwrite() (surfCubemapLayeredread() and surfCubemapLayeredwrite()) as a two-dimensional layered surface, i.e., using an integer index denoting a face of one of the cubemaps and two floating-point texture coordinates addressing a texel within the layer corresponding to this face. Faces are ordered as indicated in Table 1, so index ((2 * 6) + 3), for example, accesses the fourth face of the third cubemap. 

Cubemap layered surfaces 作为个二维分层的 surface 是通过 *surfCubemapLayeredread()* 和 *surfCubemapLayeredwrite()* 来访问的（参阅 [link]() “surfCubemapLayeredread”章节 和 [link]() “surfCubemapLayeredwrite” 章节）。即，使用一个整型索引来表示其中一个立方体贴图的某一个面，两个浮点型纹理坐标用来寻址对应面的层内的纹素。面的排序如 [link]() Table 1 所示，例如，访问第三个 cubemap 的第四个面，index = ((2*6)+3) 。


#### 3.2.11.3. CUDA Arrays
#### 3.2.11.3. CUDA 数组

> CUDA arrays are opaque memory layouts optimized for texture fetching. They are one dimensional, two dimensional, or three-dimensional and composed of elements, each of which has 1, 2 or 4 components that may be signed or unsigned 8-, 16-, or 32-bit integers, 16-bit floats, or 32-bit floats. CUDA arrays are only accessible by kernels through texture fetching as described in Texture Memory or surface reading and writing as described in Surface Memory. 

CUDA 数组是为了纹理取回优化的不透明内存布局。它们是一维、二维、或三维，由1、2 或 4 个成员组成的元素，它们的类型可能是有符号的或者无符号的 8-、16-、或 32 位整型，16 位或 32 位浮点型。CUDA 数组只能由 kernel 通过纹理取回访问（如 “Texture Memory” 章节所述）或表面读写（如 “Surface Memory” 章节所述）访问

#### 3.2.11.4. Read/Write Coherency
#### 3.2.11.4. 读写的一致性

> The texture and surface memory is cached (see Device Memory Accesses) and within the same kernel call, the cache is not kept coherent with respect to global memory writes and surface memory writes, so any texture fetch or surface read to an address that has been written to via a global write or a surface write in the same kernel call returns undefined data. In other words, a thread can safely read some texture or surface memory location only if this memory location has been updated by a previous kernel call or memory copy, but not if it has been previously updated by the same thread or another thread from the same kernel call. 

纹理和表面内存在同一个 kernel 调用中是被缓存的（参阅 [link]()“Device Memory Accesses” 章节），对于全局内存写和表面内存写，缓存并没有保持一致，因此纹理 fetch 或表面读取一个在相同 kernel 内已经通过全局内存写或表面内存写的地址时，返回的结果将是未定义的数据。换句话说，一个线程能安全地读取一些纹理或表面的内存位置的前提是—该内存位置是由先前的 kernel 调用更新或内存拷贝，但是如果该内存位置被相同线程或相同 kernel 内的不同线程更新则不是安全的。

---

### 3.2.12. Graphics Interoperability
### 3.2.12. 图形交互

> Some resources from OpenGL and Direct3D may be mapped into the address space of CUDA, either to enable CUDA to read data written by OpenGL or Direct3D, or to enable CUDA to write data for consumption by OpenGL or Direct3D. 

一些 GL 和 D3D 的资源可以被映射到 CUDA 地址空间，可以使得 CUDA 能读取 GL 或 D3D 写入的数据，或使得 CUDA 能写数据以提供给 GL 或 D3D 使用。

> A resource must be registered to CUDA before it can be mapped using the functions mentioned in OpenGL Interoperability and Direct3D Interoperability. These functions return a pointer to a CUDA graphics resource of type struct cudaGraphicsResource. Registering a resource is potentially high-overhead and therefore typically called only once per resource. A CUDA graphics resource is unregistered using cudaGraphicsUnregisterResource(). Each CUDA context which intends to use the resource is required to register it separately. 

在将资源通过  [link]() “OpenGL Interoperability” 章节和 [link]() “Direct3D Interoperability” 章节中 提到的交互方法进行映射之前，必须将该资源注册到 CUDA。这些方法会返回一个指向 *cudaGraphicsResource* CUDA 结构体类型的图形资源的指针。注册资源是一种潜在的高开销的操作，因此通常每个资源只注册一次。通过 *cudaGraphicsUnregisterResource()* 方法可以卸载 CUDA 图形资源。每一个想要使用该资源的 CUDA 上下文必须分别对资源进行注册。

> Once a resource is registered to CUDA, it can be mapped and unmapped as many times as necessary using cudaGraphicsMapResources() and cudaGraphicsUnmapResources(). cudaGraphicsResourceSetMapFlags() can be called to specify usage hints (write-only, read-only) that the CUDA driver can use to optimize resource management. 

一旦将一个资源注册到了 CUDA，它就可以根据需要通过 *cudaGraphicsMapResources()* 或 *cudaGraphicsUnmapResources()* 被映射或者逆映射多次。*cudaGraphicsResourceSetMapFlags()* 可以被用于指定使用提示（只写，只读），这样 CUDA 驱动便可以优化对资源的管理。

> A mapped resource can be read from or written to by kernels using the device memory address returned by cudaGraphicsResourceGetMappedPointer() for buffers and cudaGraphicsSubResourceGetMappedArray() for CUDA arrays. 

kernel 可以分别通过 *cudaGraphicsResourceGetMappedPointer()* 和 *cudaGraphicsSubResourceGetMappedArray()* 来获取 buffer 和 CUDA 数组的 device 内存地址，以实现对映射资源的读写。

> Accessing a resource through OpenGL, Direct3D, or another CUDA context while it is mapped produces undefined results. OpenGL Interoperability and Direct3D Interoperability give specifics for each graphics API and some code samples. SLI Interoperability gives specifics for when the system is in SLI mode. 

通过 GL, D3D, 或其他 CUDA 上下文访问正在映射的资源时会产生未定义的结果。 [link]() “OpenGL Interoperability” 章节和 [link]() “Direct3D Interoperability” 章节给出每个图形 API的细节和一些代码示例。“SLI Interoperability” 章节为系统是 SLI 模式时给出了详细说明。

#### 3.2.12.1. OpenGL Interoperability
#### 3.2.12.1. OpenGL 的互操作

> The OpenGL resources that may be mapped into the address space of CUDA are OpenGL buffer, texture, and renderbuffer objects.

GL 中可以映射为 CUDA 地址空间的资源有 texture 和 renderbuffer 对象。

> A buffer object is registered using cudaGraphicsGLRegisterBuffer(). In CUDA, it appears as a device pointer and can therefore be read and written by kernels or via cudaMemcpy() calls. 

一个 buffer 对象可以通过 *cudaGraphicsGLRegisterBuffer()* 进行注册。在 CUDA 中，它看起来就像是一个 device 指针，并因此可以被 kernel 读写或通过 cudaMemcpy() 调用。

> A texture or renderbuffer object is registered using cudaGraphicsGLRegisterImage(). In CUDA, it appears as a CUDA array. Kernels can read from the array by binding it to a texture or surface reference. They can also write to it via the surface write functions if the resource has been registered with the cudaGraphicsRegisterFlagsSurfaceLoadStore flag. The array can also be read and written via cudaMemcpy2D() calls. cudaGraphicsGLRegisterImage() supports all texture formats with 1, 2, or 4 components and an internal type of float (e.g., GL_RGBA_FLOAT32), normalized integer (e.g., GL_RGBA8, GL_INTENSITY16), and unnormalized integer (e.g., GL_RGBA8UI) (please note that since unnormalized integer formats require OpenGL 3.0, they can only be written by shaders, not the fixed function pipeline). 

一个 texture 或 renderbuffer 对象可以通过 *cudaGraphicsGLRegisterImage()* 进行注册。在 CUDA 中，它看起来就像是一个 CUDA 数组。kernel 可以将它绑定到一个 texture 或 surface 引用来从数组读取。如果该资源在注册时是带 *cudaGraphicsRegisterFlagsSurfaceLoadStore* 标志位的，那么就可以通过 surface 的写方法来写入。数组同样可以使用 *cudaMemcpy2D()* 进行读写。*cudaGraphicsGLRegisterImage()* 支持所有的 1, 2， 或 4 成员的纹理格式，以及一个浮点内部类型（例如，*GL_RGBA_FLOAT32*）、规范化的整型（例如，*GL_RGBA8*, *GL_INTENSITY16*），为规范化的整型（例如，*GL_RGBA8UI*）。（请注意，因为为规范化的整型格式需要 GL 3.0，所以他们只能被着色器写入，而不是定点方法管线）。

> The OpenGL context whose resources are being shared has to be current to the host thread making any OpenGL interoperability API calls. 

正在共享资源的 GL 上下文对于正在进行 GL 交互 API 调用的 host 线程必须是当前的上下文。

> Please note: When an OpenGL texture is made bindless (say for example by requesting an image or texture handle using the glGetTextureHandle*/glGetImageHandle* APIs) it cannot be registered with CUDA. The application needs to register the texture for interop before requesting an image or texture handle. 

请记住：当 GL 的纹理尚未绑定时（比如，通过 *glGetTextureHandle\*/glGetImageHandle\* APIs* 来请求图像或纹理句柄 ），它将不能被 CUDA 所注册。应用程序需要在请求图像或纹理句柄之前注册用于交互的纹理。

> The following code sample uses a kernel to dynamically modify a 2D width x height grid of vertices stored in a vertex buffer object: 

以下代码示例使用了一个 kernel 动态地修改一个存储在 VBO 中的二维 width x height 大小的点格：

``` cuda
GLuint positionsVBO;
struct cudaGraphicsResource* positionsVBO_CUDA;

int main()
{
    // 为 device 0 初始化 OpenGL 和 GLUT
    // 并使得 OpenGL 上下文对象成为当前
    ...
    glutDisplayFunc(display);

    // 明确地指定 device 0
    cudaSetDevice(0);

    // 创建 VBO，并将它注册到 CUDA
    glGenBuffers(1, &positionsVBO);
    glBindBuffer(GL_ARRAY_BUFFER, positionsVBO);
    unsigned int size = width * height * 4 * sizeof(float);
    glBufferData(GL_ARRAY_BUFFER, size, 0, GL_DYNAMIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    cudaGraphicsGLRegisterBuffer(&positionsVBO_CUDA,
                                 positionsVBO,
                                 cudaGraphicsMapFlagsWriteDiscard);

    // 启动渲染循环
    glutMainLoop();

    ...
}

void display()
{
    // 映射 VBO 用于 CUDA 写入
    float4* positions;
    cudaGraphicsMapResources(1, &positionsVBO_CUDA, 0);
    size_t num_bytes; 
    cudaGraphicsResourceGetMappedPointer((void**)&positions,
                                         &num_bytes,  
                                         positionsVBO_CUDA));

    // 执行 kernel
    dim3 dimBlock(16, 16, 1);
    dim3 dimGrid(width / dimBlock.x, height / dimBlock.y, 1);
    createVertices<<<dimGrid, dimBlock>>>(positions, time,
                                          width, height);

    // 逆映射 VBO
    cudaGraphicsUnmapResources(1, &positionsVBO_CUDA, 0);

    // 从 VBO 上进行渲染
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glBindBuffer(GL_ARRAY_BUFFER, positionsVBO);
    glVertexPointer(4, GL_FLOAT, 0, 0);
    glEnableClientState(GL_VERTEX_ARRAY);
    glDrawArrays(GL_POINTS, 0, width * height);
    glDisableClientState(GL_VERTEX_ARRAY);

    // 交换双缓冲
    glutSwapBuffers();
    glutPostRedisplay();
}
```

``` cuda
void deleteVBO()
{
    cudaGraphicsUnregisterResource(positionsVBO_CUDA);
    glDeleteBuffers(1, &positionsVBO);
}

__global__ void createVertices(float4* positions, float time,
                               unsigned int width, unsigned int height)
{
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

    // 计算 UV 的坐标
    float u = x / (float)width;
    float v = y / (float)height;
    u = u * 2.0f - 1.0f;
    v = v * 2.0f - 1.0f;

    // 计算简单的正弦波形
    float freq = 4.0f;
    float w = sinf(u * freq + time)
            * cosf(v * freq + time) * 0.5f;

    // 写入顶点的位置
    positions[y * width + x] = make_float4(u, w, v, 1.0f);
}
```

> On Windows and for Quadro GPUs, cudaWGLGetDevice() can be used to retrieve the CUDA device associated to the handle returned by wglEnumGpusNV(). Quadro GPUs offer higher performance OpenGL interoperability than GeForce and Tesla GPUs in a multi-GPU configuration where OpenGL rendering is performed on the Quadro GPU and CUDA computations are performed on other GPUs in the system. 

在 Windows 和 Quadro 系列 GPU 上，*cudaWGLGetDevice()* 可以用来检索与 *wglEnumGpusNV()* 返回的句柄相关联的 CUDA 设备。在多 GPU 配置中，Quadro GPU 提供了比 GeForce 系列和 Tesla  系列 GPU 更高性能的 OpenGL 互操作性，在这个系统中，OpenGL渲染是在 Quadro GPU 上执行的，而 CUDA 计算则是在系统中的其他 GPU 上执行的。

#### 3.2.12.2. Direct3D Interoperability
#### 3.2.12.2. D3D 的互操作

> Direct3D interoperability is supported for Direct3D 9Ex, Direct3D 10, and Direct3D 11.

D3D 9Ex，D3D 10, D3D 11 都支持 D3D 的互操作。

> A CUDA context may interoperate only with Direct3D devices that fulfill the following criteria: Direct3D 9Ex devices must be created with DeviceType set to D3DDEVTYPE_HAL and BehaviorFlags with the D3DCREATE_HARDWARE_VERTEXPROCESSING flag; Direct3D 10 and Direct3D 11 devices must be created with DriverType set to D3D_DRIVER_TYPE_HARDWARE. 

CUDA 上下文仅能与满足以下条件的 D3D 设备进行交互：D3D 9Ex 设备必须在创建时将 *DeviceType* 设为 *D3DDEVTYPE_HAL* 和 *BehaviorFlags* 设为 *D3DCREATE_HARDWARE_VERTEXPROCESSING*；D3D 10 和 D3D 11设备必须在创建时将 *DriverType* 设为 D3D_DRIVER_TYPE_HARDWARE。

> The Direct3D resources that may be mapped into the address space of CUDA are Direct3D buffers, textures, and surfaces. These resources are registered using cudaGraphicsD3D9RegisterResource(), cudaGraphicsD3D10RegisterResource(), and cudaGraphicsD3D11RegisterResource(). 

可以被映射到 CUDA 地址空间的 D3D 资源是 D3D buffer、texture、surface。这些资源使用  *cudaGraphicsD3D9RegisterResource()* *cudaGraphicsD3D10RegisterResource()*, 和 *cudaGraphicsD3D11RegisterResource()* 来注册。

> The following code sample uses a kernel to dynamically modify a 2D width x height grid of vertices stored in a vertex buffer object. 

以下代码示例使用一个 kernel 来动态修改一个存储在顶点缓存对象中的大小为 width x height 的二维顶点格子。

##### 3.2.12.2.1. Direct3D 9 Version
##### 3.2.12.2.1 D3D 9 版本

``` cuda
IDirect3D9* D3D;
IDirect3DDevice9* device;
struct CUSTOMVERTEX {
    FLOAT x, y, z;
    DWORD color;
};
IDirect3DVertexBuffer9* positionsVB;
struct cudaGraphicsResource* positionsVB_CUDA;

int main()
{
    int dev;
    // 初始化 Direct3D
    D3D = Direct3DCreate9Ex(D3D_SDK_VERSION);

    // 获得一个 CUDA-使能的适配器
    unsigned int adapter = 0;
    for (; adapter < g_pD3D->GetAdapterCount(); adapter++) {
        D3DADAPTER_IDENTIFIER9 adapterId;
        g_pD3D->GetAdapterIdentifier(adapter, 0, &adapterId);
        if (cudaD3D9GetDevice(&dev, adapterId.DeviceName)
            == cudaSuccess)
            break;
    }

     // 创建 device
    ...
    D3D->CreateDeviceEx(adapter, D3DDEVTYPE_HAL, hWnd,
                        D3DCREATE_HARDWARE_VERTEXPROCESSING,
                        &params, NULL, &device);

    // 使用相同的 device
    cudaSetDevice(dev);

    // 创建顶点缓存并注册到 CUDA
    unsigned int size = width * height * sizeof(CUSTOMVERTEX);
    device->CreateVertexBuffer(size, 0, D3DFVF_CUSTOMVERTEX,
                               D3DPOOL_DEFAULT, &positionsVB, 0);
    cudaGraphicsD3D9RegisterResource(&positionsVB_CUDA,
                                     positionsVB,
                                     cudaGraphicsRegisterFlagsNone);
    cudaGraphicsResourceSetMapFlags(positionsVB_CUDA,
                                    cudaGraphicsMapFlagsWriteDiscard);

    // 启动渲染循环
    while (...) {
        ...
        Render();
        ...
    }
    ...
}

```

``` cuda
void Render()
{
    // 映射顶点缓存用于从 CUDA 写入
    float4* positions;
    cudaGraphicsMapResources(1, &positionsVB_CUDA, 0);
    size_t num_bytes; 
    cudaGraphicsResourceGetMappedPointer((void**)&positions,
                                         &num_bytes,  
                                         positionsVB_CUDA));

    // 执行 kernel
    dim3 dimBlock(16, 16, 1);
    dim3 dimGrid(width / dimBlock.x, height / dimBlock.y, 1);
    createVertices<<<dimGrid, dimBlock>>>(positions, time,
                                          width, height);

    // 逆映射顶点缓存
    cudaGraphicsUnmapResources(1, &positionsVB_CUDA, 0);

    // 绘制并显示
    ...
}

void releaseVB()
{
    cudaGraphicsUnregisterResource(positionsVB_CUDA);
    positionsVB->Release();
}

__global__ void createVertices(float4* positions, float time,
                               unsigned int width, unsigned int height)
{
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

    // 计算 UV 坐标
    float u = x / (float)width;
    float v = y / (float)height;
    u = u * 2.0f - 1.0f;
    v = v * 2.0f - 1.0f;

    // 计算简单的正弦波形
    float freq = 4.0f;
    float w = sinf(u * freq + time)
            * cosf(v * freq + time) * 0.5f;

    // 写入顶点位置
    positions[y * width + x] =
                make_float4(u, w, v, __int_as_float(0xff00ff00));
}
```

##### 3.2.12.2.2. Direct3D 10 Version
##### 3.2.12.2.2. D3D 10 版本

``` cuda
ID3D10Device* device;
struct CUSTOMVERTEX {
    FLOAT x, y, z;
    DWORD color;
};
ID3D10Buffer* positionsVB;
struct cudaGraphicsResource* positionsVB_CUDA;
            
int main()
{
    int dev;
    // 获得一个 CUDA-使能的适配器
    IDXGIFactory* factory;
    CreateDXGIFactory(__uuidof(IDXGIFactory), (void**)&factory);
    IDXGIAdapter* adapter = 0;
    for (unsigned int i = 0; !adapter; ++i) {
        if (FAILED(factory->EnumAdapters(i, &adapter))
            break;
        if (cudaD3D10GetDevice(&dev, adapter) == cudaSuccess)
            break;
        adapter->Release();
    }
    factory->Release();
            
    // 创建一个 swap chain 和 device
    ...
    D3D10CreateDeviceAndSwapChain(adapter, 
                                  D3D10_DRIVER_TYPE_HARDWARE, 0, 
                                  D3D10_CREATE_DEVICE_DEBUG,
                                  D3D10_SDK_VERSION, 
                                  &swapChainDesc, &swapChain,
                                  &device);
    adapter->Release();

    // 使用相同 device
    cudaSetDevice(dev);

    // 创建顶点缓存并注册到 CUDA
    unsigned int size = width * height * sizeof(CUSTOMVERTEX);
    D3D10_BUFFER_DESC bufferDesc;
    bufferDesc.Usage          = D3D10_USAGE_DEFAULT;
    bufferDesc.ByteWidth      = size;
    bufferDesc.BindFlags      = D3D10_BIND_VERTEX_BUFFER;
    bufferDesc.CPUAccessFlags = 0;
    bufferDesc.MiscFlags      = 0;
    device->CreateBuffer(&bufferDesc, 0, &positionsVB);
    cudaGraphicsD3D10RegisterResource(&positionsVB_CUDA,
                                      positionsVB,
                                      cudaGraphicsRegisterFlagsNone);
                                      cudaGraphicsResourceSetMapFlags(positionsVB_CUDA,
                                      cudaGraphicsMapFlagsWriteDiscard);

    // 启动渲染循环
    while (...) {
        ...
        Render();
        ...
    }
    ...
}
```

``` cuda
void Render()
{
    // 映射顶点缓存用于 CUDA 写入
    float4* positions;
    cudaGraphicsMapResources(1, &positionsVB_CUDA, 0);
    size_t num_bytes; 
    cudaGraphicsResourceGetMappedPointer((void**)&positions,
                                         &num_bytes,  
                                         positionsVB_CUDA));

    // 执行 kernel
    dim3 dimBlock(16, 16, 1);
    dim3 dimGrid(width / dimBlock.x, height / dimBlock.y, 1);
    createVertices<<<dimGrid, dimBlock>>>(positions, time,
                                          width, height);

    // 逆映射顶点缓存
    cudaGraphicsUnmapResources(1, &positionsVB_CUDA, 0);

    // 绘制并显示
    ...
}

void releaseVB()
{
    cudaGraphicsUnregisterResource(positionsVB_CUDA);
    positionsVB->Release();
}

__global__ void createVertices(float4* positions, float time,
                               unsigned int width, unsigned int height)
{
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

    // 计算 UV 坐标
    float u = x / (float)width;
    float v = y / (float)height;
    u = u * 2.0f - 1.0f;
    v = v * 2.0f - 1.0f;

    // 计算简单的正弦波形
    float freq = 4.0f;
    float w = sinf(u * freq + time)
            * cosf(v * freq + time) * 0.5f;
            
    // 写入顶点位置
    positions[y * width + x] =
                make_float4(u, w, v, __int_as_float(0xff00ff00));
}
```
##### 3.2.12.2.3. Direct3D 11 Version
##### 3.2.12.2.3. D3D 11 版本

``` cuda
ID3D11Device* device;
struct CUSTOMVERTEX {
    FLOAT x, y, z;
    DWORD color;
};
ID3D11Buffer* positionsVB;
struct cudaGraphicsResource* positionsVB_CUDA;

int main()
{
    int dev;
    // 获得一个 CUDA-使能的适配器
    IDXGIFactory* factory;
    CreateDXGIFactory(__uuidof(IDXGIFactory), (void**)&factory);
    IDXGIAdapter* adapter = 0;
    for (unsigned int i = 0; !adapter; ++i) {
        if (FAILED(factory->EnumAdapters(i, &adapter))
            break;
        if (cudaD3D11GetDevice(&dev, adapter) == cudaSuccess)
            break;
        adapter->Release();
    }
    factory->Release();

    // 创建一个 swap chain 和 device
    ...
    sFnPtr_D3D11CreateDeviceAndSwapChain(adapter, 
                                         D3D11_DRIVER_TYPE_HARDWARE,
                                         0, 
                                         D3D11_CREATE_DEVICE_DEBUG,
                                         featureLevels, 3,
                                         D3D11_SDK_VERSION, 
                                         &swapChainDesc, &swapChain,
                                         &device,
                                         &featureLevel,
                                         &deviceContext);
    adapter->Release();

    // 使用相同 device
    cudaSetDevice(dev);

    // 创建顶点缓存并注册到 CUDA
    unsigned int size = width * height * sizeof(CUSTOMVERTEX);
    D3D11_BUFFER_DESC bufferDesc;
    bufferDesc.Usage          = D3D11_USAGE_DEFAULT;
    bufferDesc.ByteWidth      = size;
    bufferDesc.BindFlags      = D3D11_BIND_VERTEX_BUFFER;
    bufferDesc.CPUAccessFlags = 0;
    bufferDesc.MiscFlags      = 0;
    device->CreateBuffer(&bufferDesc, 0, &positionsVB);
    cudaGraphicsD3D11RegisterResource(&positionsVB_CUDA,
                                      positionsVB,
                                      cudaGraphicsRegisterFlagsNone);
    cudaGraphicsResourceSetMapFlags(positionsVB_CUDA,
                                    cudaGraphicsMapFlagsWriteDiscard);

    // 启动渲染循环
    while (...) {
        ...
        Render();
        ...
    }
    ...
}
```

``` cuda
void Render()
{
    // 映射顶点缓存并从 CUDA 写入
    float4* positions;
    cudaGraphicsMapResources(1, &positionsVB_CUDA, 0);
    size_t num_bytes; 
    cudaGraphicsResourceGetMappedPointer((void**)&positions,
                                         &num_bytes,  
                                         positionsVB_CUDA));

    // 执行 kernel
    dim3 dimBlock(16, 16, 1);
    dim3 dimGrid(width / dimBlock.x, height / dimBlock.y, 1);
    createVertices<<<dimGrid, dimBlock>>>(positions, time,
                                          width, height);

    // 逆映射顶点缓存
    cudaGraphicsUnmapResources(1, &positionsVB_CUDA, 0);

    // 绘制并显示
    ...
}

void releaseVB()
{
    cudaGraphicsUnregisterResource(positionsVB_CUDA);
    positionsVB->Release();
}

    __global__ void createVertices(float4* positions, float time,
                          unsigned int width, unsigned int height)
{
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

    // 计算 UV 坐标
    float u = x / (float)width;
    float v = y / (float)height;
    u = u * 2.0f - 1.0f;
    v = v * 2.0f - 1.0f;

    // 计算简单的正弦波形
    float freq = 4.0f;
    float w = sinf(u * freq + time)
            * cosf(v * freq + time) * 0.5f;

    // 写入顶点位置
    positions[y * width + x] =
                make_float4(u, w, v, __int_as_float(0xff00ff00));
}
```

#### 3.2.12.3. SLI Interoperability
#### 3.2.12.3. SLI 的互操作

> In a system with multiple GPUs, all CUDA-enabled GPUs are accessible via the CUDA driver and runtime as separate devices. There are however special considerations as described below when the system is in SLI mode. 

在多 GPU 系统下，所有激活 CUDA 的 GPU 都可以通过 CUDA 驱动和运行时作为单独的设备来访问。但是，当系统处于 SLI 模式下存在以下描述的特殊情况。

> First, an allocation in one CUDA device on one GPU will consume memory on other GPUs that are part of the SLI configuration of the Direct3D or OpenGL device. Because of this, allocations may fail earlier than otherwise expected. 

首先，在一个 GPU 的一个 CUDA 设备上进行分配，将会消耗其他作为 D3D 或 GL 设备的 SLI 配置的一部分的其他 GPU 的内存。因此，分配可能会比其他预期提前失败。

> Second, applications should create multiple CUDA contexts, one for each GPU in the SLI configuration. While this is not a strict requirement, it avoids unnecessary data transfers between devices. The application can use the cudaD3D[9|10|11]GetDevices() for Direct3D and cudaGLGetDevices() for OpenGL set of calls to identify the CUDA device handle(s) for the device(s) that are performing the rendering in the current and next frame. Given this information the application will typically choose the appropriate device and map Direct3D or OpenGL resources to the CUDA device returned by cudaD3D[9|10|11]GetDevices() or cudaGLGetDevices() when the deviceList parameter is set to cudaD3D[9|10|11]DeviceListCurrentFrame or cudaGLDeviceListCurrentFrame. 

其次，应用程序本该创建多个 CUDA 上下文，SLI 中的每个 GPU 一个。但是这并不是一个严格的要求，这样可以避免不必要的设备间的数据传输。应用程序可以分别为 D3D 和 GL 利用 *cudaD3D[9|10|11]GetDevices()* 和 *cudaGLGetDevices()* 形式的调用集合来区分为当前帧执行渲染和下一帧执行渲染的 CUDA 设备的句柄。鉴于这些信息，当 deviceList 参数被设置为 *cudaD3D[9|10|11]DeviceListCurrentFrame* 或 *cudaGLDeviceListCurrentFrame* 时，应用程序通常就可以选择出合适的设备并映射 D3D 或 GL 的资源到由 *cudaD3D[9|10|11]GetDevices()* 或 *cudaGLGetDevices()* 返回的 CUDA 设备上。

> Please note that resource returned from cudaGraphicsD9D[9|10|11]RegisterResource and cudaGraphicsGLRegister[Buffer|Image] must be only used on device the registration happened. Therefore on SLI configurations when data for different frames is computed on different CUDA devices it is necessary to register the resources for each separatly. 

请注意，从 *cudaGraphicsD9D[9|10|11]RegisterResource* 和 *cudaGraphicsGLRegister[Buffer|Image]* 返回的资源必须且仅能在注册的设备上使用。因此在 SLI 配置下，不同帧的数据是在不同 CUDA 设备计算得到的时候，必须为每个设备单独注册资源。

> See [link]() "Direct3D Interoperability" and [link]() "OpenGL Interoperability" for details on how the CUDA runtime interoperate with Direct3D and OpenGL, respectively. 

关于 CUDA 运行时分别如何和 D3D 和 GL 交互的细节，请参阅 [link]() "Direct3D Interoperability" 章节和 [link]() "OpenGL Interoperability" 章节。

## 3.3. Versioning and Compatibility
## 3.3. 版本和兼容性

> There are two version numbers that developers should care about when developing a CUDA application: The compute capability that describes the general specifications and features of the compute device (see Compute Capability) and the version of the CUDA driver API that describes the features supported by the driver API and runtime. 

当开发 CUDA 应用程序的时候，有两个版本号开发者需要关心：计算能力，它描述计算设备的一般规范和特性的（请参阅 [link]() “Compute Capability” 章节）； CUDA 驱动 API，它描述了驱动 API 和运行时所支持的特性。

> The version of the driver API is defined in the driver header file as CUDA_VERSION. It allows developers to check whether their application requires a newer device driver than the one currently installed. This is important, because the driver API is backward compatible, meaning that applications, plug-ins, and libraries (including the C runtime) compiled against a particular version of the driver API will continue to work on subsequent device driver releases as illustrated in Figure 11. The driver API is not forward compatible, which means that applications, plug-ins, and libraries (including the C runtime) compiled against a particular version of the driver API will not work on previous versions of the device driver. 

驱动 API 的版本被定义在名叫 *CUDA_VERSION* 的驱动头文件当中。它允许开发者检查他们的应用程序是否需要一个比当前安装的版本更新的设备驱动。这一点很重要，因为驱动 API 是向后兼容的，意味着依赖于特定的驱动 API 版本进行编译的应用程序、插件、库文件（包括 C 运行时）将可以继续在随后发布的设备驱动上运行，如 Figure 11 所示。而依赖于特定的驱动 API 版本进行编译的应用程序、插件、库文件（包括 C 运行时）在之前发布的设备驱动上将不可运行。

> It is important to note that there are limitations on the mixing and matching of versions that is supported: 

需要注意的是，支持的版本混合和匹配是有限制的:

> •Since only one version of the CUDA Driver can be installed at a time on a system, the installed driver must be of the same or higher version than the maximum Driver API version against which any application, plug-ins, or libraries that must run on that system were built. 

•因为在系统上只能安装一个版本的 CUDA 驱动程序，那么安装的驱动必须等于或高于那些先前编译且在该系统上必须运行的应用程序、插件、库文件所依赖的驱动 API 版本的最大值。

> •All plug-ins and libraries used by an application must use the same version of the CUDA Runtime unless they statically link to the Runtime, in which case multiple versions of the runtime can coexist in the same process space. Note that if nvcc is used to link the application, the static version of the CUDA Runtime library will be used by default, and all CUDA Toolkit libraries are statically linked against the CUDA Runtime. 

•应用程序使用的所有插件和库必须使用 CUDA 运行时的相同版本，除非它们静态地链接到运行时，在这种情况下，运行时的多个版本可以在相同的进程空间中共存。请注意，如果 nvcc 被用于链接应用程序，那么 CUDA 运行时库的静态版本将在默认情况下使用，所有 CUDA 工具包库都是静态链接到 CUDA 运行时。

> •All plug-ins and libraries used by an application must use the same version of any libraries that use the runtime (such as cuFFT, cuBLAS, ...) unless statically linking to those libraries. 

•应用程序使用的所有插件和库必须使用与运行时（如 *cuFFT*、*cuBLAS* 等。）相同的库，除非静态地链接到这些库。

> Figure 11. The Driver API Is Backward but Not Forward Compatible

Figure 11. 驱动 API 是后向兼容的，而不是前向兼容

![image](https://docs.nvidia.com/cuda/common/graphics/compatibility-of-cuda-versions.png)

---

## 3.4. Compute Modes
## 3.4. 计算模式

> On Tesla solutions running Windows Server 2008 and later or Linux, one can set any device in a system in one of the three following modes using NVIDIA's System Management Interface (nvidia-smi), which is a tool distributed as part of the driver: 

在运行 Windows Server 2008 和稍后或 Linux 的特斯拉解决方案中，你可以使用 NVIDIA 的系统管理接口（nvidia-smi，这是一种作为驱动程序的一部分分发的工具）在以下三种模式中的一种设置任何设备：

> •Default compute mode: Multiple host threads can use the device (by calling cudaSetDevice() on this device, when using the runtime API, or by making current a context associated to the device, when using the driver API) at the same time. 

•默认的计算模式：多个 host 线程可以同时使用设备（当使用运行时 API 时，通过在该设备上调用 *cudaSetDevice()* 或当使用驱动 API 时，把和当前设备相关的上下文设置为当前）。

> •Exclusive-process compute mode: Only one CUDA context may be created on the device across all processes in the system and that context may be current to as many threads as desired within the process that created that context. 

•独占进程的计算模式：在系统的所有进程中只有一个 CUDA 上下文能够在设备上被创建，在创建了该上下文的进程内可以使该上下文成为任意多的线程的当前上下文。

> •Exclusive-process-and-thread compute mode: Only one CUDA context may be created on the device across all processes in the system and that context may only be current to one thread at a time. 

•独占进程和线程的计算模式：在系统的所有进程中只有一个 CUDA 上下文能够在设备上被创建，该上下文一次只能成为一个线程的当前上下文。

> •Prohibited compute mode: No CUDA context can be created on the device. 

•禁止计算模式：在设备上不能创建 CUDA 上下文。

> This means, in particular, that a host thread using the runtime API without explicitly calling cudaSetDevice() might be associated with a device other than device 0 if device 0 turns out to be in the exclusive-process mode and used by another process, or in the exclusive-process-and-thread mode and used by another thread, or in prohibited mode. cudaSetValidDevices() can be used to set a device from a prioritized list of devices. 

这意味着，如果设备 0 是 exclusive-process 模式并且被另一个进程所使用,或者是 exclusive-process-and-thread 模式被另一个线程所使用，亦或者是禁止模式时，特别地，使用运行时 API 的 host 线程没有显式地调用 *cudaSetDevice()* 的话，可能会被关联到设备 0 以外的一个设备。*cudaSetValidDevices()* 可被用于从设备优先级队列中设置一个设备。

> Note also that, for devices featuring the Pascal architecture onwards (compute capability with major revision number 6 and higher), there exists support for Compute Preemption. This allows compute tasks to be preempted at instruction-level granularity, rather than thread block granularity as in prior Maxwell and Kepler GPU architecture, with the benefit that applications with long-running kernels can be prevented from either monopolizing the system or timing out. However, there will be context switch overheads associated with Compute Preemption, which is automatically enabled on those devices for which support exists. The individual attribute query function cudaDeviceGetAttribute() with the attribute cudaDevAttrComputePreemptionSupported can be used to determine if the device in use supports Compute Preemption. Users wishing to avoid context switch overheads associated with different processes can ensure that only one process is active on the GPU by selecting exclusive-process mode. 

还要注意的是，对于具有 Pascal 架构的设备（具有主要修订号 6 甚至更高版本的计算能力），存在对计算抢占的支持。这使得计算任务可以在指令级的粒度上抢占，而不是像之前的 Maxwell 和 Kepler GPU 架构那样的线程块粒度，这样就可以避免使用长时间运行的内核的应用程序导致的垄断系统或超时。但是，将会有与计算抢占相关的上下文切换开销，因为计算抢占会在支持的设备上自动启用。通过属性查询函数 *cudaDeviceGetAttribute()* 查询 *cudaDevAttrComputePreemptionSupported* 属性， 可以用来确定使用的设备是否支持计算抢占。希望避免与不同进程相关连的上下文切换的开销，可以通过选择 exclusive-process 模式，使得只有一个进程在 GPU 上是激活的。

> Applications may query the compute mode of a device by checking the computeMode device property (see Device Enumeration). 

应用程序可以通过检查 computeMode 设备属性来查询设备的计算模式（参阅 [link]() “Device Enumeration” 章节）

---

## 3.5. Mode Switches
## 3.5. 模式切换

> GPUs that have a display output dedicate some DRAM memory to the so-called primary surface, which is used to refresh the display device whose output is viewed by the user. When users initiate a mode switch of the display by changing the resolution or bit depth of the display (using NVIDIA control panel or the Display control panel on Windows), the amount of memory needed for the primary surface changes. For example, if the user changes the display resolution from 1280x1024x32-bit to 1600x1200x32-bit, the system must dedicate 7.68 MB to the primary surface rather than 5.24 MB. (Full-screen graphics applications running with anti-aliasing enabled may require much more display memory for the primary surface.) On Windows, other events that may initiate display mode switches include launching a full-screen DirectX application, hitting Alt+Tab to task switch away from a full-screen DirectX application, or hitting Ctrl+Alt+Del to lock the computer. 

GPU 将一些 DRAM 内存专门分配给所谓的 primary surface（刷新提供用户观看的显示设备）用于显示输出。当用户通过改变显示器的分辨率或位深度来启动显示器的模式切换时（使用 NVIDIA 控制面板或 Windows 上的显示控制面板），primary surface 所需的内存总量会发生变化。例如，如果用户将显示分辨率从 1280x1024x32 位改为 1600x1200x32-位，系统必须将 7.68 MB 的内存用于 primary surface，而不是 5.24 MB。（启用了抗锯齿的全屏幕图形应用程序可能需要更多的显示内存用于 primary surface。）在 Windows 上，其他可能启动显示模式切换的事件包括启动全屏幕 DirectX 应用程序，按 Alt+Tab 键从全屏 DirectX 应用程序中任务切出，或者按 Ctrl+Alt+Del 键锁定计算机。

> If a mode switch increases the amount of memory needed for the primary surface, the system may have to cannibalize memory allocations dedicated to CUDA applications. Therefore, a mode switch results in any call to the CUDA runtime to fail and return an invalid context error. 

如果模式切换增加了主表面所需的内存数量，系统可能不得不占用专用于 CUDA 应用程序的内存分配。

---

## 3.6. Tesla Compute Cluster Mode for Windows
## 3.6. 用于 Windows 的 Tesla 计算集群模式

> Using NVIDIA's System Management Interface (nvidia-smi), the Windows device driver can be put in TCC (Tesla Compute Cluster) mode for devices of the Tesla and Quadro Series of compute capability 2.0 and higher. 

使用 NVIDIA 的系统管理接口（nvidia-smi）可以将 Windows 的设备驱动程序放入 TCC（Tesla Compute Cluster） 模式，用于计算能力在 2.0 及更高的 Tesla 和 Quadro 系列设备上。因此，模式切换会导致 CUDA 运行时的任何调用失败并返回无效的上下文错误。

> This mode has the following primary benefits:  
•It makes it possible to use these GPUs in cluster nodes with non-NVIDIA integrated graphics;  
•It makes these GPUs available via Remote Desktop, both directly and via cluster management   systems that rely on Remote Desktop;  
•It makes these GPUs available to applications running as a Windows service (i.e., in Session 0). 

该模式具有以下主要的优点：  
•它使得在集群节点中使用非 NVIDIA 集成图形的 GPU 成为可能;
•既可以直接通过远程桌面又可以通过依赖于远程桌面的集群管理系统来实现这些 GPU 的可用性；
•它使这些 GPU 可被作为 Windows 服务运行的应用程序所用（即，会话 0 中）。  

> However, the TCC mode removes support for any graphics functionality. 

然而，TCC 模式移除了所有图形功能的支持。

