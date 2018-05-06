# 3.编程接口
```
CUDA C provides a simple path for users familiar with the C programming language to
easily write programs for execution by the device.
```
CUDA C为熟悉C编程语言的人提供了简单快捷的方法来写出可以在device上执行的程序。
```
It consists of a minimal set of extensions to the C language and a runtime library.
```
CUDA C包括一个C语言的最小扩展集和运行时库。
```
The core language extensions have been introduced in Programming Model. They allow
programmers to define a kernel as a C function and use some new syntax to specify the
grid and block dimension each time the function is called. A complete description of all
extensions can be found in C Language Extensions. Any source file that contains some of
these extensions must be compiled with nvcc as outlined in Compilation with NVCC.
```
核心的语言扩展在编程模型[link]章节中已经介绍过。这些扩展允许程序员像定义C函数一样定义一个kernel，以及使用一些新的语法来定义每次函数调用时的grid和block尺寸。完整的扩展描述可以参考C语言扩展章节[link]。如同使用NVCC编译章节[link]提到的，包含这些扩展的源文件必须用 **nvcc** 编译。
```
The runtime is introduced in Compilation Workflow. It provides C functions that
execute on the host to allocate and deallocate device memory, transfer data between host
memory and device memory, manage systems with multiple devices, etc. A complete
description of the runtime can be found in the CUDA reference manual.
```
Runtime在编译流程[link]中有相关介绍。它提供了一些在host端执行的C函数，用来分配和释放device memory，在host memory和device memory间传输数据，管理多device系统等等。Runtime的完整描述可以参考CUDA reference manual。
```
The runtime is built on top of a lower-level C API, the CUDA driver API, which is
also accessible by the application. The driver API provides an additional level of
control by exposing lower-level concepts such as CUDA contexts - the analogue of host
processes for the device - and CUDA modules - the analogue of dynamically loaded
libraries for the device. Most applications do not use the driver API as they do not
need this additional level of control and when using the runtime, context and module
management are implicit, resulting in more concise code. The driver API is introduced
in Driver API and fully described in the reference manual.
```
Runtime是建立在更底层的C API之上的，即CUDA driver API，它同样可以被应用程序访问。Driver API将更底层的概念暴露给用户从而提供额外的控制，诸如CUDA contexts（类似于device对应的host进程）、CUDA modules（类似于device上动态加载的库）。大多数应用不需要调用driver API，因为它们不需要这一层面的控制。当使用runtime的时候，context和module的管理是隐式的，代码更加简洁。Driver API在驱动API章节[link]介绍，完整的描述请参考reference manual。

## 3.1 使用NVCC编译
```
Kernels can be written using the CUDA instruction set architecture, called PTX, which
is described in the PTX reference manual. It is however usually more effective to use a
high-level programming language such as C. In both cases, kernels must be compiled
into binary code by nvcc to execute on the device.
```
Kernel可以用CUDA指令集架构编写，称为*PTX*，具体描述参见PTX reference manual。一般而言使用C这样的高级语言会更高效。使用任意一种方式来编写，都需要将kernel代码使用 **nvcc** 编译成二进制代码才能在device上执行。
```
nvcc is a compiler driver that simplifies the process of compiling C or PTX code: It
provides simple and familiar command line options and executes them by invoking the
collection of tools that implement the different compilation stages. This section gives an overview of nvcc workflow and command options. A complete description can be
found in the nvcc user manual.
```
**nvcc** 是一个编译器驱动，它简化了编译C代码或者PTX代码的过程。它提供了简单常用的命令行选项并通过调用了一系列实现不同编译阶段的工具来执行。这一章介绍 **nvcc** 的工作流和命令选项。完整的介绍请参照**nvcc** user manual。

### 3.1.1 编译流程
#### 3.1.1.1 离线编译
```
Source files compiled with nvcc can include a mix of host code (i.e., code that executes
on the host) and device code (i.e., code that executes on the device). nvcc's basic
workflow consists in separating device code from host code and then:
‣ compiling the device code into an assembly form (PTX code) and/or binary form
(cubin object),
‣ and modifying the host code by replacing the <<<...>>> syntax introduced in
Kernels (and described in more details in Execution Configuration) by the necessary
CUDA C runtime function calls to load and launch each compiled kernel from the
PTX code and/or cubin object.
```
使用 **nvcc** 编译的原文件可以包含host和device混合的代码（即在host执行的代码和在device执行的代码）。**nvcc** 的基本流程包含于一些操作中，先将device代码从host代码中分离出来，然后：  
‣ 将device代码编译成汇编的形式（*PTX*代码）和/或二进制形式（*cubin*对象），  
‣ 并修改host代码，将在kernel中引入的 **<<<...>>>** 语法（在执行配置[link]中有更详细的介绍）用必要的CUDA C runtime函数调用替换，从而从*PTX*代码和/或*cubin*对象来加载并调用编译好的kernel。
```
The modified host code is output either as C code that is left to be compiled using
another tool or as object code directly by letting nvcc invoke the host compiler during
the last compilation stage.
Applications can then:
‣ Either link to the compiled host code (this is the most common case),
‣ Or ignore the modified host code (if any) and use the CUDA driver API (see Driver
API) to load and execute the PTX code or cubin object.
```
修改后的host代码可以输出成C代码然后用其他工具编译，也可以直接让 **nvcc** 在最后编译阶段调用host端的编译器编译成object代码。  
然后应用程序可以：  
‣ 链接到编译好的host代码（一般是这种情形），  
‣ 或者忽略修改过后的host代码（如果有的话）并使用CUDA driver API 来加载和执行*PTX*代码或*cubin*对象。

#### 3.1.1.2 Just-in-Time编译
```
Any PTX code loaded by an application at runtime is compiled further to binary code
by the device driver. This is called just-in-time compilation. Just-in-time compilation
increases application load time, but allows the application to benefit from any new
compiler improvements coming with each new device driver. It is also the only way
for applications to run on devices that did not exist at the time the application was
compiled, as detailed in Application Compatibility.
```
应用程序在runtime加载的*PTX*代码会被device driver进一步地编译成二进制代码。这个过程成为*Just-in-time编译*。Just-in-time编译增加了应用程序加载的时间，但是这使得应用程序可以在每次device driver更新后受益于新的编译器改进。这也是让应用程序可以运行在编译时还不存在的未来的设备上的唯一方式，在应用兼容性[link]会有详细介绍。
```
When the device driver just-in-time compiles some PTX code for some application, it
automatically caches a copy of the generated binary code in order to avoid repeating
the compilation in subsequent invocations of the application. The cache - referred to as
compute cache - is automatically invalidated when the device driver is upgraded, so that
applications can benefit from the improvements in the new just-in-time compiler built
into the device driver.
```
当device driver通过just-in-time为应用程序编译了*PTX*代码后，为了避免在之后应用程序被调用的时候重复编译，会自动地缓存生成的二进制代码的备份。缓存（指的是计算缓存,*compute cache*）在device driver更新后会失效。因此应用可以从device driver集成的最新的just-in-time编译器的改进中获益。
```
Environment variables are available to control just-in-time compilation as described in
CUDA Environment Variables
```
可以通过环境变量来控制just-in-time编译，在环境变量[link]中会有详细介绍。

### 3.1.2 二进制兼容性
```
Binary code is architecture-specific. A cubin object is generated using the compiler
option -code that specifies the targeted architecture: For example, compiling with
-code=sm_35 produces binary code for devices of compute capability 3.5. Binary compatibility is guaranteed from one minor revision to the next one, but not from one
minor revision to the previous one or across major revisions. In other words, a cubin
object generated for compute capability X.y will only execute on devices of compute
capability X.z where z≥y.
```
二进制代码是跟架构相关的。通过编译器选项 **-code** 指定目标架构来生成*cubin*对象，例如，使用 **-code=sm_35** 可以生成适用于计算能力3.5的device的二进制代码。对于不同次版本号对应的二进制代码，二进制兼容性保证前向兼容，但不保证后向兼容，也不保证跨主版本号的兼容。也就是说，为计算能力*X.y*生成的*cubin*对象只能在计算能力*X.z*的设备上执行，这里*z≥y*。

### 3.1.3 PTX兼容性
```
Some PTX instructions are only supported on devices of higher compute capabilities.
For example, Warp Shuffle Functions are only supported on devices of compute
capability 3.0 and above. The -arch compiler option specifies the compute capability
that is assumed when compiling C to PTX code. So, code that contains warp shuffle, for
example, must be compiled with -arch=compute_30 (or higher).
```
有些*PTX*指令只被一些较高计算能力的device支持。例如，warp shuffle函数只被计算能力3.0及以上的device支持。编译器选项 **-arch** 假定了将C代码编译成*PTX*代码时的计算能力。因此，包含warp shuffle的代码，必须用 **-arch=compute_30** （或者更高的计算能力）编译。
```
PTX code produced for some specific compute capability can always be compiled to
binary code of greater or equal compute capability. Note that a binary compiled from an
earlier PTX version may not make use of some hardware features. For example, a binary
targeting devices of compute capability 7.0 (Volta) compiled from PTX generated for
compute capability 6.0 (Pascal) will not make use of Tensor Core instructions, since these were not available on Pascal. As a result, the final binary may perform worse than would be possible if the binary were generated using the latest version of PTX.
```
针对特定计算能力生成的*PTX*代码总是可以被编译成同等或者更高计算能力device对应的二进制代码。请注意，通过较早版本的*PTX*代码编译得到的二进制代码不一定能使用某些新的硬件特性。例如，*PTX*代码是为计算能力6.0（Pascal）生成的，使用该*PTX*代码为计算能力7.0（Volta）的device生成的二进制代码就不能使用Tensor Core指令，因为Pascal并不支持这个指令。这将会导致最终的二进制代码的性能不及使用最新的*PTX*代码生成的二进制代码。

### 3.1.4 应用兼容性
```
To execute code on devices of specific compute capability, an application must load
binary or PTX code that is compatible with this compute capability as described in
Binary Compatibility and PTX Compatibility. In particular, to be able to execute code
on future architectures with higher compute capability (for which no binary code can be
generated yet), an application must load PTX code that will be just-in-time compiled for
these devices (see Just-in-Time Compilation).
```
为了在特定计算能力的device上执行代码，应用程序必须加载兼容计算能力的二进制或者*PTX*代码，如二进制兼容性[link]和*PTX*兼容性[link]中所述。特别地，为了在拥有更高计算能力的未来的架构上也能执行代码，对应的二进制代码现在还不能够被生成，应用程序必须加载*PTX*代码通过just-in-time编译得到可执行代码（参照Just-in-Time编译[link]）。
```
Which PTX and binary code gets embedded in a CUDA C application is controlled by
the -arch and -code compiler options or the -gencode compiler option as detailed in
the nvcc user manual. For example,
```
通过 **-arch** 和 **-code** 编译器选项或者 **-gencode** 编译器选项（在 **nvcc** user manual中有详细介绍），可以控制在CUDA C应用程序中集成哪些*PTX*和二进制代码。例如，
```C
nvcc x.cu
    -gencode arch=compute_35,code=sm_35
    -gencode arch=compute_50,code=sm_50
    -gencode arch=compute_60,code=\'compute_60,sm_60\'
```
```
embeds binary code compatible with compute capability 3.5 and 5.0 (first and second
-gencode options) and PTX and binary code compatible with compute capability 6.0
(third -gencode option).
```
集成了兼容计算能力3.5和5.0的二进制代码（第一和第二个 **-gencode** 选项）和兼容计算能力6.0的*PTX*和二进制代码（第三个 **-gencode** 选项）。
```
Host code is generated to automatically select at runtime the most appropriate code to
load and execute, which, in the above example, will be:
‣ 3.5 binary code for devices with compute capability 3.5 and 3.7,
‣ 5.0 binary code for devices with compute capability 5.0 and 5.2,
‣ 6.0 binary code for devices with compute capability 6.0 and 6.1,
‣ PTX code which is compiled to binary code at runtime for devices with compute
capability 7.0 and higher.
```
生成的Host代码会在runtime自动地选择最合适的代码进行加载和执行，在上述例子中，选择  
‣ 计算能力3.5和3.7的device上，选择3.5的二进制代码，  
‣ 计算能力5.0和5.2的device上，选择5.0的二进制代码，  
‣ 计算能力6.0和6.1的device上，选择6.0的二进制代码，  
‣ 计算能力7.0及更高的device上，选择将*PTX*代码在runtime编译成二进制代码。
```
x.cu can have an optimized code path that uses warp shuffle operations, for example,
which are only supported in devices of compute capability 3.0 and higher. The
__CUDA_ARCH__ macro can be used to differentiate various code paths based on
compute capability. It is only defined for device code. When compiling with -
arch=compute_35 for example, __CUDA_ARCH__ is equal to 350.
```
举个例子，**x.cu** 使用warp shuffle操作能够得到最优的代码路径，而这些操作只被计算能力大于等于3.0的device支持。**\_\_CUDA_ARCH\_\_** 宏可以用来根据不同计算能力区分不同的代码路径。它只定义在device代码内。当使用 **-arch=compute_35** 编译的时候，**__CUDA_ARCH__** 等于**350**。
```
Applications using the driver API must compile code to separate files and explicitly load
and execute the most appropriate file at runtime.
```
调用driver API的应用必须将代码编译到单独的文件中，并且在runtime显式地加载和执行最合适的文件。
```
The Volta architecture introduces Independent Thread Scheduling which changes the
way threads are scheduled on the GPU. For code relying on specific behavior of SIMT
scheduling in previous architecures, Independent Thread Scheduling may alter the set of
participating threads, leading to incorrect results. To aid migration while implementing
the corrective actions detailed in Independent Thread Scheduling, Volta developers
can opt-in to Pascal's thread scheduling with the compiler option combination -
arch=compute_60 -code=sm_70.
```
Volta架构引入了*Independent Thread Scheduling*，用于控制thread在GPU上的调度方式。对于那些依赖于较早架构的SIMT调度[link]行为的代码来说，*Independent Thread Scheduling*可能会改变参与执行的thread集合，导致不正确的结果。为了帮助在迁移代码时实施Independent Thread Scheduling[link]中详细介绍的修正操作，Volta的开发人员可以选择性地使用Pascal架构的thread调度方法，使用编译器选项组合 **-arch=compute_60 -code=sm_70**。
```
The nvcc user manual lists various shorthand for the -arch, -code, and -gencode
compiler options. For example, -arch=sm_35 is a shorthand for -arch=compute_35 -
code=compute_35,sm_35 (which is the same as -gencode arch=compute_35,code=
\'compute_35,sm_35\').
```
**nvcc** user manual 列出了各种 **-arch**, **-code** 和 **-gencode** 编译器选项的缩写。比如， **-arch=sm_35** 是 **-arch=compute_35 -code=compute_35,sm_35** 的缩写（与 **-gencode arch=compute_35,code=\\'compute_35,sm_35\\'** 相同）。 

### 3.1.5 C/C++兼容性
```
The front end of the compiler processes CUDA source files according to C++ syntax
rules. Full C++ is supported for the host code. However, only a subset of C++ is fully
supported for the device code as described in C/C++ Language Support.
```
编译器前端根据C++语法处理CUDA源文件。host端代码有完整的C++支持。但是对于device代码，只有提供C++的一个子集的完整支持，在C/C++语言支持中有详细介绍[link]。

### 3.1.6 64位兼容性
```
The 64-bit version of nvcc compiles device code in 64-bit mode (i.e., pointers are 64-bit). Device code compiled in 64-bit mode is only supported with host code compiled in 64-bit mode.
```
64位的 **nvcc** 将device代码以64位模式编译，即指针是64位的。当host代码是64位模式编译的时候，64位模式编译的device代码才被支持。
```
Similarly, the 32-bit version of nvcc compiles device code in 32-bit mode and device
code compiled in 32-bit mode is only supported with host code compiled in 32-bit mode.
```
同样的，32位的 **nvcc** 将device代码以32位模式编译。只有当host代码是以32位模式编译的情况下，32位模式编译的device代码才被支持。
```
The 32-bit version of nvcc can compile device code in 64-bit mode also using the -m64
compiler option.
The 64-bit version of nvcc can compile device code in 32-bit mode also using the -m32
compiler option.
```
32位的 **nvcc** 通过 **-m64** 编译器选项同样可以编译64位的device代码。  
64位的 **nvcc** 通过 **-m32** 编译器选项同样可以编译32位的device代码。
