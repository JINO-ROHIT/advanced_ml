### Concurrency and Parallelism

This section explores multi-threading, asyncio and multi-processing and how to scale beyond linear processing.

#### Multi-Threading 

- Handy for I/O bound activities like reading a file, writing to a database etc.
- Multiple threads can be running concurrently but not at the same time. Threads can never run at the same time!
- If you want to do heavy CPU computation work, switch to multiprocessing!

Thread Pool Executor

There are three important things you need with pool executor -
1. submit() - sends a function to be executed and returns a future, so it doesnot block.
2. map() - apply a function to an iterable of elements. Also happens without blocking.
3. shutdown() - shut down the executor.


#### Asyncio

- Super similar to multi-threading. (still has to obey GIL, locks stec).
- Instead of threads, you spawn coroutines in asyncio.
- Ideally would be best for IO operations and NOT CPU operations.
- Learning curve higher than multi-threading.
- Which is faster or better? Benchmark this for your task. ( Coroutines are lighter compared to threads)

