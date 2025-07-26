### Concurrency and Parallelism

This section explores multi-threading and multi-processing and how to scale beyond linear processing.

1. Multi-Threading 

> for I/O bound activities like reading a file, writing to a database etc.
> threads can be running concurrently but not at the same time. Threads can never run at the same time!
> If you want to do heavy CPU computation work, switch to multiprocessing!