import asyncio

async def hello_world(delay):
    await asyncio.sleep(delay)
    return f"hello after {delay}s"

async def yoink(msg):
    return f"yoink says: {msg}"

async def main():
    group = asyncio.gather(
        hello_world(2), 
        yoink("i'm speed!") 
    )
    results = await group
    print("results:", results) # this returns results in the order of task created, not when its completed

asyncio.run(main())
