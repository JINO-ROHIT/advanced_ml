import asyncio

async def hello_world(delay):
    await asyncio.sleep(delay)
    return f"hello after {delay}s"

async def yoink(msg):
    return f"yoink says: {msg}"

async def main():
    for coro in asyncio.as_completed([hello_world(2), yoink("im speed")]):
        result = await coro
        print(result) # now look at the results order, returned in the order of completion and not execution.

asyncio.run(main())