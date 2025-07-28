""" in this script we will see yoink() gets executed first, but since its awaited hello_world, it waits for it to run to completion"""

import asyncio

async def hello_world():
    await asyncio.sleep(2)
    print('hello from hello world, wait huh?')

async def yoink():
    await hello_world()
    print("yoink")

asyncio.run(yoink())

