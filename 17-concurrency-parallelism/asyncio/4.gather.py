"""how to run a group of tasks/coroutines together"""

import asyncio

async def hello_world():
    await asyncio.sleep(2)
    print('hello from hello world, wait huh?')

async def yoink():
    print("yoink, im faster this time bc not wait!")

async def main():
    group = asyncio.gather(hello_world(), yoink())
    print("this print statements shows it immediately returns")
    await group

asyncio.run(main())