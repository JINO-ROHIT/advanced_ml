import asyncio

async def hello_world():
    await asyncio.sleep(1)
    print("yoink")


asyncio.run(hello_world())