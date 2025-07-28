"""In this script, lets look at how to fire and forget a task. In a task, you dont have to await unlike corountine"""

import asyncio

async def hello_world():
    await asyncio.sleep(2)
    print('hello from hello world, wait huh?')

async def yoink():
    asyncio.create_task(hello_world())
    print("yoink, im faster this time bc not wait!")

asyncio.run(yoink())

"""wait did you notice, the hello_world never printed? its because the run only waits for yoink to finish, so two ways to handle this -
1. add a delay - asyncio.sleep(3)
2. await the task
(Remember to comment out the previous code!)
"""

import asyncio

async def hello_world():
    await asyncio.sleep(2)
    print('hello from hello world, wait huh?')

async def yoink():
    task = asyncio.create_task(hello_world())
    print("yoink, im faster this time bc not wait!")
    await task

asyncio.run(yoink())
