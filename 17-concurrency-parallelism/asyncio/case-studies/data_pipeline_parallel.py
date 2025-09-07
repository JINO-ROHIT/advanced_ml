import time
import asyncio
from concurrent.futures import ProcessPoolExecutor

from dataclasses import dataclass

@dataclass
class Item:
    file_id: int
    file_path: str
    data: str = None
    status: str = 'NA'

def _clean_data_heavy(item):
    """this cannot be a coroutine to be run in executor()"""
    time.sleep(1)
    return f"cleaned_{item}"

class Pipeline:
    def __init__(self, max_workers = 3):
        self.max_workers = max_workers
        self.executor = ProcessPoolExecutor(max_workers=max_workers)
    
    async def extract_data(self, item):
        await asyncio.sleep(1) # mock extraction
        return f"extracted_{item}"
    
    async def clean_data(self, item):
        loop = asyncio.get_running_loop()
        future = await loop.run_in_executor(self.executor, _clean_data_heavy, item)

        return future
    
    async def process_single_file(self, file_id, file_path, semaphore):

        async with semaphore:
            item = Item(file_id = file_id, file_path = file_path)

            item.data = await self.extract_data(item.data)
            item.data = await self.clean_data(item.data)

            return item
    
    async def process_files(self, files):
        semaphore = asyncio.Semaphore(self.max_workers)

        async with asyncio.TaskGroup() as tg:
            tasks = [ tg.create_task(
                        self.process_single_file(idx, file, semaphore)) 
                            for idx, file in enumerate(files) ]
        
        results = [task.result() for task in tasks]

        return results


async def main():
    test_files = [
        'document1.txt',
        'spreadsheet2.xlsx', 
        'report3.pdf',
        'data4.csv',
        'presentation5.pptx'
    ]

    pipe = Pipeline(max_workers = 5)

    t1 = time.perf_counter()
    results = await pipe.process_files(test_files * 100) # 500 files
    t2 = time.perf_counter()
    print(f"Finished in {t2 - t1:.2f} seconds")

if __name__ == '__main__':
    asyncio.run(main(), debug = True)

        


