#include <tbb/parallel_for.h> // install this first
#include <algorithm>
#include <chrono>
#include <iterator>
#include <random>
#include <thread>
#include <vector>
#include <iostream>

int main() {
  std::random_device rd;
  std::mt19937 mt(rd());

  std::uniform_int_distribution<int> bin(20, 30);

  int num_work_items = 1 << 18;

  std::vector<int> work_items;
  std::generate_n(std::back_inserter(work_items), num_work_items,
                  [&] { return bin(mt); });

  auto start = std::chrono::high_resolution_clock::now();

  tbb::parallel_for(
      tbb::blocked_range<int>(0, num_work_items),
      [&](tbb::blocked_range<int> r) {
        for (int i = r.begin(); i < r.end(); i++) {
          std::this_thread::sleep_for(std::chrono::microseconds(work_items[i]));
        }
      },
      tbb::static_partitioner());

  auto end = std::chrono::high_resolution_clock::now();

  std::chrono::duration<double> duration = end - start;
  std::cout << "Total time: " << duration.count() << " seconds\n";

  return 0;
}
