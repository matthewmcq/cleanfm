//
// Created by Matthew McQuistion on 3/6/25.
//

#ifndef THREADPOOL_H
#define THREADPOOL_H

#include <vector>
#include <complex>
#include <algorithm>
#include <thread>
#include <mutex>
#include <future>
#include <functional>
#include <cmath>
#include <iostream>
#include <queue>
#include <atomic>

// Structure to hold frequency band info
struct FrequencyBand {
    size_t start_bin;
    size_t end_bin;
};

// Structure to hold peak information
struct PeakInfo {
    int bin;
    double magnitude;
    double frequency;
    double phase;
    double amplitude;
    double spectral_centroid;
};

// Simplified thread pool implementation
class ThreadPool {
private:
    std::vector<std::thread> workers;
    std::queue<std::function<void()>> tasks;
    std::mutex queue_mutex;
    std::condition_variable condition;
    bool stop;

public:
    ThreadPool(size_t threads) : stop(false) {
        for (size_t i = 0; i < threads; ++i) {
            workers.emplace_back([this] {
                while (true) {
                    std::function<void()> task;
                    {
                        std::unique_lock<std::mutex> lock(queue_mutex);
                        condition.wait(lock, [this] { return stop || !tasks.empty(); });
                        if (stop && tasks.empty()) return;
                        task = std::move(tasks.front());
                        tasks.pop();
                    }
                    task();
                }
            });
        }
    }

    template<class F>
    void enqueue(F&& f) {
        {
            std::unique_lock<std::mutex> lock(queue_mutex);
            if (stop) throw std::runtime_error("enqueue on stopped ThreadPool");
            tasks.emplace(std::forward<F>(f));
        }
        condition.notify_one();
    }

    ~ThreadPool() {
        {
            std::unique_lock<std::mutex> lock(queue_mutex);
            stop = true;
        }
        condition.notify_all();
        for (std::thread &worker : workers) {
            worker.join();
        }
    }
};


#endif //THREADPOOL_H
