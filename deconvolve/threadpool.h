/**
 * @file threadpool.h
 * @author Matthew McQuistion
 * @date 04/25/25
 * @brief Defines helper structures and a basic ThreadPool class for parallel processing.
 *
 * This file contains definitions for structures used in the DKD algorithm
 * (`FrequencyBand`, `PeakInfo`) and provides a simple thread pool implementation
 * (`ThreadPool`) to manage worker threads for parallel task execution.
 */

#ifndef THREADPOOL_H
#define THREADPOOL_H

#include <vector>
#include <complex> // Included but not directly used in this header's definitions
#include <algorithm> // Included but not directly used
#include <thread>
#include <mutex>
#include <condition_variable> // For thread synchronization
#include <future> // Included but not directly used
#include <functional> // For std::function
#include <cmath> // Included but not directly used
#include <iostream> // Included but not directly used
#include <queue> // For task queue
#include <atomic> // Included but not directly used
#include <stdexcept> // For runtime_error

// Forward declaration or include constants if Complex type is needed here
// Assuming Complex is defined elsewhere (e.g., constants.h)
// using Complex = std::complex<double>; // Example if not defined elsewhere

/**
 * @brief Structure to define a frequency range (band) in terms of DFT bins.
 * Used for dividing the spectrum for parallel processing.
 */
struct FrequencyBand {
    size_t start_bin; ///< The starting bin index of the frequency band (inclusive).
    size_t end_bin;   ///< The ending bin index of the frequency band (inclusive or exclusive, depending on usage).
};

/**
 * @brief Structure to hold detailed information about a detected spectral peak.
 * Used during the peak finding and component estimation phases.
 */
struct PeakInfo {
    int bin;                  ///< The integer DFT bin index where the peak maximum was initially detected.
    double magnitude;         ///< The magnitude measured at the `bin`.
    double frequency;         ///< The estimated true frequency of the peak (potentially fractional bin value).
    double phase;             ///< The estimated true phase of the peak (in radians) at its `frequency`.
    double amplitude;         ///< The estimated true amplitude of the underlying component.
    double spectral_centroid; ///< Optional: A measure of the peak's frequency distribution (calculated if needed).
};

/**
 * @brief A basic thread pool implementation for managing worker threads.
 *
 * This class creates a fixed number of worker threads upon construction.
 * Tasks (as `std::function<void()>`) can be enqueued, and worker threads
 * will pick them up and execute them concurrently. The pool handles thread
 * synchronization using a mutex and condition variable.
 */
class ThreadPool {
private:
    std::vector<std::thread> workers;             ///< Vector holding the worker thread objects.
    std::queue<std::function<void()>> tasks;      ///< Queue storing tasks waiting to be executed.
    std::mutex queue_mutex;                       ///< Mutex to protect access to the tasks queue.
    std::condition_variable condition;            ///< Condition variable to signal workers about new tasks or shutdown.
    std::atomic<bool> stop;                       ///< Atomic flag to signal threads to stop processing.

public:
    /**
     * @brief Constructs the ThreadPool and starts the worker threads.
     *
     * Creates the specified number of worker threads. Each thread runs a loop,
     * waiting for tasks to appear in the queue or for the stop signal.
     *
     * @param threads The number of worker threads to create in the pool.
     */
    explicit ThreadPool(size_t threads) : stop(false) {
        workers.reserve(threads); // Reserve space for efficiency
        for (size_t i = 0; i < threads; ++i) {
            // Add a new worker thread to the vector
            workers.emplace_back([this] { // Lambda function executed by each worker thread
                while (true) { // Worker loop
                    std::function<void()> task; // Variable to hold the task function

                    // --- Critical Section: Accessing the queue ---
                    {
                        std::unique_lock<std::mutex> lock(this->queue_mutex); // Lock the queue mutex

                        // Wait until notified AND (stop is true OR tasks queue is not empty)
                        this->condition.wait(lock, [this] { return this->stop.load() || !this->tasks.empty(); });

                        // If stop is signaled and the queue is empty, the thread can exit
                        if (this->stop.load() && this->tasks.empty()) {
                            return; // Exit the worker loop
                        }

                        // If tasks are available, get the next task
                        task = std::move(this->tasks.front()); // Move task from queue
                        this->tasks.pop(); // Remove task from queue
                    } // Mutex lock released here

                    // --- Execute the Task ---
                    // Execute the retrieved task outside the lock
                    try {
                         task();
                    } catch (const std::exception& e) {
                         std::cerr << "ThreadPool worker caught exception: " << e.what() << std::endl;
                         // Decide how to handle task exceptions (log, ignore, etc.)
                    } catch (...) {
                         std::cerr << "ThreadPool worker caught unknown exception." << std::endl;
                    }

                } // End worker loop
            }); // End lambda function
        } // End for loop creating threads
    }

    /**
     * @brief Enqueues a new task to be executed by a worker thread.
     *
     * Adds the given function (or callable object) to the task queue and notifies
     * one waiting worker thread.
     *
     * @tparam F The type of the callable object (e.g., function pointer, lambda, functor).
     * @param f The callable object representing the task to execute.
     * @throws std::runtime_error if enqueue is called on a stopped ThreadPool.
     */
    template<class F>
    void enqueue(F&& f) {
        // --- Critical Section: Adding to the queue ---
        {
            std::unique_lock<std::mutex> lock(queue_mutex); // Lock the queue mutex

            // Don't allow enqueueing after stopping the pool
            if (stop.load()) {
                throw std::runtime_error("enqueue on stopped ThreadPool");
            }

            // Add the task to the queue using perfect forwarding
            tasks.emplace(std::forward<F>(f));
        } // Mutex lock released here

        // Notify one waiting worker thread that a new task is available
        condition.notify_one();
    }

    /**
     * @brief Destructor for the ThreadPool.
     *
     * Signals all worker threads to stop, notifies them via the condition variable,
     * and then joins each worker thread to ensure they complete execution before
     * the pool object is destroyed.
     */
    ~ThreadPool() {
        // --- Signal Stop ---
        {
            std::unique_lock<std::mutex> lock(queue_mutex); // Lock the queue mutex
            stop.store(true); // Set the stop flag
        } // Mutex lock released here

        // Notify all waiting threads to wake up and check the stop condition
        condition.notify_all();

        // --- Join Workers ---
        // Wait for each worker thread to finish execution
        for (std::thread &worker : workers) {
            if (worker.joinable()) { // Check if the thread is joinable
                worker.join(); // Wait for the thread to complete
            }
        }
    }

    // Prevent copying and assignment
    ThreadPool(const ThreadPool&) = delete;
    ThreadPool& operator=(const ThreadPool&) = delete;
    ThreadPool(ThreadPool&&) = delete;
    ThreadPool& operator=(ThreadPool&&) = delete;
};


#endif // THREADPOOL_H