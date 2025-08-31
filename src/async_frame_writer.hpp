///// Otter: AsyncFrameWriter – asynchronous BMP24 saving; decouples disk IO from render thread.
///// Schneefuchs: Queue drops oldest on overflow (bounded latency); configurable max queue size.
///// Maus: ASCII logs only via LUCHS_LOG_HOST; no printf/fprintf in production path.
///// Datei: src/async_frame_writer.hpp

#pragma once
#include <cstdint>
#include <string>
#include <vector>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <atomic>
#include <chrono>
#include <fstream>

#include "luchs_log_host.hpp"

namespace otterdream {

class AsyncFrameWriter {
public:
    explicit AsyncFrameWriter(std::size_t maxQueuedJobs = 4);
    ~AsyncFrameWriter();

    void start();
    void stop();

    // Enqueue RGB8 frame; strideBytes >= width*3. Returns immediately (internal copy).
    void enqueue(const std::string& path, const uint8_t* rgb8, int width, int height, int strideBytes);

    std::size_t pending() const noexcept;

private:
    struct Job {
        std::string path;
        int width;
        int height;
        int stride;
        std::vector<uint8_t> pixels;
    };

    static bool writeBmp24(const std::string& path, const uint8_t* src, int w, int h, int stride);
    void workerLoop();

    mutable std::mutex mtx_;
    std::condition_variable cv_;
    std::queue<Job> queue_;
    std::size_t maxQueue_ = 4;
    std::atomic<bool> running_{false};
    std::thread worker_;
};

} // namespace otterdream
