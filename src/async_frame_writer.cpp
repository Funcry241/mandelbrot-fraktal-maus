///// Otter: AsyncFrameWriter implementation – measurable, deterministic, robust failure logs.
///// Schneefuchs: Bottom-up BMP rows; row padding; single pass write; bounded queue with drop-oldest.
///// Maus: No functional surprises; behavior change (async save) is explicit and logged; ASCII-only.
///// Datei: src/async_frame_writer.cpp

#include "pch.hpp"
#include "async_frame_writer.hpp"
#include <cstring> // memcpy

namespace otterdream {

AsyncFrameWriter::AsyncFrameWriter(std::size_t maxQueuedJobs)
    : maxQueue_(maxQueuedJobs ? maxQueuedJobs : 1) {}

AsyncFrameWriter::~AsyncFrameWriter() {
    stop();
}

void AsyncFrameWriter::start() {
    bool expected = false;
    if (running_.compare_exchange_strong(expected, true)) {
        worker_ = std::thread(&AsyncFrameWriter::workerLoop, this);
        LUCHS_LOG_HOST("[AsyncFrameWriter] started; maxQueue=%zu", maxQueue_);
    }
}

void AsyncFrameWriter::stop() {
    bool expected = true;
    if (running_.compare_exchange_strong(expected, false)) {
        {
            std::lock_guard<std::mutex> lock(mtx_);
            cv_.notify_all();
        }
        if (worker_.joinable()) worker_.join();
        LUCHS_LOG_HOST("[AsyncFrameWriter] stopped; remaining=%zu", queue_.size());
    }
}

void AsyncFrameWriter::enqueue(const std::string& path, const uint8_t* rgb8, int width, int height, int strideBytes) {
    if (!running_.load()) {
        LUCHS_LOG_HOST("[AsyncFrameWriter] enqueue ignored; writer not running");
        return;
    }
    if (!rgb8 || width <= 0 || height <= 0 || strideBytes < width * 3) {
        LUCHS_LOG_HOST("[AsyncFrameWriter] enqueue invalid args; path=%s w=%d h=%d stride=%d",
                       path.c_str(), width, height, strideBytes);
        return;
    }

    std::unique_lock<std::mutex> lock(mtx_);
    if (queue_.size() >= maxQueue_) {
        LUCHS_LOG_HOST("[AsyncFrameWriter] queue full -> dropping oldest (size=%zu)", queue_.size());
        queue_.pop();
    }

    Job j;
    j.path   = path;
    j.width  = width;
    j.height = height;
    j.stride = strideBytes;
    j.pixels.resize(static_cast<std::size_t>(height) * static_cast<std::size_t>(strideBytes));
    std::memcpy(j.pixels.data(), rgb8, j.pixels.size());

    queue_.push(std::move(j));
    cv_.notify_one();
}

std::size_t AsyncFrameWriter::pending() const noexcept {
    std::lock_guard<std::mutex> lock(mtx_);
    return queue_.size();
}

void AsyncFrameWriter::workerLoop() {
    while (running_.load()) {
        Job job;
        {
            std::unique_lock<std::mutex> lock(mtx_);
            cv_.wait(lock, [&] { return !running_.load() || !queue_.empty(); });
            if (!running_.load() && queue_.empty()) break;
            job = std::move(queue_.front());
            queue_.pop();
        }

        const auto t0 = std::chrono::high_resolution_clock::now();
        const bool ok = writeBmp24(job.path, job.pixels.data(), job.width, job.height, job.stride);
        const auto t1 = std::chrono::high_resolution_clock::now();
        const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

        if (ok) {
            LUCHS_LOG_HOST("[AsyncFrameWriter] saved file=%s w=%d h=%d stride=%d time=%.3f ms",
                           job.path.c_str(), job.width, job.height, job.stride, ms);
        } else {
            LUCHS_LOG_HOST("[AsyncFrameWriter] FAILED to save file=%s w=%d h=%d stride=%d time=%.3f ms",
                           job.path.c_str(), job.width, job.height, job.stride, ms);
        }
    }
}

// Minimal BMP writer for RGB8; bottom-up rows; 24 bpp; uncompressed.
bool AsyncFrameWriter::writeBmp24(const std::string& path, const uint8_t* src, int w, int h, int stride) {
    const int bytesPerPixel = 3;
    const int rowOut = ((w * bytesPerPixel + 3) / 4) * 4;
    const uint32_t fileHeaderSize = 14;
    const uint32_t infoHeaderSize = 40;
    const uint32_t dataSize = rowOut * h;
    const uint32_t fileSize = fileHeaderSize + infoHeaderSize + dataSize;

    std::ofstream f(path, std::ios::binary);
    if (!f.is_open()) {
        LUCHS_LOG_HOST("[AsyncFrameWriter] cannot open path=%s", path.c_str());
        return false;
    }

    unsigned char fileHeader[14] = {
        'B','M',
        (unsigned char)(fileSize      ),
        (unsigned char)(fileSize >> 8 ),
        (unsigned char)(fileSize >> 16),
        (unsigned char)(fileSize >> 24),
        0,0, 0,0,
        (unsigned char)((fileHeaderSize + infoHeaderSize)      ),
        (unsigned char)((fileHeaderSize + infoHeaderSize) >> 8 ),
        (unsigned char)((fileHeaderSize + infoHeaderSize) >> 16),
        (unsigned char)((fileHeaderSize + infoHeaderSize) >> 24)
    };
    f.write(reinterpret_cast<char*>(fileHeader), sizeof(fileHeader));

    unsigned char infoHeader[40] = {0};
    infoHeader[ 0] = (unsigned char)(infoHeaderSize);
    infoHeader[ 4] = (unsigned char)( w       );
    infoHeader[ 5] = (unsigned char)( w >> 8  );
    infoHeader[ 6] = (unsigned char)( w >> 16 );
    infoHeader[ 7] = (unsigned char)( w >> 24 );
    infoHeader[ 8] = (unsigned char)( h       );
    infoHeader[ 9] = (unsigned char)( h >> 8  );
    infoHeader[10] = (unsigned char)( h >> 16 );
    infoHeader[11] = (unsigned char)( h >> 24 );
    infoHeader[12] = 1;   // planes
    infoHeader[14] = 24;  // bpp

    f.write(reinterpret_cast<char*>(infoHeader), sizeof(infoHeader));

    std::vector<unsigned char> row(static_cast<std::size_t>(rowOut));
    for (int y = h - 1; y >= 0; --y) {
        const uint8_t* srcRow = src + static_cast<std::size_t>(y) * static_cast<std::size_t>(stride);
        unsigned char* dst = row.data();
        for (int x = 0; x < w; ++x) {
            const unsigned char r = srcRow[x*3 + 0];
            const unsigned char g = srcRow[x*3 + 1];
            const unsigned char b = srcRow[x*3 + 2];
            *dst++ = b; *dst++ = g; *dst++ = r;
        }
        const int pad = rowOut - w * bytesPerPixel;
        for (int p = 0; p < pad; ++p) *dst++ = 0;

        f.write(reinterpret_cast<char*>(row.data()), rowOut);
        if (!f.good()) {
            LUCHS_LOG_HOST("[AsyncFrameWriter] write error at y=%d", y);
            return false;
        }
    }
    return true;
}

} // namespace otterdream
