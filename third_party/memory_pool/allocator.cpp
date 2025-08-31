#include "allocator.h"

#include <cstdlib>
#include <iostream>

// #define OUT_PUT_MEMORY_POOL_LOG

#ifdef OUT_PUT_MEMORY_POOL_LOG
#define LOG_MP_INFO(x)                                                  \
  std::cout << std::setprecision(15) << "[INFO] [" << __FILE__ << "] [" \
            << __func__ << "] [" << __LINE__ << "] " << x << std::endl;
#define LOG_MP_ERROR(x)                                                  \
  std::cout << std::setprecision(15) << "[ERROR] [" << __FILE__ << "] [" \
            << __func__ << "] [" << __LINE__ << "] " << x << std::endl;
#define LOG_MP_WARNING(x)                                                  \
  std::cout << std::setprecision(15) << "[WARNING] [" << __FILE__ << "] [" \
            << __func__ << "] [" << __LINE__ << "] " << x << std::endl;
#else
#define LOG_MP_INFO(x)
#define LOG_MP_ERROR(x)
#define LOG_MP_WARNING(x)
#endif

#define LOG_MP_FATAL(x)                                                  \
  std::cout << std::setprecision(15) << "[FATAL] [" << __FILE__ << "] [" \
            << __func__ << "] [" << __LINE__ << "] " << x << std::endl;

#undef OUT_PUT_MEMORY_POOL_LOG

#define BLOCK_SIZE sizeof(struct Block)
#define BLOCK_POINTER_SIZE sizeof(struct Block *)

#define MP_LOCK(flag, lock_obj)                    \
  do {                                             \
    if (flag) pthread_mutex_lock(&lock_obj->lock); \
  } while (0)
#define MP_UNLOCK(flag, lock_obj)                    \
  do {                                               \
    if (flag) pthread_mutex_unlock(&lock_obj->lock); \
  } while (0)

#define MP_ALIGN_SIZE(_n) ((_n) + sizeof(long) - ((sizeof(long) - 1) & (_n)))

#define MP_INIT_MEMORY_STRUCT(mm, mem_sz)        \
  do {                                           \
    mm->mem_pool_size = mem_sz;                  \
    mm->allow_mem_ = 0;                          \
    mm->alloc_program_mem_ = 0;                  \
    mm->free_block_list_ = (Block *)mm->start;   \
    mm->free_block_list_->is_free_ = 1;          \
    mm->free_block_list_->allow_mem_ = mem_sz;   \
    mm->free_block_list_->prev_block_ = nullptr; \
    mm->free_block_list_->next_block_ = nullptr; \
    mm->alloc_block_list_ = nullptr;             \
  } while (0)

#define MP_DLINKLIST_INS_FRT(head, x) \
  do {                                \
    x->prev_block_ = nullptr;         \
    x->next_block_ = head;            \
    if (head) head->prev_block_ = x;  \
    head = x;                         \
  } while (0)

#define MP_DLINKLIST_DEL(head, x)                                       \
  do {                                                                  \
    if (!x->prev_block_) {                                              \
      head = x->next_block_;                                            \
      if (x->next_block_) x->next_block_->prev_block_ = nullptr;        \
    } else {                                                            \
      x->prev_block_->next_block_ = x->next_block_;                     \
      if (x->next_block_) x->next_block_->prev_block_ = x->prev_block_; \
    }                                                                   \
  } while (0)

namespace Autoalg {
namespace Allocator {

using ChunkId = unsigned long;
constexpr MemorySize KB = (MemorySize)(MemorySize(1) << 10);
constexpr MemorySize MB = (MemorySize)(MemorySize(1) << 20);
constexpr MemorySize GB = (MemorySize)(MemorySize(1) << 30);
constexpr MemorySize TB = (MemorySize)(MemorySize(1) << 40);

struct Block {
  MemorySize allow_mem_;
  Block *prev_block_;
  Block *next_block_;
  int is_free_;
};

struct Chunk {
  char *start;
  ChunkId id;
  MemorySize mem_pool_size;
  MemorySize allow_mem_;
  MemorySize alloc_program_mem_;
  Block *free_block_list_;
  Block *alloc_block_list_;
  struct Chunk *next_chunk_;
  template <typename Stream>
  friend Stream &operator<<(Stream &stream, const Chunk &chunk) {
    stream << std::endl;
    stream << "Chunk:" << std::endl;
    stream << "Start address:[" << (int *)chunk.start << "]" << std::endl;
    stream << "Chunk id:[" << chunk.id << "]" << std::endl;
    stream << "Memory pool size:[" << chunk.mem_pool_size << "]" << std::endl;
    stream << "Allow memory size:[" << chunk.allow_mem_ << "]" << std::endl;
    stream << "Allow program memory size:[" << chunk.alloc_program_mem_ << "]"
           << std::endl;
    stream << "Next chunk address:[" << (int *)chunk.next_chunk_ << "]";
    return stream;
  }
};

struct MemoryPool {
  ChunkId last_chunk_id_;
  int is_allow_extend_;
  int is_thread_safe_;
  MemorySize mem_pool_size;
  MemorySize total_mem_pool_size;
  MemorySize max_mem_pool_size;
  struct Chunk *chunk_list_;

  template <typename Stream>
  friend Stream &operator<<(Stream &stream, const MemoryPool &mp) {
    stream << std::endl;
    stream << "Memory pool:" << std::endl;
    stream << "Last chunk id:[" << mp.last_chunk_id_ << "]" << std::endl;
    stream << "Is allow extend:[" << mp.is_allow_extend_ << "]" << std::endl;
    stream << "Memory pool size:[" << mp.mem_pool_size << "]" << std::endl;
    stream << "Total memory pool size:[" << mp.total_mem_pool_size << "]"
           << std::endl;
    stream << "Max memory pool size:[" << mp.max_mem_pool_size << "]";
    return stream;
  }
};
class FirstFitAllocator {
 public:
  MemorySize GetMemoryListCount(MemoryPool *mp) {
    MemorySize result = 0;
    Chunk *mm = mp->chunk_list_;
    while (mm) {
      result++;
      mm = mm->next_chunk_;
    }
    return result;
  }
  std::pair<MemorySize, MemorySize> GetMemoryInfo(MemoryPool *mp, Chunk *mm) {
    MemorySize free_n = 0;
    MemorySize alloc_n = 0;
    Block *p = mm->free_block_list_;
    while (p) {
      free_n++;
      p = p->next_block_;
    }
    p = mm->alloc_block_list_;
    while (p) {
      alloc_n++;
      p = p->next_block_;
    }
    return std::make_pair(free_n, alloc_n);
  }

  static MemoryPool *MemoryPoolInit(MemorySize max_mp_size, MemorySize mp_size,
                                    int thread_safe) {
    LOG_MP_INFO("Init memory pool max size:[" << max_mp_size << "]");
    LOG_MP_INFO("Current need memory pool size:[" << mp_size << "]");
    if (mp_size > max_mp_size) {
      return nullptr;
    }
    auto *mp = (MemoryPool *)malloc(sizeof(MemoryPool));
    if (!mp) {
      return nullptr;
    }
    mp->last_chunk_id_ = 0;
    if (mp_size < max_mp_size) {
      mp->is_allow_extend_ = 1;
    }
    mp->is_thread_safe_ = thread_safe;
    mp->max_mem_pool_size = max_mp_size;
    mp->mem_pool_size = mp_size;
    mp->total_mem_pool_size = mp_size;
    char *s = (char *)malloc(sizeof(Chunk) + sizeof(char) * mp->mem_pool_size);
    if (!s) {
      return nullptr;
    }
    mp->chunk_list_ = (Chunk *)s;
    mp->chunk_list_->start = s + sizeof(Chunk);
    MP_INIT_MEMORY_STRUCT(mp->chunk_list_, mp->mem_pool_size);
    mp->chunk_list_->next_chunk_ = nullptr;
    mp->chunk_list_->id = mp->last_chunk_id_++;
    LOG_MP_INFO(*mp);
    return mp;
  }

  static void *MemoryPoolAlloc(MemoryPool *mp, MemorySize need_size) {
    MemorySize total_needed_size =
        MP_ALIGN_SIZE(need_size + BLOCK_SIZE + BLOCK_POINTER_SIZE);
    if (total_needed_size > mp->mem_pool_size) {
      return nullptr;
    }

    Chunk *current_chunk = nullptr;
    Block *current_free_block_ = nullptr;
    Block *current_not_free_block_ = nullptr;

  FIND_FREE_CHUNK:
    current_chunk = mp->chunk_list_;
    while (current_chunk) {
      if (mp->mem_pool_size - current_chunk->allow_mem_ < total_needed_size) {
        current_chunk = current_chunk->next_chunk_;
        continue;
      }

      current_free_block_ = current_chunk->free_block_list_;
      current_not_free_block_ = nullptr;

      while (current_free_block_) {
        if (current_free_block_->allow_mem_ >= total_needed_size) {
          // 如果free块分割后剩余内存足够大 则进行分割
          if (current_free_block_->allow_mem_ - total_needed_size >
              BLOCK_SIZE + BLOCK_POINTER_SIZE) {
            // 从free块头开始分割出alloc块
            current_not_free_block_ = current_free_block_;

            current_free_block_ =
                (Block *)((char *)current_not_free_block_ + total_needed_size);
            *current_free_block_ = *current_not_free_block_;
            current_free_block_->allow_mem_ -= total_needed_size;
            *(Block **)((char *)current_free_block_ +
                        current_free_block_->allow_mem_ - BLOCK_POINTER_SIZE) =
                current_free_block_;

            // update free_block_list_
            if (!current_free_block_->prev_block_) {
              current_chunk->free_block_list_ = current_free_block_;
              if (current_free_block_->next_block_) {
                current_free_block_->next_block_->prev_block_ =
                    current_free_block_;
              }
            } else {
              current_free_block_->prev_block_->next_block_ =
                  current_free_block_;
              if (current_free_block_->next_block_) {
                current_free_block_->next_block_->prev_block_ =
                    current_free_block_;
              }
            }

            current_not_free_block_->is_free_ = 0;
            current_not_free_block_->allow_mem_ = total_needed_size;

            *(Block **)((char *)current_not_free_block_ + total_needed_size -
                        BLOCK_POINTER_SIZE) = current_not_free_block_;
          } else {
            current_not_free_block_ = current_free_block_;
            MP_DLINKLIST_DEL(current_chunk->free_block_list_,
                             current_not_free_block_);
            current_not_free_block_->is_free_ = 0;
          }
          MP_DLINKLIST_INS_FRT(current_chunk->alloc_block_list_,
                               current_not_free_block_);

          current_chunk->allow_mem_ += current_not_free_block_->allow_mem_;
          current_chunk->alloc_program_mem_ +=
              (current_not_free_block_->allow_mem_ - BLOCK_SIZE -
               BLOCK_POINTER_SIZE);

          return (void *)((char *)current_not_free_block_ + BLOCK_SIZE);
        }
        current_free_block_ = current_free_block_->next_block_;
      }
      current_chunk = current_chunk->next_chunk_;
    }
    if (mp->is_allow_extend_) {
      if (mp->total_mem_pool_size >= mp->max_mem_pool_size) {
        std::cerr << "Total memory size has great max memory limit."
                  << std::endl;
      }
      MemorySize add_mem_sz = mp->mem_pool_size;
      if (!ExtendChunkList(mp, add_mem_sz)) {
        return nullptr;
      }
      mp->total_mem_pool_size += add_mem_sz;
      goto FIND_FREE_CHUNK;
    }
    return nullptr;
  }

  static bool MemoryPoolFree(MemoryPool *mp, void *p) {
    if (p == nullptr || mp == nullptr) {
      return false;
    }
    Chunk *chunk = mp->chunk_list_;
    if (mp->is_allow_extend_) {
      chunk = find_memory_list(mp, p);
    }
    LOG_MP_INFO(*chunk);

    Block *ck = (Block *)((char *)p - BLOCK_SIZE);

    MP_DLINKLIST_DEL(chunk->alloc_block_list_, ck);
    MP_DLINKLIST_INS_FRT(chunk->free_block_list_, ck);
    ck->is_free_ = 1;

    chunk->allow_mem_ -= ck->allow_mem_;
    chunk->alloc_program_mem_ -=
        (ck->allow_mem_ - BLOCK_SIZE - BLOCK_POINTER_SIZE);

    MergeFreeChunk(mp, chunk, ck);
    return true;
  }

  static MemoryPool *MemoryPoolClear(MemoryPool *mp) {
    if (!mp) {
      return nullptr;
    }
    Chunk *mm = mp->chunk_list_;
    while (mm) {
      MP_INIT_MEMORY_STRUCT(mm, mm->mem_pool_size);
      mm = mm->next_chunk_;
    }
    return mp;
  }

  static int MemoryPoolDestroy(MemoryPool *mp) {
    if (mp == nullptr) {
      return 1;
    }
    Chunk *mm = mp->chunk_list_, *mm1 = nullptr;
    while (mm) {
      mm1 = mm;
      mm = mm->next_chunk_;
      free(mm1);
    }
    free(mp);

    return 0;
  }

  static int MemoryPoolSetThreadSafe(MemoryPool *mp, int thread_safe) {
    if (mp->is_thread_safe_) {
      mp->is_thread_safe_ = thread_safe;
    } else {
      mp->is_thread_safe_ = thread_safe;
    }
    return 0;
  }
  static double GetMemPoolUsage(MemoryPool *mp) {
    return (double)GetUsedMemory(mp) / mp->total_mem_pool_size;
  }

  static double GetMemPoolProgramUsage(MemoryPool *mp) {
    return (double)GetProgramMemory(mp) / mp->total_mem_pool_size;
  }

 private:
  static Chunk *ExtendChunkList(MemoryPool *mp, MemorySize new_mem_sz) {
    char *s = (char *)malloc(sizeof(Chunk) + new_mem_sz * sizeof(char));
    if (!s) {
      return nullptr;
    }

    Chunk *mm = (Chunk *)s;
    mm->start = s + sizeof(Chunk);

    MP_INIT_MEMORY_STRUCT(mm, new_mem_sz);
    mm->id = mp->last_chunk_id_++;
    mm->next_chunk_ = mp->chunk_list_;
    mp->chunk_list_ = mm;
    return mm;
  }

  static Chunk *find_memory_list(MemoryPool *mp, void *p) {
    Chunk *tmp = mp->chunk_list_;
    while (tmp) {
      if (tmp->start <= (char *)p &&
          tmp->start + mp->mem_pool_size > (char *)p) {
        break;
      }
      tmp = tmp->next_chunk_;
    }

    return tmp;
  }

  static void MergeFreeChunk(MemoryPool *mp, Chunk *mm, Block *c) {
    Block *p0 = c, *p1 = c;
    while (p0->is_free_) {
      p1 = p0;
      if ((char *)p0 - BLOCK_POINTER_SIZE - BLOCK_SIZE <= mm->start) {
        break;
      }
      p0 = *(Block **)((char *)p0 - BLOCK_POINTER_SIZE);
    }

    p0 = (Block *)((char *)p1 + p1->allow_mem_);
    while ((char *)p0 < mm->start + mp->mem_pool_size && p0->is_free_) {
      MP_DLINKLIST_DEL(mm->free_block_list_, p0);
      p1->allow_mem_ += p0->allow_mem_;
      p0 = (Block *)((char *)p0 + p0->allow_mem_);
    }

    *(Block **)((char *)p1 + p1->allow_mem_ - BLOCK_POINTER_SIZE) = p1;
  }

  static MemorySize GetUsedMemory(MemoryPool *mp) {
    MemorySize total_alloc = 0;
    Chunk *mm = mp->chunk_list_;
    while (mm) {
      total_alloc += mm->allow_mem_;
      mm = mm->next_chunk_;
    }
    return total_alloc;
  }

  static MemorySize GetProgramMemory(MemoryPool *mp) {
    MemorySize total_alloc_program = 0;
    Chunk *mm = mp->chunk_list_;
    while (mm) {
      total_alloc_program += mm->alloc_program_mem_;
      mm = mm->next_chunk_;
    }
    return total_alloc_program;
  }
};

static struct MemoryPool *memory_pool;
static bool is_memory_pool_inited = false;

void *AllocatorInterface::Allocate(MemorySize n_bytes) {
  if (!is_memory_pool_inited) {
    LOG_MP_INFO("Process Init memory pool");
    memory_pool = FirstFitAllocator::MemoryPoolInit(2048 * MB * 4, 1024 * MB, 0);
    is_memory_pool_inited = true;
  }
  void *p = FirstFitAllocator::MemoryPoolAlloc(memory_pool, n_bytes);
  if (!p) {
    LOG_MP_FATAL("Memory return nullptr.")
  }
  return p;
}

void AllocatorInterface::Deallocate(void *ptr) {
  FirstFitAllocator::MemoryPoolFree(memory_pool, ptr);
}

}  // namespace Allocator
}  // namespace Autoalg