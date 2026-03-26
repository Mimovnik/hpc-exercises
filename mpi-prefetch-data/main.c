#include <errno.h>
#include <mpi.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

typedef uint64_t u64;

typedef struct timeval timeval;

static inline u64 min(u64 a, u64 b) { return a < b ? a : b; }

typedef struct {
  u64 upper_limit;
  int avg_tasks_per_proc;
} args;

u64 parse_u64(char *str) {
  char *end;
  errno = 0;
  u64 value = strtoull(str, &end, 10);

  if (errno != 0 || end == str || *end != '\0') {
    fprintf(stderr, "[Error] Cannot parse %s to u64\n", str);
    exit(EXIT_FAILURE);
  }

  return value;
}

int parse_int(char *str) {
  char *end;
  errno = 0;
  u64 value = strtod(str, &end);

  if (errno != 0 || end == str || *end != '\0') {
    fprintf(stderr, "[Error] Cannot parse %s to int\n", str);
    exit(EXIT_FAILURE);
  }

  return value;
}

args parse_args(int argc, char **argv) {
  if (argc != 3) {
    fprintf(stderr, "[Error] Wrong number of arguments\n");
    fprintf(stderr, "Usage: %s <upper-limit> <avg_tasks_per_proc>\n", argv[0]);
    exit(EXIT_FAILURE);
  }

  return (args){
      .upper_limit = parse_u64(argv[1]),
      .avg_tasks_per_proc = parse_int(argv[2]),
  };
}

void print_exec_time(timeval *start, timeval *stop, int rank) {
  long s = stop->tv_sec - start->tv_sec;
  long us = stop->tv_usec - start->tv_usec;
  if (us < 0) {
    s -= 1;
    us += 1000 * 1000;
  }
  long ms = us / 1000;
  long us_remainder = us % 1000;

  printf("[rank%d]: Execution time: %lds %ldms %ldus\n", rank, s, ms,
         us_remainder);

  return;
}

int is_prime(u64 number) {
  if (number <= 1)
    return 0;
  if (number == 2)
    return 1;
  if (number % 2 == 0)
    return 0;
  for (u64 i = 3; i * i <= number; i += 2) {
    if (number % i == 0)
      return 0;
  }

  return 1;
}

#define PRIME_GAP 2
// Calculate the number of twin primes in the range
// including start, excluding end
int number_of_twin_primes_between(u64 start, u64 end) {
  int count = 0;
  for (u64 i = start; i < end - PRIME_GAP; i++) {
    if (!is_prime(i))
      continue;

    if (is_prime(i + PRIME_GAP))
      count++;
  }
  return count;
}

static inline int is_master(int rank) { return rank == 0; }

void guard_proc_count(int proc_count, u64 upper_limit, int rank) {
  if (proc_count >= upper_limit) {
    if (is_master(rank)) {
      fprintf(stderr,
              "Error: Number of processes (%d) exceeds <upper-limit> (%lu).",
              proc_count, upper_limit);
      fprintf(
          stderr,
          "Increase <upper-limit> or decrease the number of processes (-np X)");
    }
    MPI_Finalize();
    exit(EXIT_FAILURE);
  }
}

typedef struct {
  u64 start;
  u64 end;
} chunk_t;

#define TAG_WORK 1
#define TAG_RESULT 2
#define TAG_TERMINATE 3

int main(int argc, char **argv) {
  int final_count = 0;
  timeval t_start, t_stop;
  int rank, proc_count;

  MPI_Status status;
  MPI_Init(&argc, &argv);

  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &proc_count);

  int worker_count = proc_count - 1;

  if (is_master(rank)) {
    args args = parse_args(argc, argv);
    u64 upper_limit = args.upper_limit;
    guard_proc_count(proc_count, upper_limit, rank);
    int avg_tasks_per_proc = args.avg_tasks_per_proc;

    gettimeofday(&t_start, NULL);

    MPI_Request send_work_reqs[worker_count];

    MPI_Request recv_results[worker_count];
    int results[worker_count];

    u64 chunk_size =
        upper_limit / (proc_count * avg_tasks_per_proc); // heuristic
    u64 next_start = 0;

    // Sent initial work
    for (int worker_i = 0; worker_i < worker_count; worker_i++) {
      if (next_start >= upper_limit)
        break;

      chunk_t chunk = {
          .start = next_start,
          .end = min(next_start + chunk_size + PRIME_GAP, upper_limit),
      };

      int worker_rank = worker_i + 1;

      MPI_Isend(&chunk, sizeof(chunk_t), MPI_BYTE, worker_rank, TAG_WORK,
                MPI_COMM_WORLD, &send_work_reqs[worker_i]);

      MPI_Irecv(&results[worker_i], 1, MPI_INT, worker_rank, TAG_RESULT,
                MPI_COMM_WORLD, &recv_results[worker_i]);

      next_start += chunk_size;
    }

    // Dynamic scheduling- receive and send until there is work
    int active_workers = proc_count - 1;

    while (active_workers > 0) {
      int partial_count;
      int worker_i;

      MPI_Waitany(worker_count, recv_results, &worker_i, &status);

      partial_count = results[worker_i];
      final_count += partial_count;

      int worker_rank = worker_i + 1;

      if (next_start < upper_limit) {
        chunk_t chunk = {
            .start = next_start,
            .end = min(next_start + chunk_size + PRIME_GAP, upper_limit),
        };

        MPI_Wait(&send_work_reqs[worker_i], MPI_STATUS_IGNORE);
        MPI_Isend(&chunk, sizeof(chunk_t), MPI_BYTE, worker_rank, TAG_WORK,
                  MPI_COMM_WORLD, &send_work_reqs[worker_i]);

        MPI_Irecv(&results[worker_i], 1, MPI_INT, worker_rank, TAG_RESULT,
                  MPI_COMM_WORLD, &recv_results[worker_i]);

        next_start += chunk_size;
      } else {
        MPI_Isend(NULL, 0, MPI_BYTE, worker_rank, TAG_TERMINATE, MPI_COMM_WORLD,
                  &send_work_reqs[worker_i]);
        active_workers--;
      }
    }

    for (int worker_i = 0; worker_i < worker_count; worker_i++) {
      MPI_Wait(&send_work_reqs[worker_i], MPI_STATUS_IGNORE);
    }

    gettimeofday(&t_stop, NULL);
    print_exec_time(&t_start, &t_stop, rank);
    printf("There are %d twin primes in [0, %lu)\n", final_count, upper_limit);
  }

  if (!is_master(rank)) {
    gettimeofday(&t_start, NULL);
    chunk_t chunks[worker_count];
    chunk_t next_chunks[worker_count];

    int worker_i = rank - 1;

    MPI_Recv(&chunks[worker_i], sizeof(chunk_t), MPI_BYTE, 0, MPI_ANY_TAG,
             MPI_COMM_WORLD, &status);

    MPI_Request recv_next, send_result;
    int send_in_flight = 0;

    while (status.MPI_TAG != TAG_TERMINATE) {
      MPI_Irecv(&next_chunks[worker_i], sizeof(chunk_t), MPI_BYTE, 0,
                MPI_ANY_TAG, MPI_COMM_WORLD, &recv_next);

      chunk_t chunk = chunks[worker_i];
      int result = number_of_twin_primes_between(chunk.start, chunk.end);

      if (send_in_flight)
        MPI_Wait(&send_result, MPI_STATUS_IGNORE);

      MPI_Isend(&result, 1, MPI_INT, 0, TAG_RESULT, MPI_COMM_WORLD,
                &send_result);
      send_in_flight = 1;

      MPI_Wait(&recv_next, &status);
      chunks[worker_i] = next_chunks[worker_i];
    }
    // Wait for last send
    if (send_in_flight)
      MPI_Wait(&send_result, MPI_STATUS_IGNORE);

    gettimeofday(&t_stop, NULL);
    print_exec_time(&t_start, &t_stop, rank);
  }

  MPI_Finalize();
}
