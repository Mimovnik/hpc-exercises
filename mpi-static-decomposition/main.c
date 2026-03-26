#include <errno.h>
#include <mpi.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

typedef uint64_t u64;

typedef struct timeval timeval;

static inline u64 min(u64 a, u64 b) { return a < b ? a : b; }

u64 parse_args(int argc, char **argv) {
  if (argc != 2) {
    fprintf(stderr, "[Error] Wrong number of arguments\n");
    fprintf(stderr, "Usage: %s <upper-limit>\n", argv[0]);
    exit(EXIT_FAILURE);
  }

  char *end;
  errno = 0;
  u64 value = strtoull(argv[1], &end, 10);

  if (errno != 0 || end == argv[1] || *end != '\0') {
    fprintf(stderr, "[Error] Cannot parse the <upper-limit>\n");
    exit(EXIT_FAILURE);
  }

  return value;
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

u64 do_work(int rank, int proc_count, u64 upper_limit) {
  u64 workers_count = proc_count - 1;
  u64 idx = rank - 1;
  u64 start = idx * upper_limit / workers_count;
  u64 end = (idx + 1) * upper_limit / workers_count;
  // Overlap with next range to catch twins on boundaries
  end = end + PRIME_GAP;
  // Don't go further than upper limit
  end = min(end, upper_limit);

  return number_of_twin_primes_between(start, end);
}

int main(int argc, char **argv) {
  u64 upper_limit = parse_args(argc, argv);
  int partial_count = 0, final_count;
  timeval t_start, t_stop;
  int rank, proc_count;

  MPI_Init(&argc, &argv);

  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &proc_count);

  if (is_master(rank))
    gettimeofday(&t_start, NULL);

  guard_proc_count(proc_count, upper_limit, rank);

  if (!is_master(rank)) {
    gettimeofday(&t_start, NULL);
    partial_count = do_work(rank, proc_count, upper_limit);
    gettimeofday(&t_stop, NULL);
    print_exec_time(&t_start, &t_stop, rank);
  }

  MPI_Reduce(&partial_count, &final_count, 1, MPI_INT, MPI_SUM, 0,
             MPI_COMM_WORLD);

  if (is_master(rank)) {
    gettimeofday(&t_stop, NULL);
    print_exec_time(&t_start, &t_stop, rank);
    printf("There are %d twin primes in [0, %lu)\n", final_count, upper_limit);
  }

  MPI_Finalize();
}
