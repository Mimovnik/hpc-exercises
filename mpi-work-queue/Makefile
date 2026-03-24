CC=mpicc
CFLAGS=-O2
LIBS=-lm

ifeq (run,$(firstword $(MAKECMDGOALS)))
  # use the rest as arguments for "run"
  RUN_ARGS := $(wordlist 2,$(words $(MAKECMDGOALS)),$(MAKECMDGOALS))
  # first argument is the number of processes
  NP := $(word 1,$(RUN_ARGS))
  # remaining arguments are program arguments
  PROG_ARGS := $(wordlist 2,$(words $(RUN_ARGS)),$(RUN_ARGS))
  # ...and turn them into do-nothing targets
  $(eval $(RUN_ARGS):;@:)
endif


main: main.c
	${CC} ${CFLAGS} $< -o main ${LIBS}

run: main
	mpirun -np $(NP) ./main $(PROG_ARGS)

clean:
	rm main
