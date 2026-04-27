/*
 * traffic_circle_mpi.c
 *
 * Parallel Monte Carlo simulation of a traffic circle (roundabout)
 * as described in Section 10.5.6.
 *
 * Each MPI process runs an independent simulation for a given number
 * of iterations; results are then reduced (averaged) on rank 0.
 *
 * Compile:
 *   mpicc -O2 -o traffic_circle_mpi traffic_circle_mpi.c -lm
 *
 * Run (example – 4 processes, 1 000 000 iterations, 4 roads):
 *   mpirun -np 4 ./traffic_circle_mpi 1000000 4
 *
 * Usage:
 *   ./traffic_circle_mpi <iterations> <num_roads>
 *
 *   num_roads : number of roads; each road has exactly ONE entrance AND
 *               one exit on the circle  (default 4, max 16)
 *
 * When num_roads == 4 the simulation uses exactly the parameters from
 * the textbook (Figure 10.21).  For other values the program generates
 * symmetric default parameters automatically.
 *
 * The circle is divided into CIRCLE_SIZE = 4 * num_roads segments so
 * that every entrance/exit pair gets its own slot.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <mpi.h>

/* ------------------------------------------------------------------ */
/* Constants                                                            */
/* ------------------------------------------------------------------ */
#define MAX_ENT   16   /* hard upper bound on entrances / exits */

/* ------------------------------------------------------------------ */
/* Simple LCG random-number generator (per-process seed)               */
/* ------------------------------------------------------------------ */
static unsigned long long rng_state;

static void rng_seed(unsigned long long s) { rng_state = s; }

/* Returns uniform in (0,1) */
static double rng_uniform(void)
{
    rng_state = rng_state * 6364136223846793005ULL + 1442695040888963407ULL;
    unsigned int hi = (unsigned int)(rng_state >> 33);
    return (hi + 0.5) / 4294967296.0;
}

/* Returns exponential random variable with mean `mean` */
static double rng_exp(double mean)
{
    double u;
    do { u = rng_uniform(); } while (u == 0.0);
    return -mean * log(u);
}

/* ------------------------------------------------------------------ */
/* Choose exit for a car entering at entrance `ent`                     */
/* Returns the circle index of the chosen exit                          */
/* ------------------------------------------------------------------ */
static int choose_exit(int ent,
                       int   num_ent,
                       int   num_exits,
                       double d[MAX_ENT][MAX_ENT],
                       int   exit_offset[])
{
    double u = rng_uniform();
    double cum = 0.0;
    for (int j = 0; j < num_exits; j++) {
        cum += d[ent][j];
        if (u < cum) return exit_offset[j];
    }
    return exit_offset[num_exits - 1]; /* rounding safety */
}

/* ------------------------------------------------------------------ */
/* Single-process simulation                                            */
/* ------------------------------------------------------------------ */
static void simulate(long iterations,
                     int  num_ent,
                     int  num_exits,
                     int  circle_size,
                     double f[],            /* mean inter-arrival time */
                     double d[][MAX_ENT],   /* exit probability matrix */
                     int  ent_offset[],     /* circle index of entrance i */
                     int  exit_offset[],    /* circle index of exit j     */
                     /* outputs */
                     double wait_prob[],    /* P(wait) per entrance */
                     double avg_queue[])    /* avg queue length per entrance */
{
    /* Allocate circle buffers */
    int *circle     = calloc(circle_size, sizeof(int));
    int *new_circle = calloc(circle_size, sizeof(int));

    /* Per-entrance statistics */
    long *arrival_cnt = calloc(num_ent, sizeof(long));
    long *wait_cnt    = calloc(num_ent, sizeof(long));
    long *queue       = calloc(num_ent, sizeof(long));
    double *queue_accum = calloc(num_ent, sizeof(double));

    /* Time until next arrival (continuous-time interleaved with discrete steps) */
    double *next_arrival = malloc(num_ent * sizeof(double));

    /* Initialise circle to empty (-1) */
    for (int i = 0; i < circle_size; i++) circle[i] = -1;

    /* Initialise next-arrival times */
    for (int i = 0; i < num_ent; i++)
        next_arrival[i] = rng_exp(f[i]);

    int *arrival = calloc(num_ent, sizeof(int));

    /* ----- warm-up: skip first 10% of iterations ----- */
    long warmup = iterations / 10;
    long total  = iterations + warmup;

    for (long iter = 0; iter < total; iter++) {

        /* Phase 1: new cars arrive */
        for (int i = 0; i < num_ent; i++) {
            arrival[i] = 0;
            next_arrival[i] -= 1.0;
            if (next_arrival[i] <= 0.0) {
                arrival[i] = 1;
                if (iter >= warmup) arrival_cnt[i]++;
                next_arrival[i] += rng_exp(f[i]);
            }
        }

        /* Phase 2: cars inside circle advance simultaneously */
        for (int i = 0; i < circle_size; i++) new_circle[i] = -1;

        for (int i = 0; i < circle_size; i++) {
            if (circle[i] == -1) continue;
            int j = (i + 1) % circle_size;
            if (circle[i] == j) {
                /* Car has reached its exit – leaves the circle */
                new_circle[j] = -1;
            } else {
                new_circle[j] = circle[i];
            }
        }
        memcpy(circle, new_circle, circle_size * sizeof(int));

        /* Phase 3: cars enter circle */
        for (int i = 0; i < num_ent; i++) {
            int slot = ent_offset[i];
            if (circle[slot] == -1) {
                /* Slot is free */
                if (queue[i] > 0) {
                    queue[i]--;
                    circle[slot] = choose_exit(i, num_ent, num_exits, d, exit_offset);
                } else if (arrival[i]) {
                    arrival[i] = 0;
                    circle[slot] = choose_exit(i, num_ent, num_exits, d, exit_offset);
                }
            } else {
                /* Slot is occupied – new arrivals must queue */
                if (arrival[i]) {
                    if (iter >= warmup) wait_cnt[i]++;
                    queue[i]++;
                }
            }
        }

        /* Accumulate queue lengths (after warm-up) */
        if (iter >= warmup) {
            for (int i = 0; i < num_ent; i++)
                queue_accum[i] += (double)queue[i];
        }
    }

    /* Compute statistics */
    for (int i = 0; i < num_ent; i++) {
        wait_prob[i]  = (arrival_cnt[i] > 0)
                        ? (double)wait_cnt[i] / (double)arrival_cnt[i]
                        : 0.0;
        avg_queue[i]  = queue_accum[i] / (double)iterations;
    }

    free(circle); free(new_circle);
    free(arrival_cnt); free(wait_cnt);
    free(queue); free(queue_accum);
    free(next_arrival); free(arrival);
}

/* ================================================================== */
int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);

    int rank, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    /* ---- Parse arguments ---- */
    long iterations  = 500000;
    int  num_roads   = 4;   /* entrances == exits == num_roads */

    if (argc >= 2) iterations = atol(argv[1]);
    if (argc >= 3) num_roads  = atoi(argv[2]);

    int num_ent   = num_roads;
    int num_exits = num_roads;   /* always equal */

    if (num_roads  < 1 || num_roads  > MAX_ENT) { if (!rank) fprintf(stderr, "num_roads must be 1-%d\n", MAX_ENT); MPI_Finalize(); return 1; }
    if (iterations < 1)                          { if (!rank) fprintf(stderr, "iterations must be > 0\n");          MPI_Finalize(); return 1; }

    /* ---- Build circle geometry ---- */
    /* Each road shares the same slot for its entrance and exit,
       exactly as in the textbook (N=0, W=4, S=8, E=12 for 4 roads). */
    int circle_size = 4 * num_roads;

    int ent_offset[MAX_ENT];
    int exit_offset[MAX_ENT];

    /* Entrance i and exit i occupy the same circle slot (road i). */
    for (int i = 0; i < num_roads; i++)
        ent_offset[i] = exit_offset[i] = i * 4;

    /* ---- Traffic parameters ---- */
    double f[MAX_ENT];     /* mean inter-arrival time */
    double d[MAX_ENT][MAX_ENT]; /* exit probability matrix */

    if (num_roads == 4) {
        /* ---- Textbook parameters (Figure 10.21) ---- */
        /* N=0, W=1, S=2, E=3 */
        double f4[4] = {3.0, 3.0, 4.0, 2.0};
        double d4[4][4] = {
            {0.1, 0.2, 0.5, 0.2},   /* N */
            {0.2, 0.1, 0.3, 0.4},   /* W */
            {0.5, 0.1, 0.1, 0.3},   /* S */
            {0.3, 0.4, 0.2, 0.1}    /* E */
        };
        for (int i = 0; i < 4; i++) {
            f[i] = f4[i];
            for (int j = 0; j < 4; j++) d[i][j] = d4[i][j];
        }
    } else {
        /* ---- Generic symmetric parameters ---- */
        /* mean inter-arrival: alternate between 3 and 4 */
        for (int i = 0; i < num_roads; i++)
            f[i] = (i % 2 == 0) ? 3.0 : 4.0;

        /* Uniform exit probabilities */
        for (int i = 0; i < num_roads; i++)
            for (int j = 0; j < num_roads; j++)
                d[i][j] = 1.0 / num_roads;
    }

    /* ---- Per-process share of iterations ---- */
    long local_iter = iterations / nprocs;
    if (rank == nprocs - 1)
        local_iter += iterations % nprocs;   /* remainder to last rank */

    /* Seed each process differently */
    rng_seed((unsigned long long)(rank + 1) * 1234567891ULL
             ^ (unsigned long long)time(NULL));

    /* ---- Run local simulation ---- */
    double local_wait_prob[MAX_ENT] = {0};
    double local_avg_queue[MAX_ENT] = {0};

    simulate(local_iter, num_ent, num_exits, circle_size,
             f, d, ent_offset, exit_offset,
             local_wait_prob, local_avg_queue);

    /* ---- Reduce results to rank 0 ---- */
    double global_wait_prob[MAX_ENT] = {0};
    double global_avg_queue[MAX_ENT] = {0};

    MPI_Reduce(local_wait_prob, global_wait_prob, num_roads,
               MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(local_avg_queue, global_avg_queue, num_roads,
               MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    /* ---- Print results ---- */
    if (rank == 0) {
        for (int i = 0; i < num_roads; i++) {
            global_wait_prob[i] /= nprocs;
            global_avg_queue[i] /= nprocs;
        }

        const char *label4[4] = {"N", "W", "S", "E"};

        printf("\n");
        printf("=============================================================\n");
        printf("  Traffic Circle MPI Simulation\n");
        printf("  Processes    : %d\n", nprocs);
        printf("  Iterations   : %ld (per process: ~%ld)\n",
               iterations, iterations / nprocs);
        printf("  Roads (in=out): %d\n", num_roads);
        printf("  Circle size  : %d segments\n", circle_size);
        printf("=============================================================\n");
        printf("\n");
        printf("  %-10s  %-12s  %-12s  %-12s\n",
               "Entrance", "MeanArrival", "P(wait)", "AvgQueue");
        printf("  %-10s  %-12s  %-12s  %-12s\n",
               "--------", "-----------", "-------", "--------");

        for (int i = 0; i < num_roads; i++) {
            const char *lbl;
            char buf[8];
            if (num_roads == 4) {
                lbl = label4[i];
            } else {
                snprintf(buf, sizeof(buf), "%d", i);
                lbl = buf;
            }
            printf("  %-10s  %-12.2f  %-12.4f  %-12.4f\n",
                   lbl, f[i], global_wait_prob[i], global_avg_queue[i]);
        }
        printf("\n");
        printf("Answer to question (a) – steady-state P(wait) shown in column 3.\n");
        printf("Answer to question (b) – steady-state avg queue shown in column 4.\n");
        printf("\n");
    }

    MPI_Finalize();
    return 0;
}

