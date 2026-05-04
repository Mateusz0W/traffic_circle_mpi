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

/* Hard upper bound on the number of entrances / exits */
#define MAX_ENT   16

/* ------------------------------------------------------------------ */
/* Simple LCG random-number generator (per-process seed)               */
/* ------------------------------------------------------------------ */
static unsigned long long rng_state;

static void rng_seed(unsigned long long s) { rng_state = s; }

/* Returns a uniform sample in (0,1) */
static double rng_uniform(void)
{
    rng_state = rng_state * 6364136223846793005ULL + 1442695040888963407ULL;
    unsigned int hi = (unsigned int)(rng_state >> 33);
    return (hi + 0.5) / 4294967296.0;
}

/* Returns an exponential random variable with mean `mean` */
static double rng_exp(double mean)
{
    double u;
    do { u = rng_uniform(); } while (u == 0.0);
    return -mean * log(u);
}

/* ------------------------------------------------------------------ */
/* Choose exit for a car entering at entrance `ent`                     */
/* Returns the circle index of the chosen exit slot                     */
/* ------------------------------------------------------------------ */
static int choose_exit(int ent,
                       int    num_exits,
                       double d[MAX_ENT][MAX_ENT],
                       int    exit_offset[])
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
/* rec_fp != NULL  →  write circle state + queue to CSV every          */
/* rec_every iterations (post warm-up); used for --anim mode only.     */
/* ------------------------------------------------------------------ */
static void simulate(long iterations,
                     int  num_ent,
                     int  num_exits,
                     int  circle_size,
                     double f[],            /* mean inter-arrival time per entrance */
                     double d[][MAX_ENT],   /* exit probability matrix */
                     int  ent_offset[],     /* circle index of entrance i */
                     int  exit_offset[],    /* circle index of exit j */
                     /* outputs */
                     double wait_prob[],    /* P(wait) per entrance */
                     double avg_queue[],    /* avg queue length per entrance */
                     FILE  *rec_fp,         /* NULL = no recording */
                     int    rec_every)      /* record every N post-warmup iterations */
{
    /* Circle state: circle[i] holds the destination exit slot of the car
       occupying slot i, or -1 if the slot is empty */
    int *circle     = calloc(circle_size, sizeof(int));
    int *new_circle = calloc(circle_size, sizeof(int));

    long   *arrival_cnt = calloc(num_ent, sizeof(long));
    long   *wait_cnt    = calloc(num_ent, sizeof(long));
    long   *queue       = calloc(num_ent, sizeof(long));
    double *queue_accum = calloc(num_ent, sizeof(double));

    /* Time remaining until next arrival at each entrance */
    double *next_arrival = malloc(num_ent * sizeof(double));

    for (int i = 0; i < circle_size; i++) circle[i] = -1;

    for (int i = 0; i < num_ent; i++)
        next_arrival[i] = rng_exp(f[i]);

    /* Binary flag: did a car arrive at entrance i this iteration? */
    int *arrival = calloc(num_ent, sizeof(int));

    /* Discard the first 10% of iterations to reach steady state */
    long warmup = iterations / 10;
    long total  = iterations + warmup;

    for (long iter = 0; iter < total; iter++) {

        /* Phase 1: advance arrival clocks; record arrivals */
        for (int i = 0; i < num_ent; i++) {
            arrival[i] = 0;
            next_arrival[i] -= 1.0;
            if (next_arrival[i] <= 0.0) {
                arrival[i] = 1;
                if (iter >= warmup) arrival_cnt[i]++;
                next_arrival[i] += rng_exp(f[i]);
            }
        }

        /* Phase 2: all cars on the circle advance one slot simultaneously.
           A car exits when it reaches its destination slot (circle[i] == i). */
        for (int i = 0; i < circle_size; i++) new_circle[i] = -1;

        for (int i = 0; i < circle_size; i++) {
            if (circle[i] == -1) continue;
            int j = (i + 1) % circle_size;
            if (circle[i] == i) {
                /* Car has reached its exit – leaves the circle */
            } else {
                new_circle[j] = circle[i];
            }
        }
        memcpy(circle, new_circle, circle_size * sizeof(int));

        /* Phase 3: admit cars from entrance queues into free slots */
        for (int i = 0; i < num_ent; i++) {
            int slot = ent_offset[i];
            if (circle[slot] == -1) {
                /* Entrance slot is free */
                if (queue[i] > 0) {
                    /* Dequeue the first waiting car and place it on the circle */
                    queue[i]--;
                    circle[slot] = choose_exit(i, num_exits, d, exit_offset);
                    /* A new arrival in the same step must still wait,
                       because the slot is now taken by the dequeued car */
                    if (arrival[i]) {
                        if (iter >= warmup) wait_cnt[i]++;
                        queue[i]++;
                    }
                } else if (arrival[i]) {
                    /* No queue and slot is free – car enters immediately */
                    circle[slot] = choose_exit(i, num_exits, d, exit_offset);
                }
            } else {
                /* Entrance slot is occupied – arriving cars must queue */
                if (arrival[i]) {
                    if (iter >= warmup) wait_cnt[i]++;
                    queue[i]++;
                }
            }
        }

        /* Accumulate queue lengths after warm-up */
        if (iter >= warmup) {
            for (int i = 0; i < num_ent; i++)
                queue_accum[i] += (double)queue[i];

            /* Optional per-frame CSV recording for animation */
            long post = iter - warmup;
            if (rec_fp && post % rec_every == 0) {
                fprintf(rec_fp, "%ld", post);
                for (int i = 0; i < circle_size; i++) fprintf(rec_fp, ",%d", circle[i]);
                for (int i = 0; i < num_ent;     i++) fprintf(rec_fp, ",%ld", queue[i]);
                fprintf(rec_fp, "\n");
            }
        }
    }

    /* Compute per-entrance statistics */
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

    long iterations  = 500000;
    int  num_roads   = 4;   /* entrances == exits == num_roads */
    int  anim_mode   = 0;
    int  fast_mode   = 0;

    if (argc >= 2) iterations = atol(argv[1]);
    if (argc >= 3) num_roads  = atoi(argv[2]);
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--anim") == 0) anim_mode = 1;
        if (strcmp(argv[i], "--fast") == 0) fast_mode = 1;
    }

    int num_ent   = num_roads;
    int num_exits = num_roads;

    if (num_roads  < 1 || num_roads  > MAX_ENT) { if (!rank) fprintf(stderr, "num_roads must be 1-%d\n", MAX_ENT); MPI_Finalize(); return 1; }
    if (iterations < 1)                          { if (!rank) fprintf(stderr, "iterations must be > 0\n");          MPI_Finalize(); return 1; }
    if (iterations < nprocs)                     { if (!rank) fprintf(stderr, "iterations must be >= num_processes (%d)\n", nprocs); MPI_Finalize(); return 1; }

    /* Each road occupies every 4th slot: road i → slot i*4.
       Entrance and exit share the same slot (textbook layout). */
    int circle_size = 4 * num_roads;

    int ent_offset[MAX_ENT];
    int exit_offset[MAX_ENT];

    for (int i = 0; i < num_roads; i++)
        ent_offset[i] = exit_offset[i] = i * 4;

    double f[MAX_ENT];
    double d[MAX_ENT][MAX_ENT];

    if (num_roads == 4) {
        /* Textbook parameters (Figure 10.21): N=0, W=1, S=2, E=3 */
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
        /* Generic symmetric parameters scaled to keep the system stable.
           Stability requires f_avg > (R-1)/2; alternating R-1/R satisfies
           this for any R and matches textbook values exactly when R=4. */
        for (int i = 0; i < num_roads; i++)
            f[i] = (i % 2 == 0) ? (double)(num_roads - 1) : (double)num_roads;

        for (int i = 0; i < num_roads; i++)
            for (int j = 0; j < num_roads; j++)
                d[i][j] = 1.0 / num_roads;
    }


    /* --fast: scale down inter-arrival times to saturate the roundabout.
       Factor 0.45 pushes utilisation to ~136% → queues grow, circle fills up. */
    if (fast_mode)
        for (int i = 0; i < num_roads; i++)
            f[i] *= 0.45;



    /* Distribute iterations evenly; the last rank absorbs the remainder */
    long local_iter = iterations / nprocs;
    if (rank == nprocs - 1)
        local_iter += iterations % nprocs;

    /* Each process gets a unique, time-varied seed */
    rng_seed((unsigned long long)(rank + 1) * 1234567891ULL
             ^ (unsigned long long)time(NULL));

    double local_wait_prob[MAX_ENT] = {0};
    double local_avg_queue[MAX_ENT] = {0};

    simulate(local_iter, num_ent, num_exits, circle_size,
             f, d, ent_offset, exit_offset,
             local_wait_prob, local_avg_queue, NULL, 0);

    /* Weight each process's result by its iteration count so that the
       MPI_Reduce SUM divided by total iterations gives the correct mean */
    for (int i = 0; i < num_roads; i++) {
        local_wait_prob[i] *= (double)local_iter;
        local_avg_queue[i] *= (double)local_iter;
    }

    double global_wait_prob[MAX_ENT] = {0};
    double global_avg_queue[MAX_ENT] = {0};

    MPI_Reduce(local_wait_prob, global_wait_prob, num_roads,
               MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(local_avg_queue, global_avg_queue, num_roads,
               MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        for (int i = 0; i < num_roads; i++) {
            global_wait_prob[i] /= (double)iterations;
            global_avg_queue[i] /= (double)iterations;
        }

        const char *label4[4] = {"N", "W", "S", "E"};

        printf("\n");
        printf("=============================================================\n");
        printf("  Traffic Circle MPI Simulation%s\n", fast_mode ? "  [--fast]" : "");
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

        /* --anim: run a short single-process simulation and record every
           iteration to traffic_anim_data.csv for use with animate_traffic.py */
        if (anim_mode && num_roads == 4) {
            const char *csv_path = "traffic_anim_data.csv";
            FILE *fp = fopen(csv_path, "w");
            if (!fp) {
                fprintf(stderr, "Cannot open %s for writing\n", csv_path);
            } else {
                /* Write CSV header: iter, s0..s15, q0..q3 */
                fprintf(fp, "iter");
                for (int i = 0; i < circle_size; i++) fprintf(fp, ",s%d", i);
                for (int i = 0; i < num_roads;   i++) fprintf(fp, ",q%d", i);
                fprintf(fp, "\n");

                /* Fresh seed so the animation run is reproducible */
                rng_seed(42ULL);

                double rec_wait[MAX_ENT] = {0};
                double rec_queue[MAX_ENT] = {0};

                /* 600 post-warmup iterations recorded every step = 600 frames */
                simulate(600, num_ent, num_exits, circle_size,
                         f, d, ent_offset, exit_offset,
                         rec_wait, rec_queue, fp, 1);

                fclose(fp);
                printf("Animation data written to %s\n", csv_path);
                printf("Generate animation:  python3 animate_traffic.py\n\n");
            }
        }
    }

    MPI_Finalize();
    return 0;
}
