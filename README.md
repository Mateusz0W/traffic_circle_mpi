# Traffic Circle MPI

Parallel Monte Carlo simulation of a roundabout using MPI (Section 10.5.6).

## Kompilacja

```bash
make
# lub ręcznie:
mpicc -O2 -o traffic_circle_mpi traffic_circle_mpi.c -lm
```

## Uruchomienie

### Lokalnie
```bash
mpirun -np <P> ./traffic_circle_mpi <iteracje> <R>
```

### Na klastrze (hostfile `nodes`)
```bash
mpiexec -f nodes -n <P> ./traffic_circle_mpi <iteracje> <R>
```

### Przez Makefile (domyślnie: 4 procesy, 500 000 iteracji, 4 drogi)
```bash
make run                        # lokalnie
make run-cluster                # klaster
make run-cluster NP=16 ITER=1000000 ROADS=4
```

## Parametry

| Parametr    | Domyślnie | Opis                              |
|-------------|-----------|-----------------------------------|
| `P`         | —         | liczba procesów MPI               |
| `iteracje`  | 500 000   | łączna liczba kroków symulacji    |
| `R`         | 4         | liczba wjazdów/wyjazdów (max 16)  |

## Przykłady

```bash
# 16 węzłów, 1M iteracji, 4 drogi (parametry z podręcznika)
mpiexec -f nodes -n 16 ./traffic_circle_mpi 1000000 4

# 16 węzłów, 1M iteracji, 16 dróg
mpiexec -f nodes -n 16 ./traffic_circle_mpi 1000000 16

# 1 proces, szybki test
mpirun -np 1 ./traffic_circle_mpi 100000 4
```

## Animacja (tylko R=4)

Generuje `traffic_anim_data.csv` ze stanem ronda klatka po klatce,
a następnie `traffic_animation.gif`.

```bash
# 1. uruchom symulację z flagą --anim (na klastrze)
mpiexec -f nodes -n 16 ./traffic_circle_mpi 500000 4 --anim
# lub przez make:
make anim NP=16

# 2. wygeneruj GIF (wymaga Python + matplotlib)
python3 animate_traffic.py
# lub przez make:
make animate

# opcje skryptu
python3 animate_traffic.py --input traffic_anim_data.csv --output rondo.gif --fps 20
```
