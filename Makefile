CC     = mpicc
CFLAGS = -O2
TARGET = traffic_circle_mpi
SRC    = traffic_circle_mpi.c

# overridable defaults
NP    ?= 4
ITER  ?= 500000
ROADS ?= 4
HOSTS ?= nodes

.PHONY: all clean run run-cluster anim animate

all: $(TARGET)

$(TARGET): $(SRC)
	$(CC) $(CFLAGS) -o $(TARGET) $(SRC) -lm

# lokalnie
run: $(TARGET)
	mpirun -np $(NP) ./$(TARGET) $(ITER) $(ROADS)

# na klastrze z hostfile
run-cluster: $(TARGET)
	mpiexec -f $(HOSTS) -n $(NP) ./$(TARGET) $(ITER) $(ROADS)

# generuj CSV ze stanem ronda (tylko ROADS=4)
anim: $(TARGET)
	mpiexec -f $(HOSTS) -n $(NP) ./$(TARGET) $(ITER) 4 --anim

# generuj GIF z CSV
animate:
	python3 animate_traffic.py

clean:
	rm -f $(TARGET) traffic_anim_data.csv traffic_animation.gif
