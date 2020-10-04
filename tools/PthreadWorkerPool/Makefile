CC      := gcc
CCFLAGS := 
LDFLAGS := -pthread -lm

TARGETS:= example
MAINS  := $(addsuffix .o, $(TARGETS) )
OBJ    := $(MAINS)
DEPS   := pthreadWorkerPool.h

.PHONY: all clean

all: $(TARGETS)

clean:
	rm -f $(TARGETS) $(OBJ)

$(OBJ): %.o : %.c $(DEPS)
	$(CC) -c -o $@ $< $(CCFLAGS)

$(TARGETS): % : $(filter-out $(MAINS), $(OBJ)) %.o
	$(CC) -o $@ $(LIBS) $^ $(CCFLAGS) $(LDFLAGS)
