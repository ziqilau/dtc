CC = g++ -Wall
#CFLAGS = -g -Wall -O3 -ffast-math -DHAVE_INLINE -DGSL_RANGE_CHECK_OFF
# CFLAGS = -g -Wall
LDFLAGS = -lgsl -lm -lgslcblas

GSL_INCLUDE_MAC = /usr/local/include
GSL_LIB_MAC = /usr/local/lib

GSL_INCLUDE_LINUX = /usr/include
GSL_LIB_LINUX = /usr/lib

LSOURCE =  utils.cpp corpus.cpp state.cpp dtc.cpp main.cpp
LHEADER =  utils.h corpus.h dtc.h state.h

mac: $(LSOURCE) $(HEADER)
	#$(CC) $(LSOURCE) -o $@ $(LDFLAGS)
	$(CC)  -I$(GSL_INCLUDE_MAC) -L$(GSL_LIB_LINUX) $(LSOURCE) -o $ dtc $(LDFLAGS)
	
d-mac: $(LSOURCE) $(HEADER)
	#$(CC) $(LSOURCE) -o $@ $(LDFLAGS)
	$(CC) -g -I$(GSL_INCLUDE_MAC) -L$(GSL_LIB_LINUX) $(LSOURCE) -o $ dtc-d $(LDFLAGS)

linux: $(LSOURCE) $(HEADER)
	#$(CC) $(LSOURCE) -o $@ $(LDFLAGS)
	$(CC)  -I$(GSL_INCLUDE_LINUX) -L$(GSL_LIB_LINUX) $(LSOURCE) -o $ dtc $(LDFLAGS)
	
d-linux: $(LSOURCE) $(HEADER)
	#$(CC) $(LSOURCE) -o $@ $(LDFLAGS)
	$(CC) -g -I$(GSL_INCLUDE_LINUX) -L$(GSL_LIB_LINUX) $(LSOURCE) -o $ dtc-d $(LDFLAGS)

clean:
	rm -f *.o dtc
