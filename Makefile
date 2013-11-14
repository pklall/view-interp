CPP_FILES = CImg.cpp main.cpp

HEADER_FILES = CImg.h common.h

TARGET = main

CXX = g++

CXXFLAGS = -std=c++11 -g -Wall -pthread -fopenmp -Ofast -finline-functions -ffast-math
# Use address sanitizer to detect bad memory access
# CXXFLAGS = -std=c++11 -g -Wall -pthread -fsanitize-address -O1 -fno-omit-frame-pointer
INC_FLAGS =
LD_FLAGS = `pkg-config --cflags --libs x11 eigen3 opencv` -lgomp -ljpeg -lpng

OBJDIR = obj/
SRCDIR = src/

all: $(TARGET)

OBJS = $(addprefix $(OBJDIR), $(notdir $(CPP_FILES:.cpp=.o)))

HEADERS = $(addprefix $(SRCDIR), $(HEADER_FILES))

$(OBJDIR)%.o: $(SRCDIR)%.cpp $(HEADERS)
	$(CXX) $(CXXFLAGS) -c -o $@ $<

$(TARGET):	$(OBJS)
	$(CXX) -o $(TARGET) $(CXXFLAGS) $(OBJS) $(LD_FLAGS) 

all: $(TARGET)

clean:
	rm -f $(OBJS) $(TARGET)
