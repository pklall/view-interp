MODULES   := cimg pmstereo cvstereo main
TARGET    := main

CXXFLAGS   = -std=c++11 -g -Wall -pthread -fopenmp -Ofast -finline-functions -ffast-math -Wfatal-errors -Iextern/libDAI/include/
# CXXFLAGS   = -std=c++11 -g -Wall -pthread -fopenmp -Wfatal-errors -Iextern/libDAI/include/ -fno-omit-frame-pointer #-fsanitize=address 
LD_FLAGS   = `pkg-config --cflags --libs x11 opencv eigen3` -lgomp -ljpeg -lpng -lpthread extern/libDAI/lib/libdai.a -llbfgs -ladolc

CXX       := g++
LD        := g++

SRC_DIR   := $(addprefix src/,$(MODULES))
BUILD_DIR := $(addprefix build/,$(MODULES))
SRC       := $(foreach sdir,$(SRC_DIR),$(wildcard $(sdir)/*.cpp))
OBJ       := $(patsubst src/%.cpp,build/%.o,$(SRC))

vpath %.cpp $(SRC_DIR)

define make-goal
$1/%.o: %.cpp
	$(CXX) -Isrc -I$$(<D) $(CXXFLAGS) -c $$< -o $$@
endef

.PHONY: all checkdirs clean

all: checkdirs $(TARGET)

$(TARGET): $(OBJ)
	$(LD) $^ $(CXXFLAGS) $(LD_FLAGS) -o $@ 

checkdirs: $(BUILD_DIR)

$(BUILD_DIR):
	@mkdir -p $@

clean:
	rm -rf $(BUILD_DIR)

$(foreach bdir,$(BUILD_DIR),$(eval $(call make-goal,$(bdir))))

