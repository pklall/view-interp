MODULES   := cimg pmstereo cvstereo main
TARGET    := main

OPT_LEVEL := -O1

INCLUDES  := -Iextern/libDAI/include \
             -I/usr/include/eigen3 \
             -Iextern/ceres-solver/include

CXXFLAGS  := $(OPT_LEVEL) \
             -std=c++11 \
             -g \
             -Wall \
             -pthread \
             -fopenmp \
             -finline-functions \
             -ffast-math \
             -Wfatal-errors \

# CXXFLAGS   = -std=c++11 -g -Wall -pthread -fopenmp -Wfatal-errors -Iextern/libDAI/include/ -fno-omit-frame-pointer #-fsanitize=address 

LD_DIRS   := -Lextern/libDAI/lib \
             -Lextern/ceres-solver/build/lib

LD_STATIC := -lceres

LD_FLAGS  := $(LD_DIRS) \
             -lpthread \
             -lgflags \
             -lglog \
             -lm \
             -lgomp \
             -fopenmp \
             -lcamd \
             -lamd \
             -lcolamd \
             -lcholmod \
             -lccolamd \
             -lblas \
             -llapack \
             `pkg-config --cflags --libs x11 opencv eigen3`

CXX       := g++
LD        := g++

SRC_DIR   := $(addprefix src/,$(MODULES))
BUILD_DIR := $(addprefix build/,$(MODULES))
SRC       := $(foreach sdir,$(SRC_DIR),$(wildcard $(sdir)/*.cpp))
OBJ       := $(patsubst src/%.cpp,build/%.o,$(SRC))

vpath %.cpp $(SRC_DIR)

define make-goal
$1/%.o: %.cpp
	$(CXX) -Isrc -I$$(<D) $(CXXFLAGS) $(INCLUDES) -c $$< -o $$@
endef

.PHONY: all checkdirs clean

all: checkdirs $(TARGET)

$(TARGET): $(OBJ)
	$(LD) $^ $(CXXFLAGS) $(LD_STATIC) $(LD_FLAGS) -o $@ 

checkdirs: $(BUILD_DIR)

$(BUILD_DIR):
	@mkdir -p $@

clean:
	rm -rf $(BUILD_DIR)

$(foreach bdir,$(BUILD_DIR),$(eval $(call make-goal,$(bdir))))

