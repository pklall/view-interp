MODULES   := cimg cvutil main
TARGET    := main

OPT_LEVEL := 

INCLUDES  := -Iextern/libDAI/include \
             -I/usr/include/eigen3 \
             -Iextern/ceres-solver/include \
			 -Iextern/SLIC-Superpixels \
			 -Iextern/Halide/include

CXXFLAGS  := $(OPT_LEVEL) \
             -std=c++11 \
             -g \
             -Wall \
             -pthread \
             -finline-functions \
             -ffast-math \
             -Wfatal-errors \
# -fopenmp \

# CXXFLAGS  += -DPRECOMPILE_CIMG

LD_DIRS   := -Lextern/libDAI/lib \
             -Lextern/ceres-solver/build/lib \
             -Lextern/SLIC-Superpixels \
             -Lextern/Halide/bin \

LD_STATIC := -lceres \
             -lslic \
			 -lHalide

LD_FLAGS  := $(LD_DIRS) \
             -ldl \
             -lpthread \
             -lgflags \
             -lglog \
             -lm \
             -lgomp \
             -lcamd \
             -lamd \
             -lcolamd \
             -lcholmod \
             -lccolamd \
             -lblas \
             -llapack \
			 -lpng \
			 -ljpeg \
			 -lX11 \
			 -lopencv_calib3d \
			 -lopencv_core \
			 -lopencv_features2d \
			 -lopencv_imgproc \
			 -lopencv_legacy \
			 -lopencv_video

CXX       := clang++
LD        := clang++

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

all: clangcomplete checkdirs $(TARGET)

$(TARGET): $(OBJ)
	$(LD) $^ $(LD_STATIC) $(LD_FLAGS) -o $@ 

checkdirs: $(BUILD_DIR)

clangcomplete:
	@echo "-Isrc $(addprefix -Isrc/,$(MODULES)) $(INCLUDES)" > .clang_complete

$(BUILD_DIR):
	@mkdir -p $@

clean:
	rm -rf $(BUILD_DIR)

$(foreach bdir,$(BUILD_DIR),$(eval $(call make-goal,$(bdir))))

