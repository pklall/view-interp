MODULES   := cimg cvutil qpbo poly2tri main
TARGET    := main

# define DEBUG
# endef

ifdef DEBUG

OPT_LEVEL := -O1

CXX_ASAN  := -g\
             -fsanitize=address \
             -fno-omit-frame-pointer \
             -fno-optimize-sibling-calls \

LD_ASAN   := -fsanitize=address \
             -lasan \

else

OPT_LEVEL := -Ofast

CXX_ASAN  :=

LD_ASAN   :=

endif

INCLUDES  := -I/usr/include/eigen3 \
			 -Iextern/ \
             -Iextern/ceres-solver/include \
             -Iextern/SLIC-Superpixels \
             -Iextern/Halide/include \
             -Iextern/opengm/include \
             -Iextern/daisy-1.8.1/include \
			 -Iextern/nanoflann-1.1.7/include \
			 -Isrc/qpbo \

CXXFLAGS  := $(OPT_LEVEL) \
             $(CXX_ASAN) \
             -std=c++11 \
             -g \
             -Wall \
             -pthread \
             -finline-functions \
             -ffast-math \
             -Wfatal-errors \


LD_DIRS   := -Lextern/ceres-solver/build/lib \
             -Lextern/SLIC-Superpixels \
             -Lextern/Halide/bin \
             -Lextern/daisy-1.8.1/lib \
             -Lextern/klt \

LD_STATIC := -lceres \
             -lslic \
             -lHalide \
			 -ldaisy \
			 -lklt \

LD_FLAGS  := $(LD_DIRS) \
             $(LD_ASAN) \
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
             -lopencv_video \
			 -lflann \

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

