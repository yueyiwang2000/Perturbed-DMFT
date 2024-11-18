# Makefile for compiling the pybind11 C++ module with Blitz++ and GSL support

# Compiler and flags
CXX = c++
CXXFLAGS = -O3 -Wall -shared -std=c++2a -fPIC -undefined dynamic_lookup
INCLUDES = $(shell python3 -m pybind11 --includes) -I./cmn

# Python configuration
PYTHON_VERSION = 3.11
PYTHON_INCLUDE = /Users/yueyiwang/anaconda3/include/python$(PYTHON_VERSION)
PYTHON_LIBDIR = /Users/yueyiwang/anaconda3/lib
EXT_SUFFIX = $(shell python3-config --extension-suffix)

# Blitz++ and GSL configuration
BLITZ_INCLUDE = /opt/homebrew/Cellar/blitz/1.0.2/include
BLITZ_LIBDIR = /opt/homebrew/Cellar/blitz/1.0.2/lib
GSL_INCLUDE = /opt/homebrew/Cellar/gsl/2.7.1/include
GSL_LIBDIR = /opt/homebrew/Cellar/gsl/2.7.1/lib

# Directories
SRCDIR = src
BUILDDIR = build
TARGETDIR = .
INCLUDEDIR = include

# Source and object files
SOURCES = $(wildcard $(SRCDIR)/*.cpp)

# Target shared library
TARGET = $(TARGETDIR)/svdwrap.so

# Rules
all: $(TARGET)

$(TARGET): $(SOURCES)
	@mkdir -p $(TARGETDIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -I$(BLITZ_INCLUDE) -I$(GSL_INCLUDE) $^ svdwrap.cpp -o $@ -L$(PYTHON_LIBDIR) -L$(BLITZ_LIBDIR) -lblitz -L$(GSL_LIBDIR) -lgsl -lgslcblas -lm -llapack -lblas

clean:
	@rm -rf $(TARGET)

# Phony targets
.PHONY: all clean
