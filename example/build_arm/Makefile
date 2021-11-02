# Copyright (C) 2013-2018 Altera Corporation, San Jose, California, USA. All rights reserved.
# Permission is hereby granted, free of charge, to any person obtaining a copy of this
# software and associated documentation files (the "Software"), to deal in the Software
# without restriction, including without limitation the rights to use, copy, modify, merge,
# publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to
# whom the Software is furnished to do so, subject to the following conditions:
# The above copyright notice and this permission notice shall be included in all copies or
# substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
# OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
# HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
# WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
# OTHER DEALINGS IN THE SOFTWARE.
#
# This agreement shall be governed in all respects by the laws of the State of California and
# by the laws of the United States of America.
# This is a GNU Makefile.

# You must configure INTELFPGAOCLSDKROOT to point the root directory of the Intel(R) FPGA SDK for OpenCL(TM)
# software installation.
# See http://www.altera.com/literature/hb/opencl-sdk/aocl_getting_started.pdf
# for more information on installing and configuring the Intel(R) FPGA SDK for OpenCL(TM).

ifeq ($(VERBOSE),1)
ECHO :=
else
ECHO := @
endif

# Where is the Intel(R) FPGA SDK for OpenCL(TM) software?
ifeq ($(wildcard $(INTELFPGAOCLSDKROOT)),)
$(error Set INTELFPGAOCLSDKROOT to the root directory of the Intel(R) FPGA SDK for OpenCL(TM) software installation)
endif
ifeq ($(wildcard $(INTELFPGAOCLSDKROOT)/host/include/CL/opencl.h),)
$(error Set INTELFPGAOCLSDKROOT to the root directory of the Intel(R) FPGA SDK for OpenCL(TM) software installation.)
endif

# OpenCL compile and link flags.
AOCL_COMPILE_CONFIG := $(shell aocl compile-config --arm)
#AOCL_LINK_CONFIG := $(shell aocl link-config --arm)
AOCL_LINK_CONFIG := $(wildcard $(INTELFPGAOCLSDKROOT)/host/arm32/lib/*.so) $(wildcard $(AOCL_BOARD_PACKAGE_ROOT)/arm32/lib/*.so)

# Compilation flags
ifeq ($(DEBUG),1)
CXXFLAGS += -g
else
CXXFLAGS += -O2
endif
# CXXFLAGS += -std=c++11

# Compiler. ARM cross-compiler.
CXX := arm-linux-gnueabihf-g++

# Target
TARGET := host_cl
TARGET_DIR := bin

# Directories
INC_DIRS :=  ../../../common/inc ../ ../../ .././backend .././mnist/
LIB_DIRS :=

# Files
INCS := $(wildcard )
SRCS := $(wildcard .././mnist/cl_utils.cpp  ../../../common/src/AOCLUtils/*.cpp)
LIBS := rt pthread

CFILES = $(wildcard ../../*.c .././backend/*.c ../../protobuf-c/*.c)
# CFILES += $(wildcard .././mnist/mnist.cpp)
CFILES += $(wildcard .././mnist/mnist_model.cpp)
# CFILES += $(wildcard .././parse/parse_test.c)
# CFILES += $(wildcard .././transpose/transpose_test.c .././backend/transpose.c .././backend/info.c)
OBJFILE = $(CFILES:.c=.o) $(SRCS:.cpp=.o)

# Make it all!
all : $(TARGET)


# Host executable target.
# $(TARGET) : $(OBJFILE)
# 	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $^ -fPIC \
# 	$(foreach D,$(INC_DIRS),-I$D) \
# 	$(AOCL_COMPILE_CONFIG) \
# 	$(SRCS) \
# 	$(AOCL_LINK_CONFIG) \
# 	$(foreach D,$(LIB_DIRS),-L$D) \
# 	$(foreach L,$(LIBS),-l$L) \
# 	-o $(TARGET)
$(TARGET) : $(OBJFILE)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $^ -fPIC \
	$(foreach D,$(INC_DIRS),-I$D) \
	$(AOCL_COMPILE_CONFIG) \
	$(AOCL_LINK_CONFIG) \
	$(foreach D,$(LIB_DIRS),-L$D) \
	$(foreach L,$(LIBS),-l$L) \
	-o $(TARGET)

%.o:%.cpp
	arm-linux-gnueabihf-g++ -fPIC $(CPPFLAGS) $(CXXFLAGS) $(AOCL_LINK_CONFIG) $(AOCL_COMPILE_CONFIG) $(foreach D,$(INC_DIRS),-I$D) $(foreach D,$(LIB_DIRS),-L$D) $(foreach L,$(LIBS),-l$L) -o $@ -c $<

%.o:%.c
	arm-linux-gnueabihf-gcc -fPIC -std=c99 $(foreach D,$(INC_DIRS),-I$D) -o $@ -c $<

$(TARGET_DIR) :
	$(ECHO)mkdir $(TARGET_DIR)

# Standard make targets
clean :
	$(ECHO)rm -f $(TARGET_DIR)/$(TARGET)
	rm -rf $(OBJFILE)

.PHONY : all clean
