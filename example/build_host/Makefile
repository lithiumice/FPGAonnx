# Compiler. 
CXX := g++

# Target
TARGET := host
TARGET_DIR := .

# Directories
INC_DIRS := ../host/src/nnom/inc ../../common/inc
LIB_DIRS :=

# Files
INCS := $(wildcard )
SRCS := $(wildcard ../host/src/*.cpp)
LIBS := 

CFILES = $(wildcard ../host/src/nnom/src/core/*.c ../host/src/nnom/src/layers/*.c ../host/src/nnom/src/backends/*.c)
OBJFILE = $(CFILES:.c=.o)

# Make it all!
all : $(TARGET)

# Host executable target.
$(TARGET) : $(OBJFILE)
	$(CXX)  -g $^ -fPIC $(SRCS) -o $(TARGET) \
	$(foreach D,$(INC_DIRS),-I$D) \
	$(foreach D,$(LIB_DIRS),-L$D) \
	$(foreach L,$(LIBS),-l$L) \
	
%.o:%.c
	gcc -g -fPIC -std=c99 $(foreach D,$(INC_DIRS),-I$D) -o $@ -c $<

clean :
	rm -f $(TARGET).exe
	rm -rf $(OBJFILE)

.PHONY : all clean
