ARCH?=$(shell uname -m)
#--------------------------------Output------------------------------
# OUT_TYPE: 0-exe, 1-dll, 2-static
#--------------------------------------------------------------------
OUT_TYPE = 0
OUT_FILE = nn.cpp
ROOT=../
3RD_DIR=$(ROOT)/3rdparty

#-------------------------------Include------------------------------
#
# INCLUDES = $(addprefix -I,$(INCLUDES)) # add -I prefix
#--------------------------------------------------------------------
INCLUDES = $(ROOT)/base/include ./include \
		   $(3RD_DIR)/ggml/$(ARCH)/include  \
		   $(ROOT)/base/miniaudio 

#-------------------------------Source-------------------------------
#
#
#--------------------------------------------------------------------
SOURCE_PATHS = $(ROOT)/base/src src

SOURCE_FILES = $(foreach ext,cpp c cc,$(foreach dir,$(SOURCE_PATHS),$(wildcard $(dir)/*.$(ext))))

#-----------------------------Library--------------------------------
#
# LIB_PATHS = $(addprefix -L,$(LIBPATHS)) # add -L prefix
#--------------------------------------------------------------------
LIB_PATHS = $(3RD_DIR)/ggml/$(ARCH)/lib
ifdef RELEASE
# relase library path
LIB_PATHS +=
else
LIB_PATHS +=
endif

LIBS = ggml ggml-base ggml-cpu #ggml-cuda llama mtmd

STATIC_LIBS =

#-----------------------------DEFINES--------------------------------
#
# DEFINES := $(addprefix -D,$(DEFINES)) # add -L prefix
#--------------------------------------------------------------------
DEFINES =

include $(ROOT)/gcc.mk

install:
	
#
