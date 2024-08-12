#-----------------------------------------------------------------------
#  Makefile for hv

VERSION = 1.3

#----------------------------------------------------------------------
#
#                       Copyright (c) 2010
#                  Carlos M. Fonseca <cmfonsec@ualg.pt>
#             Manuel Lopez-Ibanez <manuel.lopez-ibanez@ulb.ac.be>
#                    Luis Paquete <paquete@dei.uc.pt>
#
# This program is free software (software libre); you can redistribute
# it and/or modify it under the terms of the GNU General Public
# License as published by the Free Software Foundation; either 
# version 2 of the License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, you can obtain a copy of the GNU
# General Public License at:
#                  http://www.gnu.org/copyleft/gpl.html
# or by writing to:
#            Free Software Foundation, Inc., 59 Temple Place,
#                  Suite 330, Boston, MA 02111-1307 USA
#
#-----------------------------------------------------------------------

## Default to no debugging info
DEBUG ?= 0

## Quiet / verbose output:
ifneq ($(findstring $(MAKEFLAGS),s),s)
  ifdef S 
    QUIET_CC   = @echo '   ' CC $@;
    QUIET_AR   = @echo '   ' AR $@;
    QUIET_LINK = @echo '   ' LINK $@;
    QUIET_RM   = @echo '   ' RM $@;
    ECHO       = 
  else
    ECHO       = @echo "$(1)"
  endif
endif

## Deal with Windows:
ifdef OS
  ifeq ($(OS),Windows_NT)
    uname_S = Windows
    ## Technically we could be running under Win64!
    uname_M = i386
  else
    $(error Unknown operating system $(OS) detected!)
  endif
else
  ## Detect system and machine type:
  uname_S = $(shell sh -c 'uname -s 2>/dev/null || echo unknown')
  uname_M = $(shell sh -c 'uname -m 2>/dev/null || echo unknown')
  ## Do we have svnversion?
  ifeq ($(shell sh -c 'which svnversion 1> /dev/null 2>&1 && echo y'),y)
    ## Is this a working copy?
    ifneq ($(shell sh -c 'LC_ALL=C svnversion -n .'),exported)
      $(shell sh -c 'svnversion -n . > svn_version')
    endif
  endif
endif

## Set version information:
SVN_REV = $(shell sh -c 'cat svn_version 2> /dev/null')

## Define source files
SRCS  = main-hv.c io.c timer.c
HDRS  = io.h timer.h
OBJS  = $(SRCS:.c=.o)

DIST_SRC_FILES = Makefile Makefile.lib mk/README mk/*.mk \
		 README LICENSE \
		 Hypervolume_MEX.c svn_version \
		 $(SRCS) $(HDRS) \
		 $(HV_SRCS) $(HV_HDRS)

DIST_SRC       = hv-$(VERSION)-src

################################################################################
## Configure the compiler / linker:

## Global list of CPP flags
CPPFLAGS = -D DEBUG=$(DEBUG) -D VERSION='"$(VERSION)"'

ifneq ($(DEBUG), 0)
CPPFLAGS += -DMALLOC_CHECK_=3
endif
ifneq ($(uname_S),Cygwin)
CPPFLAGS += -D_GNU_SOURCE
else
CPPFLAGS += -U_GNU_SOURCE
endif 

ifdef march
MARCH=$(march)
endif

## Matlab extension compiler
MEX = mex

## Define optimizing CFLAGS based on compiler and operating system
ifndef OPT_CFLAGS
  -include mk/$(uname_S)_$(uname_M)_$(CC).mk
  ## Include failed or could not find optimizing CFLAGS, try compiler include
  ifndef OPT_CFLAGS
    -include mk/$(CC).mk
    ## Still no OPT_CFLAGS, see if gcc variant
    ifndef OPT_CFLAGS
      ifneq ($(findstring gcc,$(CC)),)
	$(warning Unknown C compiler. Assuming a GCC variant.)
	-include mk/gcc.mk
      endif
    endif
  endif
endif

ifeq ($(DEBUG),0)
  ifndef OPT_CFLAGS
    $(error No optimizing CFLAGS set. Please manually specify OPT_CFLAGS. \
Alternatively you can create a file named 'mk/$(uname_S)_$(uname_M)_$(CC).mk' \
and place all compiler flag configuration directives in this file)
  endif
endif

ifdef ARCH
CPPFLAGS += -DARCH='"$(ARCH)"'
endif

## Collect all flags for compiler in one variable
ALL_CFLAGS  = $(CPPFLAGS) $(CFLAGS) $(OPT_CFLAGS)
ALL_LDFLAGS = $(LDFLAGS) $(OPT_LDFLAGS)

#----------------------------------------------------------------------
.PHONY: all clean dist test default mex
.NOTPARALLEL:
#----------------------------------------------------------------------
default: hv

all: clean hv

clean:
	$(call ECHO,---> Removing hv <---)
	@$(RM) hv
	$(call ECHO,---> Removing object files <---)
	@$(RM) $(OBJS) $(HV_OBJS)
	$(call ECHO,---> Removing $(HV_LIB) <---)
	@$(RM) $(HV_LIB)
	$(call ECHO,---> Removing backup files <---)
	@$(RM) *~

dist : DEBUG=0
dist : CDEBUG=
dist : all
	@(rm -f ../$(DIST_SRC).tar.gz && mkdir -p ../$(DIST_SRC) \
	&& rsync -rlpC --relative --exclude=.svn $(DIST_SRC_FILES) ../$(DIST_SRC)/ \
	&& cd .. \
	&& tar cf - $(DIST_SRC) | gzip -f9 > $(DIST_SRC).tar.gz \
	&& rm -rf ./$(DIST_SRC)/* && rmdir ./$(DIST_SRC)/ \
	&& echo "$(DIST_SRC).tar.gz created." && cd $(PWD) )

test: all
	@if test -d ../test; then  		                 \
	    cd ../test/ && ./regtest.pl $(PWD)/hv && cd $(PWD);  \
	else                                                     \
	    echo "Error: Testsuite not found in $(PWD)/../test/";\
            exit 1;                                              \
	fi

#----------------------------------------------------------------------
# Targets:
hv: main-hv.o timer.o io.o fpli_hv.a
	$(call ECHO,---> Building $@ version $(VERSION) <---)
	$(QUIET_LINK)$(CC) $(ALL_LDFLAGS) -o $@ $^

hv.ps: hv.c
	a2ps -E -g -o hv.ps hv.c

#----------------------------------------------------------------------
# Rules:
%.o: %.c
	$(QUIET_CC)$(CC) -o $*.o -c $(ALL_CFLAGS) $<

#----------------------------------------------------------------------
# Include actual HV code:
include Makefile.lib

#----------------------------------------------------------------------
# Dependencies:
main-hv.o: $(HV_HDRS) timer.h io.h
timer.o: timer.h
io.o: io.h

mex: Hypervolume_MEX.c $(HV_SRCS)
	$(MEX) $(MEXFLAGS) -DVARIANT=$(VARIANT) $^
