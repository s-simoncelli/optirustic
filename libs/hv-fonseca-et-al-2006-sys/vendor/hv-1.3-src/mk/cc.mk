# -*- Makefile-gmake -*-

## Is $(CC) a gcc variant?
ifeq ($(shell sh -c "$(CC) -v 2>&1 | tail -1 | grep gcc >/dev/null && echo y"),y)
  $(info Detected C compiler to be a GCC variant.)
  include mk/gcc.mk
endif

# Is $(CC) the Sun C compiler?
ifeq ($(shell sh -c "$(CC) -V 2>&1 | head -1 | grep Sun >/dev/null && echo y"),y)
  $(info Detected C compiler to the Sun Studio C compiler.)
  include mk/suncc.mk
endif
