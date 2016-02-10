# - Try to find Shark
# Once done this will define
#
#  SHARK_FOUND - system has Shark installed
#  SHARK_INCLUDE_DIR - the Shark include directory
#  SHARK_LIBRARIES - Link these to use Shark

# Copyright (c) 2006, Alexander Neundorf, <neundorf@kde.org>
# Copyright (c) 2014, Aydin Demircioglu, <hello@aydindemircioglu.de>
#
# Redistribution and use is allowed according to the terms of the BSD license.
# For details see the accompanying COPYING-CMAKE-SCRIPTS file.

# This file is based on cmake-2.6/Modules/FindBZip2.cmake
# Copyright (c) 2010, Yiannis Belias, <jonnyb@hol.gr>


IF (SHARK_INCLUDE_DIRS AND SHARK_LIBRARIES)
    SET(SHARK_FIND_QUIETLY TRUE)
ENDIF (SHARK_INCLUDE_DIRS AND SHARK_LIBRARIES)

FIND_PATH(SHARK_INCLUDE_DIRS shark/Core/Shark.h 
    PATHS /usr/include
    PATHS /home/xboy01/Shark/usr/local/include
    PATHS /usr/local/include)

FIND_LIBRARY(SHARK_LIBRARIES 
    PATHS /home/xboy01/Shark/usr/local/lib
    NAMES libshark.a )

# handle the QUIETLY and REQUIRED arguments and set SHARK_FOUND to TRUE if 
# all listed variables are TRUE
INCLUDE(FindPackageHandleStandardArgs)
FIND_PACKAGE_HANDLE_STANDARD_ARGS(SHARK DEFAULT_MSG SHARK_LIBRARIES SHARK_INCLUDE_DIRS)


IF(SHARK_FOUND)
  SET(SHARK_LIBRARIES ${SHARK_LIBRARY})
  SET(SHARK_INCLUDE_DIRS ${SHARK_INCLUDE_DIR})
ENDIF(SHARK_FOUND)
