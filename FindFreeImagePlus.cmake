#
# Find FreeImage
#
# Try to find FreeImage.
# This module defines the following variables:
# - FREEIMAGE_INCLUDE_DIRS
# - FREEIMAGE_LIBRARIES
# - FREEIMAGE_FOUND
#
# The following variables can be set as arguments for the module.
# - FREEIMAGE_ROOT_DIR : Root library directory of FreeImage
#

# Additional modules
include(FindPackageHandleStandardArgs)

if (WIN32)
    # Find include files
    find_path(
            FREEIMAGEPLUS_INCLUDE_DIR
            NAMES FreeImagePlus.h
            PATHS
            $ENV{PROGRAMFILES}/include
            ${FREEIMAGE_ROOT_DIR}/include
            DOC "The directory where FreeImage.h resides")

    # Find library files
    find_library(
            FREEIMAGEPLUS_LIBRARY
            NAMES FreeImagePlus
            PATHS
            $ENV{PROGRAMFILES}/lib
            ${FREEIMAGE_ROOT_DIR}/lib)
else()
    # Find include files
    find_path(
            FREEIMAGEPLUS_INCLUDE_DIR
            NAMES FreeImagePlus.h
            PATHS
            /usr/include
            /usr/local/include
            /sw/include
            /opt/local/include
            DOC "The directory where FreeImage.h resides")

    # Find library files
    find_library(
            FREEIMAGEPLUS_LIBRARY
            NAMES freeimage
            PATHS
            /usr/lib64
            /usr/lib
            /usr/local/lib64
            /usr/local/lib
            /sw/lib
            /opt/local/lib
            ${FREEIMAGEPLUS_ROOT_DIR}/lib
            DOC "The FreeImage library")
endif()

# Handle REQUIRD argument, define *_FOUND variable
find_package_handle_standard_args(FreeImagePlus DEFAULT_MSG FREEIMAGEPLUS_INCLUDE_DIR FREEIMAGEPLUS_LIBRARY)

# Define GLFW_LIBRARIES and GLFW_INCLUDE_DIRS
if (FREEIMAGE_FOUND)
    set(FREEIMAGEPLUS_LIBRARIES ${FREEIMAGEPLUS_LIBRARY})
    set(FREEIMAGEPLUS_INCLUDE_DIRS ${FREEIMAGEPLUS_INCLUDE_DIR})
endif()

# Hide some variables
mark_as_advanced(FREEIMAGEPLUS_INCLUDE_DIR FREEIMAGEPLUS_LIBRARY)
