if(WIN32)
set(MAT_TOP_DIR "C:\Program Files\MATLAB")
set(MATLAB_VERSION "R2018b")
endif(WIN32)

FIND_PATH(MAT_INCLUDE_DIR 
   mat.h
   ${MAT_TOP_DIR}/${MATLAB_VERSION}/extern/include
)

FIND_PATH(MAT_MX_INCLUDE_DIR
   matrix.h
   ${MAT_TOP_DIR}/${MATLAB_VERSION}/extern/include
)

FIND_LIBRARY(MAT_LIBRARY
   libmat 
   ${MAT_TOP_DIR}/${MATLAB_VERSION}/extern/lib/win64/microsoft
)

FIND_LIBRARY(MAT_MX_LIBRARY
   libmx
   ${MAT_TOP_DIR}/${MATLAB_VERSION}/extern/lib/win64/microsoft
)

if(MAT_INCLUDE_DIR AND MAT_MX_INCLUDE_DIR)
   set(MAT_FOUND TRUE)
endif(MAT_INCLUDE_DIR AND MAT_MX_INCLUDE_DIR)
	 
if(MAT_FOUND)
   if(NOT MAT_FIND_QUIETLY)
      message(STATUS "Found MAT: ${MAT_INCLUDE_DIR} and ${MAT_MX_INCLUDE_DIR}")
   endif(NOT MAT_FIND_QUIETLY)
else(MAT_FOUND)
   if(MAT_FIND_REQUIRED)
      message(FATAL_ERROR "could NOT find MATLAB")
   endif(MAT_FIND_REQUIRED)
endif(MAT_FOUND)

MARK_AS_ADVANCED(MAT_INCLUDE_DIR)
MARK_AS_ADVANCED(MAT_MX_INCLUDE_DIR)


