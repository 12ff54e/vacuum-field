# VfieldWarnings.cmake — target-scoped warning configuration.
#
# `vfield_enable_warnings(<target>)` applies the library warning set to a
# single target. Keeping this per-target (rather than in CMAKE_CXX_FLAGS) is
# what lets host-only `.cpp` targets and CUDA `.cu` targets use different
# warning levels.

function(vfield_enable_warnings target)
  target_compile_options(${target} PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra>
    $<$<COMPILE_LANGUAGE:CUDA>:-Wall -Wextra>)
  if(VFIELD_WARNINGS_AS_ERRORS)
    target_compile_options(${target} PRIVATE
      $<$<COMPILE_LANGUAGE:CXX>:-Werror>
      $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Werror>)
  endif()
endfunction()
