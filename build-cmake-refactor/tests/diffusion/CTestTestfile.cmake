# CMake generated Testfile for 
# Source directory: /Users/catherinepemblington/Documents/GitHub/Latient_Diffusion_NLP/tests/diffusion
# Build directory: /Users/catherinepemblington/Documents/GitHub/Latient_Diffusion_NLP/build-cmake-refactor/tests/diffusion
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test(diffusion_harness "/Users/catherinepemblington/Documents/GitHub/Latient_Diffusion_NLP/build-cmake-refactor/tests/diffusion/test_diffusion_harness")
set_tests_properties(diffusion_harness PROPERTIES  LABELS "p0;diffusion" _BACKTRACE_TRIPLES "/Users/catherinepemblington/Documents/GitHub/Latient_Diffusion_NLP/tests/diffusion/CMakeLists.txt;3;add_test;/Users/catherinepemblington/Documents/GitHub/Latient_Diffusion_NLP/tests/diffusion/CMakeLists.txt;0;")
add_test(noise_schedule "/Users/catherinepemblington/Documents/GitHub/Latient_Diffusion_NLP/build-cmake-refactor/tests/diffusion/test_noise_schedule")
set_tests_properties(noise_schedule PROPERTIES  LABELS "p0;diffusion" _BACKTRACE_TRIPLES "/Users/catherinepemblington/Documents/GitHub/Latient_Diffusion_NLP/tests/diffusion/CMakeLists.txt;9;add_test;/Users/catherinepemblington/Documents/GitHub/Latient_Diffusion_NLP/tests/diffusion/CMakeLists.txt;13;add_diffusion_test;/Users/catherinepemblington/Documents/GitHub/Latient_Diffusion_NLP/tests/diffusion/CMakeLists.txt;0;")
add_test(forward_diffusion "/Users/catherinepemblington/Documents/GitHub/Latient_Diffusion_NLP/build-cmake-refactor/tests/diffusion/test_forward_diffusion")
set_tests_properties(forward_diffusion PROPERTIES  LABELS "p0;diffusion" _BACKTRACE_TRIPLES "/Users/catherinepemblington/Documents/GitHub/Latient_Diffusion_NLP/tests/diffusion/CMakeLists.txt;9;add_test;/Users/catherinepemblington/Documents/GitHub/Latient_Diffusion_NLP/tests/diffusion/CMakeLists.txt;14;add_diffusion_test;/Users/catherinepemblington/Documents/GitHub/Latient_Diffusion_NLP/tests/diffusion/CMakeLists.txt;0;")
add_test(reverse_diffusion "/Users/catherinepemblington/Documents/GitHub/Latient_Diffusion_NLP/build-cmake-refactor/tests/diffusion/test_reverse_diffusion")
set_tests_properties(reverse_diffusion PROPERTIES  LABELS "p0;diffusion" _BACKTRACE_TRIPLES "/Users/catherinepemblington/Documents/GitHub/Latient_Diffusion_NLP/tests/diffusion/CMakeLists.txt;9;add_test;/Users/catherinepemblington/Documents/GitHub/Latient_Diffusion_NLP/tests/diffusion/CMakeLists.txt;15;add_diffusion_test;/Users/catherinepemblington/Documents/GitHub/Latient_Diffusion_NLP/tests/diffusion/CMakeLists.txt;0;")
