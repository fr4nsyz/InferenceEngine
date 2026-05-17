# CMake generated Testfile for 
# Source directory: /home/fr4nsyz/vault/L_CACHES/InferenceEngine/tests
# Build directory: /home/fr4nsyz/vault/L_CACHES/InferenceEngine/build/tests
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test(test_matrix "/home/fr4nsyz/vault/L_CACHES/InferenceEngine/build/tests/test_matrix")
set_tests_properties(test_matrix PROPERTIES  _BACKTRACE_TRIPLES "/home/fr4nsyz/vault/L_CACHES/InferenceEngine/tests/CMakeLists.txt;10;add_test;/home/fr4nsyz/vault/L_CACHES/InferenceEngine/tests/CMakeLists.txt;0;")
add_test(test_linear "/home/fr4nsyz/vault/L_CACHES/InferenceEngine/build/tests/test_linear")
set_tests_properties(test_linear PROPERTIES  _BACKTRACE_TRIPLES "/home/fr4nsyz/vault/L_CACHES/InferenceEngine/tests/CMakeLists.txt;14;add_test;/home/fr4nsyz/vault/L_CACHES/InferenceEngine/tests/CMakeLists.txt;0;")
subdirs("../_deps/googletest-build")
