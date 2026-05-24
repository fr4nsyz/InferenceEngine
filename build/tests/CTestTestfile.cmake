# CMake generated Testfile for 
# Source directory: /home/fr4nsyz/vault/L_CACHES/InferenceEngine/tests
# Build directory: /home/fr4nsyz/vault/L_CACHES/InferenceEngine/build/tests
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test(test_tensor "/home/fr4nsyz/vault/L_CACHES/InferenceEngine/build/tests/test_tensor")
set_tests_properties(test_tensor PROPERTIES  _BACKTRACE_TRIPLES "/home/fr4nsyz/vault/L_CACHES/InferenceEngine/tests/CMakeLists.txt;10;add_test;/home/fr4nsyz/vault/L_CACHES/InferenceEngine/tests/CMakeLists.txt;0;")
add_test(test_linear "/home/fr4nsyz/vault/L_CACHES/InferenceEngine/build/tests/test_linear")
set_tests_properties(test_linear PROPERTIES  _BACKTRACE_TRIPLES "/home/fr4nsyz/vault/L_CACHES/InferenceEngine/tests/CMakeLists.txt;14;add_test;/home/fr4nsyz/vault/L_CACHES/InferenceEngine/tests/CMakeLists.txt;0;")
add_test(test_softmax "/home/fr4nsyz/vault/L_CACHES/InferenceEngine/build/tests/test_softmax")
set_tests_properties(test_softmax PROPERTIES  _BACKTRACE_TRIPLES "/home/fr4nsyz/vault/L_CACHES/InferenceEngine/tests/CMakeLists.txt;18;add_test;/home/fr4nsyz/vault/L_CACHES/InferenceEngine/tests/CMakeLists.txt;0;")
add_test(test_relu "/home/fr4nsyz/vault/L_CACHES/InferenceEngine/build/tests/test_relu")
set_tests_properties(test_relu PROPERTIES  _BACKTRACE_TRIPLES "/home/fr4nsyz/vault/L_CACHES/InferenceEngine/tests/CMakeLists.txt;22;add_test;/home/fr4nsyz/vault/L_CACHES/InferenceEngine/tests/CMakeLists.txt;0;")
add_test(test_add "/home/fr4nsyz/vault/L_CACHES/InferenceEngine/build/tests/test_add")
set_tests_properties(test_add PROPERTIES  _BACKTRACE_TRIPLES "/home/fr4nsyz/vault/L_CACHES/InferenceEngine/tests/CMakeLists.txt;26;add_test;/home/fr4nsyz/vault/L_CACHES/InferenceEngine/tests/CMakeLists.txt;0;")
add_test(test_onnx_loader "/home/fr4nsyz/vault/L_CACHES/InferenceEngine/build/tests/test_onnx_loader")
set_tests_properties(test_onnx_loader PROPERTIES  WORKING_DIRECTORY "/home/fr4nsyz/vault/L_CACHES/InferenceEngine" _BACKTRACE_TRIPLES "/home/fr4nsyz/vault/L_CACHES/InferenceEngine/tests/CMakeLists.txt;30;add_test;/home/fr4nsyz/vault/L_CACHES/InferenceEngine/tests/CMakeLists.txt;0;")
subdirs("../_deps/googletest-build")
