import tensorflow as tf

print(tf.test.is_built_with_cuda())
# return True
print("---------------------------")
print(tf.test.is_gpu_available(cuda_only=False, min_cuda_compute_capability=None))
# return True