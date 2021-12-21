#ifndef DYN_SORT_ATOMIC_H_
#define DYN_SORT_ATOMIC_H_

__device__ inline float atomicMax(float* address, float val) {
  int* address_as_i = (int*)address;
  int old = *address_as_i, assumed;
  do {
    assumed = old;
    old = ::atomicCAS(address_as_i, assumed,
                      __float_as_int(::fmaxf(val, __int_as_float(assumed))));
  } while (assumed != old);
  return __int_as_float(old);
}

__device__ inline void atomicMax(float* keyAddr, float key, unsigned* valAddr,
                                 unsigned val, float* oldK, unsigned* oldV) {
  int* address_as_i = (int*)keyAddr;
  int old_key = *address_as_i, assumed_key;
  unsigned old_val = *valAddr, assumed_val, new_val;
  do {
    assumed_key = old_key;
    assumed_val = old_val;
    new_val = key > __int_as_float(assumed_key) ? val : assumed_val;

    old_key =
        ::atomicCAS(address_as_i, assumed_key,
                    __float_as_int(::fmaxf(key, __int_as_float(assumed_key))));
    old_val = ::atomicCAS(valAddr, assumed_val, new_val);
  } while (assumed_key != old_key || assumed_val != old_val);
  *oldK = __int_as_float(old_key);
  *oldV = old_val;
}

#endif  // DYN_SORT_ATOMIC_H_
