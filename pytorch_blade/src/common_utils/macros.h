#ifndef __COMMON_MACROS_H__
#define __COMMON_MACROS_H__

#define DISALLOW_COPY_AND_ASSIGN(TypeName) \
  TypeName(const TypeName&) = delete;      \
  void operator=(const TypeName&) = delete;

#define TorchBladeDeclNewFlag(T, Name) \
  T Set##Name(T);                       \
  T& Get##Name();

#define TorchBladeDefNewFlag(T, Name) \
  T& Get##Name() {                     \
    thread_local T flag;               \
    return flag;                       \
  }                                    \
  T Set##Name(T flag) {                \
    T old_flag = Get##Name();          \
    Get##Name() = flag;                \
    return old_flag;                   \
  }

#endif //__COMMON_MACROS_H__