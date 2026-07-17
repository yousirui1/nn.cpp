#ifndef __BASE_H__
#define __BASE_H__

#include <stdio.h>
#include <errno.h>
#include <time.h>

#include <ctype.h>
#include <signal.h>

#include <fcntl.h>
#include <math.h>
#include <stdlib.h>
#include <memory.h>
#include <stdint.h>
#include <stdbool.h>
#include <sys/stat.h>
#include <float.h>
#include <stdarg.h>

#ifdef _WIN32
    #include <windows.h>
    //#include <winsock2.h>
    //#include <ws2tcpip.h>    
#else
    #include <sys/syscall.h>
    #include <sys/epoll.h>
    #include <sys/types.h>
    #include <sys/socket.h>
    #include <sys/wait.h>
    #include <sys/utsname.h>
    #include <sys/resource.h>
    #include <sys/utsname.h>
    #include <sys/ioctl.h>
    #include <sys/time.h>

    #include <netinet/tcp.h>
    #include <netinet/in.h>
    #include <arpa/inet.h>
    #include <netdb.h>
    #include <net/if.h>
	#include <sys/vfs.h>
	#include <assert.h>
    #include <unistd.h>
    #include <pthread.h>
#endif

#ifdef _WIN32
	#define FD_SETSIZE 1024
	#define usleep(x) Sleep((x)/1000)
	#define snprintf _snprintf
#endif  //WIN32

#define SUCCESS 0
#define ERROR -1

#define MAX_NAME_LEN 128
#define MAX_BUF_LEN	(1024 * 1024)
#define MAX_FILE_NAME_LEN 256
#define MAX_HADDR_LEN	16
#define MAX_IP_LEN 64

#define STACK_SIZE 100
#define gettid() syscall(__NR_gettid)

#define STRPREFIX(a,b) (strncmp((a),(b),strlen((b))) == 0)

#define BSWAP_8(x) ((x) & 0xff)
#define BSWAP_16(x) ((BSWAP_8(x) << 8) | BSWAP_8((x) >> 8))
#define BSWAP_32(x) ((BSWAP_16(x) << 16) | BSWAP_16((x) >> 16))
#define BSWAP_64(x) ((BSWAP_32(x) << 32) | BSWAP_32((x) >> 32))

#define ALIGN(x, a) (((x) + ((a) - 1)) & ~((a) - 1))

#define align_16(x) ALIGN(x,16)
#define align_4(x)  ALIGN(x,4)
#define align_2(x)  ALIGN(x,2)

#define ARRAY_SIZE(x) (sizeof(x) / sizeof((x)[0]))

#ifndef offsetof
#define offsetof(type, member) ((long) &((type *) 0)->member)
#endif  //offsetof

#define PERF_ALIGN(x, a)    __PERF_ALIGN_MASK(x, (typeof(x))(a)-1)
#define __PERF_ALIGN_MASK(x, mask)  (((x)+(mask))&~(mask))

#ifndef container_of
/**
 * container_of - cast a member of a structure out to the containing structure
 * @ptr:    the pointer to the member.
 * @type:   the type of the container struct this is embedded in.
 * @member: the name of the member within the struct.
 *
 */
#define container_of(ptr, type, member) ({          \
    const typeof(((type *)0)->member) * __mptr = (ptr); \
    (type *)((char *)__mptr - offsetof(type, member)); })
#endif

#ifndef __cplusplus
#ifndef max
#define max(x, y) ({                \
    typeof(x) _max1 = (x);          \
    typeof(y) _max2 = (y);          \
    (void) (&_max1 == &_max2);      \
    _max1 > _max2 ? _max1 : _max2; })
#endif

#ifndef min
#define min(x, y) ({                \
    typeof(x) _min1 = (x);          \
    typeof(y) _min2 = (y);          \
    (void) (&_min1 == &_min2);      \
    _min1 < _min2 ? _min1 : _min2; })
#endif
#endif //__cplusplus

#ifndef roundup
#define roundup(x, y) (                                \
{                                                      \
    const typeof(y) __y = y;               \
    (((x) + (__y - 1)) / __y) * __y;           \
}                                                      \
)
#endif

// 宏定义，用于将变量名转换为字符串
#define STRINGIFY(x) #x


#if 0
#ifndef __cplusplus
#define max(a,b)    (((a) > (b)) ? (a) : (b))
#define min(a,b)    (((a) < (b)) ? (a) : (b))
#endif //__cplusplus
#endif

#ifdef __GNUC__
#define EXPORT __attribute__((visibility("default")))
#elif _WIN32
#define EXPORT __declspec(dllimport)
#endif

#include "log.h"

#endif //__BASE_H__
