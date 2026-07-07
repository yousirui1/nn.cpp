#ifndef __LOG_H__
#define __LOG_H__

#ifdef __cplusplus
extern "C" {
#endif


#define LOG_NONE_LEVEL     0
#define LOG_ERROR_LEVEL    1
#define LOG_INFO_LEVEL     2
#define LOG_DEBUG_LEVEL    3
#define LOG_DUMP_LEVEL     4

struct log_t
{
    FILE *fp_err;
    FILE *fp_log;
    int log_level;
    char log_dir[MAX_FILE_NAME_LEN];
};

int init_logs(const char *log_dir, int log_level, const char *program_name);
void deinit_logs();
void log_debug(const char *fmt, ...);
void log_info(const char *fmt, ...);
void log_error(const char *fmt, ...);

#ifdef __cplusplus
}
#endif



#ifndef LOG_LEVEL
#define LOG_LEVEL    LOG_DUMP_LEVEL //4
#endif  //LOG_LEVEL


#if LOG_LEVEL >= LOG_ERROR_LEVEL
#define LOG_ERROR(fmt,...) \
        do { \
        log_error("[ERROR] (%s:%d fun:%s) " fmt "\r\n", __FILE__, __LINE__, __FUNCTION__, ##__VA_ARGS__); \
        }while(0)
#else
#define LOG_ERROR(fmt,...) 
#endif

#if LOG_LEVEL >= LOG_INFO_LEVEL
#define LOG_INFO(fmt,...) \
        do { \
            log_info("[INFO] (%s:%d fun:%s) " fmt "\r\n", __FILE__, __LINE__, __FUNCTION__, ##__VA_ARGS__);\
        }while(0)
#else
#define LOG_INFO(fmt,...) 
#endif

#if LOG_LEVEL >= LOG_DEBUG_LEVEL
#define LOG_DEBUG(fmt,...) \
        do { \
            log_debug("[DEBUG] (%s:%d fun:%s) " fmt "\r\n", __FILE__, __LINE__, __FUNCTION__, ##__VA_ARGS__);\
        }while(0)
#else
#define LOG_DEBUG(fmt,...)
#endif


#endif //  __LOG_H__
