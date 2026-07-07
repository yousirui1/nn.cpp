#include <stdarg.h>
#include "base.h"
#include "tools.h"
#include "log.h"

int init_log = 0;

static struct log_t programe_log = {NULL, NULL, LOG_DEBUG_LEVEL};
//static pthread_mutex_t mutex;

int init_logs(const char *log_dir, int log_level, const char *program_name)
{
    struct tm *t = NULL;;
    char c_dir[MAX_FILE_NAME_LEN];
    struct stat file_state;
    time_t current_time;
    int ret;

    memset(&programe_log, 0, sizeof(programe_log));

    programe_log.log_level = log_level;

    if(log_dir)
        strncpy(programe_log.log_dir, log_dir, sizeof(programe_log.log_dir));

    LOG_DEBUG("log_dir %s program_name %s", log_dir, program_name);

    (void)time(&current_time);
    t = localtime(&current_time);

    /* 检测文件夹是否存在 */
    ret = stat(log_dir, &file_state);
    if (ret < 0)
    {   
        if (errno == ENOENT)
        {
#ifdef _WIN32
            ret = mkdir(log_dir);
#else
            ret = mkdir(log_dir, 0755);
#endif
            if (ret < 0)
            {
                return ERROR;
            }
        }
    }   
    snprintf(c_dir, sizeof(c_dir), "%s/%s-%4.4d%2.2d%2.2d.log", log_dir, program_name, (1900 + t->tm_year), (1 + t->tm_mon), t->tm_mday);
    LOG_DEBUG("c_dir %s", c_dir);

    ret = stat(c_dir, &file_state);
    if (ret < 0)
    {   
        if (errno == ENOENT)
        {
            programe_log.fp_log = fopen(c_dir, "wb");
        }
    }   
    else
    {   
        programe_log.fp_log = fopen(c_dir, "ab");
    }   

    if (NULL == programe_log.fp_log)
    {   
        LOG_DEBUG("fopen log error ");
    }   

    sprintf(c_dir, "%s/%s-%4.4d%2.2d%2.2d-error.log", log_dir, program_name, (1900 + t->tm_year), (1 + t->tm_mon), t->tm_mday);

    ret = stat(c_dir, &file_state);
    if (ret < 0)
    {   
        if (errno == ENOENT)
        {
            programe_log.fp_err = fopen(c_dir, "wb");
        }
    }
    else
    {
        programe_log.fp_err = fopen(c_dir, "ab");
    }

    if (NULL == programe_log.fp_err)
    {
        LOG_DEBUG("fopen error %s  log error", c_dir);
    }
    init_log = 1;

    LOG_DEBUG("\n\n\n\n");
    return SUCCESS;
}

void deinit_logs()
{
    if (programe_log.fp_err)
        fclose(programe_log.fp_err);

    if (programe_log.fp_log)
        fclose(programe_log.fp_log);

    programe_log.fp_err = NULL;
    programe_log.fp_log = NULL;
    programe_log.log_level = 0;
}


void log_debug(const char *fmt, ...)
{   
    char buf[2048] = {0};
    char *ptr = buf;
    va_list ap;
    
    if(programe_log.log_level >= LOG_DEBUG_LEVEL)
    {
        va_start(ap, fmt); 
        vsprintf(ptr, fmt, ap);
        va_end(ap);
        printf("%s", buf);

        if(init_log && programe_log.fp_log)
        {
            fprintf(programe_log.fp_log, "%s:%s", get_commonlog_time(), buf);
            fflush(programe_log.fp_log);
        }
    }
}


void log_info(const char *fmt, ...)
{
    char buf[2048] = {0};
    char *ptr = buf;
    va_list ap;

    if(programe_log.log_level >= LOG_INFO_LEVEL)
    {
        va_start(ap, fmt);
        vsprintf(ptr, fmt, ap);
        va_end(ap);
        printf("%s", buf);
        
        if(init_log && programe_log.fp_log)
        {
            //pthread_mutex_lock(&mutex);
            fprintf(programe_log.fp_log, "%s:%s", get_commonlog_time(), buf);
            fflush(programe_log.fp_log);
            //pthread_mutex_unlock(&mutex);
        }
    }
}

void log_error(const char *fmt, ...)
{
    char buf[2048] = {0};
    char *ptr = buf;
    va_list ap;

    if(programe_log.log_level >= LOG_ERROR_LEVEL)
    {
        va_start(ap, fmt);
        vsprintf(ptr, fmt, ap);
        va_end(ap);
        printf("%s", buf);

        if(init_log && programe_log.fp_err)
        {
            //pthread_mutex_lock(&mutex);
            fprintf(programe_log.fp_err, "%s:%s", get_commonlog_time(), buf);
            fflush(programe_log.fp_err);

            fprintf(programe_log.fp_log, "%s:%s", get_commonlog_time(), buf);
            fflush(programe_log.fp_log);

            //pthread_mutex_unlock(&mutex);
        }
    }
}


