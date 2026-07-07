#ifndef __TOOLS_H__
#define __TOOLS_H__

char *get_commonlog_time(void);
long long get_ustime(void);
char* get_time(time_t last_time, char *year_mon_day, char *hour_min_sec);

int write_file_data(const char *path, uint8_t *file_data, uint64_t file_size);
uint64_t read_file_data(const char *path, uint8_t **file_data);
char *read_line(FILE *fp, char *buf, int *len);
int read_int32(const char *file_path, int32_t *result, int max_line,  int *valid_number);
uint64_t read_wav_file(const char *path, void **file_data, int channel, int *sample_width, int *sample_rate);
int write_wav_file(const char *path, void *file_data, uint64_t file_size, int sample_width, int sample_rate);

uint64_t get_file_line(const char *path);
const char* get_file_ext(const char *filename);

/* cpu 占用等待 */
void busy_wait_milliseconds(uint32_t millis);
void sleep_milliseconds(uint32_t millis);
/* 最高系统调度不切换线程 */
void set_max_priority(void);
/* 设置默认线程调度 */
void set_default_priority(void);


#endif //__TOOLS_H__
