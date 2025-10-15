#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>
#include <string.h>

void print_memory_usage(pid_t pid) {
    char path[64];
    snprintf(path, sizeof(path), "/proc/%d/status", pid);

    FILE *file = fopen(path, "r");
    if (!file) {
        return;
    }

    char line[256];
    while (fgets(line, sizeof(line), file)) {
        // VmRSS: Resident Set Size (actual memory usage)
        if (strncmp(line, "VmRSS:", 6) == 0)
            printf("[PID %d] %s", pid, line);
        if (strncmp(line, "RssAnon:", 8) == 0)
            printf("[PID %d] %s", pid, line);
        if (strncmp(line, "RssFile:", 8) == 0)
            printf("[PID %d] %s", pid, line);
        if (strncmp(line, "RssShmem:", 9) == 0)
            printf("[PID %d] %s", pid, line);
    }
    printf("\n");
    fclose(file);
}

int main(int argc, char** argv, char** envp) {
    pid_t pid = fork();

    char** argv_copy = malloc(sizeof(char*) * argc);
    for (int i = 0; i < argc - 1; ++i)
        argv_copy[i] = argv[i + 1];

    if (pid == 0) {
        if (execve(argv[1], argv_copy, envp) < 0) {
            perror("execve");
            return 1;
        }
        exit(18);
    } else if (pid > 0) {
        printf("[Parent] Checking child memory usage...\n");
        int child_status;
        while (waitpid(pid, &child_status, WNOHANG) >= 0) {
            print_memory_usage(pid);
            usleep(1000*1000);
        }

        wait(NULL);  // 자식 종료 기다림
    } else {
        perror("fork");
        return 1;
    }

    free((void*)argv_copy);

    return 0;
}
