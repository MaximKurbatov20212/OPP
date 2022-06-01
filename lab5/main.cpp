#include <iostream>
#include <mpi/mpi.h>
#include <pthread.h>
#include <stdlib.h>
#include <cmath>
#include <cstring>
#include <unistd.h>
#include <queue>

#define ERROR 1
#define SUCCESS 0 

#define EXECUTOR_FINISHED_WORK -1

#define SENDING_TASKS 103 
#define SENDING_TASK_COUNT 102 
#define GIVE_ME_TASKS 100

#define HAVE_NO_TASKS -1
#define ITERATION_ENDED -2
#define SEND_AGAIN -2 
#define FINISH 104

int number_of_tasks = 100;
int number_of_lists = 5;

bool tasks_have_been_recieved = false;
int rank = 0;
int size = 0;

std::queue<long long> tasks;

int global_iteration = 0;
double local_result = 0;
double global_result = 0;
int is_end = false;

pthread_mutex_t mutex;
pthread_mutexattr_t mutex_attr;

double doing(int repeat_num) {
    int result = 0;
    for(int i = 0; i < repeat_num; i++) {
        result += pow(sin((double) i) * cos((double) i), i);
    }
    return result;
}

void get_tasks() {
    if(rank != 0) return;
    pthread_mutex_lock(&mutex);
    for(int i = 0; i < number_of_tasks; i++) {
        tasks.push(abs(abs(50 - i % 100) *  abs(rank - (10 % size)) * 100));
    }
    pthread_mutex_unlock(&mutex);
}

void execute_tasks() {
    while(true) {
        pthread_mutex_lock(&mutex);
        if(tasks.empty()) {
            pthread_mutex_unlock(&mutex);
            return;
        }

        local_result += doing(tasks.front());
        tasks.pop();
        pthread_mutex_unlock(&mutex);
    }
}

int get_additional_tasks() {
    int new_task;
    bool send_again = false;

    for(int i = 0; i < size; i++) {
        if(rank == i) continue;

        MPI_Send(&rank, 1, MPI_INT, i, GIVE_ME_TASKS, MPI_COMM_WORLD);

        MPI_Recv(&new_task, 1, MPI_INT, i, SENDING_TASK_COUNT, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        if(new_task == HAVE_NO_TASKS) continue;

        if(new_task == SEND_AGAIN) {
            send_again = true;
            continue;
        }

        return new_task;
    }

    if(send_again) return SEND_AGAIN;
    return HAVE_NO_TASKS;
}

void stop_reciever() {
    is_end = true;
    int end = EXECUTOR_FINISHED_WORK;
    MPI_Send(&end, 1, MPI_INT, rank, GIVE_ME_TASKS, MPI_COMM_WORLD);
    printf("Job completed successfully\n");
}

void* execute(void* a) {
    for(int i = 0; i < number_of_lists; i++) {
        if(rank == 0) printf("\nIteration: %d\n", i);

        get_tasks();

        pthread_mutex_lock(&mutex);
        tasks_have_been_recieved = true;
        pthread_mutex_unlock(&mutex);

        printf("[%d] tasks_have_been_recieved\n", rank);
        double start = MPI_Wtime();
        execute_tasks();
        printf("[%d] I completed my tasks \n", rank);

        while(true) {
            int new_task = 0;

            new_task = get_additional_tasks();

            if(new_task == SEND_AGAIN) continue;
            if(new_task == HAVE_NO_TASKS) break;

            printf("[%d] Got additional task %d \n", rank, new_task);

            pthread_mutex_lock(&mutex);
            tasks.push(new_task);
            pthread_mutex_unlock(&mutex);
            execute_tasks();
        }
        printf("rank = %d: time = %f\n", rank, MPI_Wtime() - start);

        MPI_Barrier(MPI_COMM_WORLD);
        pthread_mutex_lock(&mutex);
        tasks_have_been_recieved = false;
        pthread_mutex_unlock(&mutex);
        global_iteration++;
    }

    stop_reciever();
    MPI_Allreduce(&local_result, &global_result, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    pthread_exit(nullptr);
}

int receive_request() {
    int request = 0;
    MPI_Recv(&request, 1, MPI_INT, MPI_ANY_SOURCE, GIVE_ME_TASKS, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    return request;
}

bool has_free_tasks() {
    return !(tasks.empty());
}

void* receive_msg(void* a) {
    while(!is_end) {
        int id = receive_request();
        if(id == EXECUTOR_FINISHED_WORK) break;

        pthread_mutex_lock(&mutex);                
        // if(has_free_tasks() && !tasks_have_been_recieved) {
        //     MPI_Abort(MPI_COMM_WORLD, 0);
        // }
        if(has_free_tasks()) {
            printf("[%d] Send task to %d \n", rank, id);
            MPI_Send(&tasks.front(), 1, MPI_INT, id, SENDING_TASK_COUNT, MPI_COMM_WORLD);
            tasks.pop();
        }
        else if(!tasks_have_been_recieved) {
            printf("[%d] I haven't received tasks yet, try again...\n", rank);
            int response = SEND_AGAIN;
            MPI_Send(&response, 1, MPI_INT, id, SENDING_TASK_COUNT, MPI_COMM_WORLD);
        }

        else {
            printf("[%d] I have no tasks \n", rank);
            int response = HAVE_NO_TASKS;
            MPI_Send(&response, 1, MPI_INT, id, SENDING_TASK_COUNT, MPI_COMM_WORLD);
        }
        pthread_mutex_unlock(&mutex);                
    }
    pthread_exit(nullptr);
}

void main_thread_wait(pthread_t* threads) {
    pthread_join(threads[0], nullptr);
    pthread_join(threads[1], nullptr);
}

int main(int argc, char** argv) {

    int mpi_init_tread_result;

    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &mpi_init_tread_result);

    if(mpi_init_tread_result != MPI_THREAD_MULTIPLE) {
        MPI_Finalize();
        return ERROR;
    }

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    pthread_mutex_init(&mutex, &mutex_attr);

    pthread_t threads[2];
    pthread_attr_t attr;
    pthread_attr_init(&attr);

    pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);
    
    pthread_create(&threads[0], &attr, execute, nullptr);
    pthread_create(&threads[1], &attr, receive_msg, nullptr);

    main_thread_wait(threads);

    pthread_attr_destroy(&attr);
    pthread_mutex_destroy(&mutex);
    
    MPI_Finalize();
    return SUCCESS;
}
